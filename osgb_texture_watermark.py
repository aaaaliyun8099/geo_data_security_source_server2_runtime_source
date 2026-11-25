import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import threading
from typing import NamedTuple

from PIL import Image

from .texture_watermark import (
    embed_texture_watermark,
    extract_texture_watermark,
    embed_texture_watermark_adaptive,
    extract_texture_watermark_adaptive,
    embed_texture_watermark_adaptive_jpeg,
    extract_texture_watermark_adaptive_jpeg,
)


_LOGGER = logging.getLogger(__name__)


class _NoEmbeddedJpeg(ValueError):
    pass


class _JpegSlotTooSmall(ValueError):
    pass


class _JpegRecord(NamedTuple):
    start: int
    end: int
    declared_length: int | None


class _UnsupportedJpegLayout(ValueError):
    def __init__(self, record: _JpegRecord, reason: str):
        super().__init__(reason)
        self.offset = record.start
        self.declared_length = record.declared_length
        self.jpeg_length = record.end - record.start


def _jpeg_spans(data: bytes):
    start = 0
    while True:
        start = data.find(b"\xff\xd8\xff", start)
        if start < 0:
            return
        end = _jpeg_end(data, start)
        if end is None:
            return
        yield start, end
        start = end


_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_PNG_END = b"IEND\xaeB`\x82"


def _image_spans(data: bytes):
    """JPEG 与 PNG 内嵌纹理的字节区间，按偏移排序。

    OSGB 纹理在真实数据里是 JPEG，但客户导出/转换工具可以把纹理改存 PNG。
    只认 JPEG 签名时，一次格式转换就让整棵模型变成「无纹理可扫」——
    实测提取器本身对 PNG 像素是能解出 DCT 水印的（``extract_jpeg`` 只依赖
    PIL 解码后的像素，与容器格式无关），失败纯粹卡在容器扫描这一步。
    """
    spans = list(_jpeg_spans(data))
    position = 0
    while True:
        start = data.find(_PNG_SIG, position)
        if start < 0:
            break
        tail = data.find(_PNG_END, start)
        if tail < 0:
            break
        end = tail + len(_PNG_END)
        spans.append((start, end))
        position = end
    return sorted(spans)


def _texture_spans(data: bytes):
    """兼容旧调用名的别名。"""
    return _image_spans(data)


def _jpeg_end(data: bytes, start: int) -> int | None:
    """Find JPEG EOI while ignoring stuffed FF D9 bytes inside scan data."""
    position = start + 2
    while position + 1 < len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        if marker == 0x00:
            position += 2
            continue
        if marker == 0xD9:
            return position + 2
        # Once SOS is reached, entropy-coded data continues until EOI. In
        # that region FF00 is byte stuffing and D0-D7 are restart markers;
        # neither carries a segment length.
        if marker == 0xDA:
            position += 2
            while position + 1 < len(data):
                if data[position] != 0xFF:
                    position += 1
                    continue
                next_marker = data[position + 1]
                if next_marker == 0x00 or 0xD0 <= next_marker <= 0xD7:
                    position += 2
                    continue
                if next_marker == 0xD9:
                    return position + 2
                return None
            return None
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            position += 2
            continue
        if position + 4 > len(data):
            return None
        segment_length = int.from_bytes(data[position + 2:position + 4], "big")
        if segment_length < 2 or position + 2 + segment_length > len(data):
            return None
        position += 2 + segment_length
    return None


def watermark_osgb_bytes(data: bytes, transform, watermark: int) -> bytes:
    """Replace embedded JPEG payloads without shifting OSGB byte offsets.

    The callback must return a JPEG with exactly the original byte length;
    variable-length replacement is rejected because OSGB offsets may point into
    the container and cannot safely be rewritten without an OSGB serializer.
    """
    output = bytearray(data)
    replaced = 0
    for start, end in _jpeg_spans(data):
        original = data[start:end]
        replacement = bytes(transform(original, watermark))
        if len(replacement) != len(original):
            raise ValueError("embedded JPEG replacement changed length")
        output[start:end] = replacement
        replaced += 1
    if not replaced:
        raise ValueError("OSGB contains no complete embedded JPEG")
    return bytes(output)


def rewrite_osgb_bytes(data: bytes, transform, watermark: int) -> bytes:
    """Rewrite embedded JPEGs and their immediately preceding uint32 lengths.

    The legacy layout permits rewriting lengths. A tile with zero-padded
    records instead keeps every JPEG slot and container offset unchanged.
    """
    rewritten, _ = _rewrite_osgb_bytes_counted(data, transform, watermark)
    return rewritten


def _jpeg_records(data: bytes) -> list[_JpegRecord]:
    records = []
    cursor = 0
    for start, end in _jpeg_spans(data):
        length_pos = start - 4
        if length_pos < cursor:
            record = _JpegRecord(start, end, None)
            raise _UnsupportedJpegLayout(record, "JPEG has no independent length field")
        declared = int.from_bytes(data[length_pos:start], "little")
        record = _JpegRecord(start, end, declared)
        record_end = start + declared
        if record_end < end or record_end > len(data):
            raise _UnsupportedJpegLayout(record, "JPEG length falls outside its record bounds")
        if record_end != end and data[end:record_end].strip(b"\x00"):
            raise _UnsupportedJpegLayout(record, "JPEG record has an unrecognized nonzero trailer")
        records.append(record)
        cursor = record_end
    if not records:
        raise _NoEmbeddedJpeg("OSGB contains no complete embedded JPEG")
    return records


def _rewrite_osgb_bytes_counted(data: bytes, transform, watermark: int) -> tuple[bytes, int]:
    records = _jpeg_records(data)
    fixed_slots = any(record.declared_length != record.end - record.start for record in records)
    chunks: list[bytes] = []
    cursor = 0
    for record in records:
        start, end, _ = record
        replacement = bytes(transform(data[start:end], watermark))
        if fixed_slots:
            try:
                replacement = _pad_jpeg(replacement, end - start)
            except _JpegSlotTooSmall as exc:
                raise _UnsupportedJpegLayout(record, str(exc)) from exc
            # The original length field and any zero trailer remain in place.
            chunks.append(data[cursor:start])
        else:
            chunks.append(data[cursor:start - 4])
            chunks.append(len(replacement).to_bytes(4, "little"))
        chunks.append(replacement)
        cursor = end
    chunks.append(data[cursor:])
    return b"".join(chunks), len(records)


def _watermark_jpeg_fixed(payload: bytes, watermark: int) -> bytes:
    """Apply the existing carrier and keep the embedded JPEG span fixed-size."""
    with Image.open(BytesIO(payload)) as image:
        image.load()
        with BytesIO() as source, BytesIO() as marked:
            image.convert("RGB").save(source, "JPEG", quality=95, optimize=False, subsampling=0)
            source.seek(0)
            with Image.open(source) as normalized:
                normalized.load()
                embed_texture_watermark_adaptive_jpeg(normalized, marked, watermark)
            encoded = marked.getvalue()
    return _pad_jpeg(encoded, len(payload))


def _small_jpeg_marker(watermark: int) -> bytes:
    return b"GDS-WM-1:" + f"{watermark:08X}".encode("ascii")


def embed_osgb_jpeg_watermark(payload: bytes, watermark: int) -> bytes:
    """Embed a reliable marker in a small OSGB JPEG's COM segment.

    Small OSGB textures cannot carry a stable pixel-domain 32-bit codeword.
    JPEG COM is part of the image record, survives container rewriting, and
    does not alter decoded pixels. It is used only for the small-texture
    fallback; larger textures continue using the pixel-domain carrier.
    """
    if not 0 <= watermark <= 0xFFFFFFFF:
        raise ValueError("watermark must be an unsigned 32-bit integer")
    marker = _small_jpeg_marker(watermark)
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError("embedded payload is not a JPEG")
    if len(marker) + 4 > 0xFFFF:
        raise ValueError("JPEG watermark comment is too long")
    # Insert a legal COM segment immediately after SOI. This preserves the
    # original JPEG entropy stream and avoids decode/re-encode cost.
    segment = b"\xff\xfe" + (len(marker) + 2).to_bytes(2, "big") + marker
    return payload[:2] + segment + payload[2:]


def extract_osgb_jpeg_watermark(payload: bytes, watermark: int) -> dict:
    marker = _small_jpeg_marker(watermark)
    detected = any(
        code == 0xFE and payload[start:end] == marker
        for code, _, start, end in _jpeg_header_segments(payload)
    )
    return {
        "detected": detected, "payload_bits": 32, "method": "jpeg_com",
        "expected": f"{watermark:08X}",
    }


def _jpeg_header_segments(data: bytes):
    """Yield marker, marker offset and payload bounds up to the first scan."""
    if not data.startswith(b"\xff\xd8"):
        return
    position = 2
    while position + 1 < len(data) and data[position] == 0xFF:
        marker_start = position
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position + 2 >= len(data):
            return
        marker = data[position]
        if marker in {0x00, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            return
        length = int.from_bytes(data[position + 1:position + 3], "big")
        end = position + 1 + length
        if length < 2 or end > len(data):
            return
        yield marker, marker_start, position + 3, end
        if marker == 0xDA:
            return
        position = end


def _pad_jpeg(encoded: bytes, target_length: int) -> bytes:
    """Fit a complete JPEG using legal APP15 segments before its first scan."""
    segments = list(_jpeg_header_segments(encoded))
    if not segments or segments[-1][0] != 0xDA or _jpeg_end(encoded, 0) != len(encoded):
        raise ValueError("watermarked output is not a complete JPEG")
    missing = target_length - len(encoded)
    if missing < 0:
        raise _JpegSlotTooSmall("watermarked JPEG is larger than its fixed OSGB slot")
    if missing == 0:
        return encoded
    if missing < 4:
        raise _JpegSlotTooSmall("fixed JPEG slot leaves an unrepresentable padding remainder")
    scan_offset = segments[-1][1]
    chunks: list[bytes] = []
    remaining = missing
    while remaining:
        segment_size = min(remaining, 65537)
        tail = remaining - segment_size
        if 0 < tail < 4:
            segment_size -= 4 - tail
        payload_size = segment_size - 4
        chunks.append(b"\xff\xef" + (payload_size + 2).to_bytes(2, "big") + b"\x00" * payload_size)
        remaining -= segment_size
    return encoded[:scan_offset] + b"".join(chunks) + encoded[scan_offset:]


def embed_osgb_bytes(data: bytes, watermark: int) -> tuple[bytes, int]:
    """Watermark one OSGB byte buffer without intermediate source files."""
    marked, count = _rewrite_osgb_bytes_counted(data, _watermark_jpeg_fixed_unbounded, watermark)
    if count <= 0:
        raise ValueError("OSGB contains no watermarkable embedded JPEG")
    return marked, count


def _embed_osgb_file_with_stats(source: Path, destination: Path, watermark: int) -> dict:
    """Watermark embedded JPEGs, using COM fallback for undersized textures."""
    source, destination = Path(source), Path(destination)
    pixel_embedded = 0
    small_fallback = 0

    def transform(original: bytes, watermark: int) -> bytes:
        nonlocal pixel_embedded, small_fallback
        with Image.open(BytesIO(original)) as image:
            if min(image.size) < 224:
                image.load()
                replacement = embed_osgb_jpeg_watermark(original, watermark)
                small_fallback += 1
                return replacement
        replacement = _watermark_jpeg_fixed_unbounded(original, watermark)
        pixel_embedded += 1
        return replacement

    rewritten, count = _rewrite_osgb_bytes_counted(source.read_bytes(), transform, watermark)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(rewritten)
    return {"embedded_jpegs": count, "pixel_embedded": pixel_embedded, "small_fallback": small_fallback}


def embed_osgb_file(source: Path, destination: Path, watermark: int) -> int:
    """Watermark embedded JPEGs and rewrite record lengths safely."""
    return int(_embed_osgb_file_with_stats(source, destination, watermark)["pixel_embedded"])


def _watermark_jpeg_fixed_unbounded(payload: bytes, watermark: int) -> bytes:
    """Embed a valid adaptive JPEG; the OSGB length field is rewritten by caller."""
    with Image.open(BytesIO(payload)) as image:
        image.load()
        with BytesIO() as marked:
            embed_texture_watermark_adaptive_jpeg(image, marked, watermark)
            return marked.getvalue()


# ── 进程池执行器 ─────────────────────────────────────────────
# 为什么需要：单个文件的 DCT/IDCT 走 scipy，不释放 GIL。实测 8 线程在 36 核机器上
# 只吃到 2.75 核，28,581 个文件要 27~30 分钟。改多进程才能真正用满 CPU。
_WORKER_WATERMARK = 0


def _pool_init(watermark: int) -> None:
    global _WORKER_WATERMARK
    _WORKER_WATERMARK = watermark


def _pool_embed_pair(pair: tuple[str, str]) -> tuple[int, int, dict | None]:
    return _embed_pair(Path(pair[0]), Path(pair[1]), _WORKER_WATERMARK)


def _embed_pair(source: Path, destination: Path, watermark: int) -> tuple[int, int, dict | None]:
    try:
        stats = _embed_osgb_file_with_stats(source, destination, watermark)
        return int(stats["pixel_embedded"]), int(stats["small_fallback"]), None
    except _UnsupportedJpegLayout as exc:
        destination.write_bytes(source.read_bytes())
        return 0, 0, {
            "source": str(source), "reason": str(exc), "offset": exc.offset,
            "declared_length": exc.declared_length, "jpeg_length": exc.jpeg_length,
        }
    except _NoEmbeddedJpeg as exc:
        destination.write_bytes(source.read_bytes())
        return 0, 0, {"source": str(source), "reason": str(exc)}


def _process_workers() -> int:
    try:
        configured = int(os.environ.get("GDS_OSGB_PROCESSES", "0") or 0)
    except ValueError:
        configured = 0
    if configured > 0:
        return min(configured, 32)
    return max(1, min(12, (os.cpu_count() or 4) - 4))


def _executor_mode() -> str:
    return (os.environ.get("GDS_OSGB_EXECUTOR", "thread") or "thread").strip().lower()


def _osgb_workers() -> int:
    """Pool size for OSGB tree watermarking.

    Measured 2026-10-17 with the vectorised DCT pass on real tiles (64-file sample):
        serial            20.37 files/s  -> 10 GB in 20.9 min
        4 threads         36.00 files/s  -> 11.9 min
        8 threads         39.13 files/s  -> 10.9 min   <-- best
        12 threads        38.80 files/s  -> 11.0 min
    10 GB Mesh sla is 30 min, so 8 threads leaves ~3x headroom. Before the vectorised pass the
    same work ran at 2.03 files/s (210 min), which is why threading alone never helped: the run
    was bound by Python-level FFT call overhead, not by CPU.
    """
    configured = int(os.environ.get("GDS_OSGB_WORKERS", "0") or 0)
    return configured if configured > 0 else 8



# 进度钩子：任务线程设置自己的回调，嵌入循环每处理一批就回报一次。
# 之前 progress_percent 只在阶段切换时写，28,581 个文件跑 8 小时期间前端一直显示 35%，
# 看起来像卡死。线程局部存储保证并发任务之间不会互相覆盖进度。
_PROGRESS_LOCAL = threading.local()


def set_progress_hook(fn) -> None:
    """为当前线程设置嵌入进度回调 ``fn(done, total)``；传 None 清除。"""
    _PROGRESS_LOCAL.fn = fn


def _current_hook():
    return getattr(_PROGRESS_LOCAL, "fn", None)


def embed_osgb_tree(source_dir: Path, destination_dir: Path, watermark: int,
                     progress=None) -> int:
    """Copy an extracted OSGB tree and watermark all OSGB files recursively."""
    source_dir, destination_dir = Path(source_dir), Path(destination_dir)
    if not source_dir.is_dir():
        raise ValueError("OSGB source must be an extracted directory")
    osgb_files = sorted(source_dir.rglob("*.osgb"))
    if not osgb_files:
        raise ValueError("OSGB package contains no .osgb files")
    destination_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    total_small_fallback = 0
    processed_files = 0
    skipped_files = []
    unsupported_layout_tiles = []
    watermarked_files = 0
    marked_files = 0
    # Directory and sidecar copies stay on this thread; each worker owns a distinct destination
    # file, so no locking is needed and a failure still surfaces as the same ValueError.
    work: list[tuple[Path, Path]] = []
    for source in source_dir.rglob("*"):
        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source.suffix.lower() == ".osgb":
            destination.parent.mkdir(parents=True, exist_ok=True)
            work.append((source, destination))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    def handle(pair: tuple[Path, Path]) -> tuple[int, int, dict | None]:
        return _embed_pair(*pair, watermark)

    workers = min(_osgb_workers(), max(1, len(work)))
    hook = progress or _current_hook()
    total_jobs = len(work)
    reported = 0

    def _tick() -> None:
        # 每 200 个文件回报一次：28,581 个文件约 143 次写入，代价可以忽略，
        # 但足以让前端看到进度在走；同时避免每文件一次写库把 SQLite 打满。
        nonlocal reported
        if hook is None:
            return
        if processed_files - reported >= 200 or processed_files == total_jobs:
            reported = processed_files
            try:
                hook(processed_files, total_jobs)
            except Exception:  # noqa: BLE001 - 进度回报失败绝不能中断嵌入
                pass

    if hook is not None:
        try:
            hook(0, total_jobs)
        except Exception:  # noqa: BLE001
            pass

    def collect(results) -> None:
        nonlocal total, total_small_fallback, processed_files, watermarked_files, marked_files
        for count, small, skip in results:
            total += count
            total_small_fallback += small
            processed_files += 1
            watermarked_files += bool(count)
            marked_files += bool(count or small)
            if skip:
                relative = Path(skip.pop("source")).relative_to(source_dir).as_posix()
                skipped_files.append({"source": relative, "reason": skip["reason"]})
                if "offset" in skip:
                    unsupported_layout_tiles.append({"path": relative, **skip})
            _tick()

    if _executor_mode() == "process" and len(work) >= 200:
        # 多进程路线：父进程只做分发与统计，解码/编码全在子进程里跑，
        # 绕开 GIL。结果按完成顺序回收，进度照样每 200 个回报一次。
        import multiprocessing as _mp
        procs = _process_workers()
        pairs = [(str(src), str(dst)) for src, dst in work]
        ctx = _mp.get_context("spawn")
        with ctx.Pool(processes=procs, initializer=_pool_init, initargs=(watermark,)) as pool:
            collect(pool.imap_unordered(_pool_embed_pair, pairs, chunksize=16))
    elif workers <= 1:
        collect(map(handle, work))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            collect(pool.map(handle, work))
    (destination_dir / "processing_report.json").write_text(
        json.dumps({
            "files_processed": processed_files,
            "watermarked_files": watermarked_files,
            "marked_files": marked_files,
            "watermark_skipped_files": len(skipped_files),
            "small_texture_fallbacks": total_small_fallback,
            "skipped": skipped_files,
            "unsupported_layout_tiles": unsupported_layout_tiles,
            "watermark_method": "jpeg_pixel_carrier_or_preserved_unwatermarked",
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if unsupported_layout_tiles:
        _LOGGER.warning(
            "Preserved %d OSGB tiles with unsupported JPEG layouts. First entries: %s",
            len(unsupported_layout_tiles), json.dumps(unsupported_layout_tiles[:3], ensure_ascii=False),
        )
    return total

__all__ = [
    "embed_texture_watermark", "extract_texture_watermark",
    "embed_texture_watermark_adaptive", "extract_texture_watermark_adaptive",
    "embed_texture_watermark_adaptive_jpeg", "extract_texture_watermark_adaptive_jpeg",
    "embed_osgb_jpeg_watermark", "extract_osgb_jpeg_watermark",
    "watermark_osgb_bytes", "rewrite_osgb_bytes",
    "embed_osgb_bytes", "embed_osgb_file", "embed_osgb_tree", "set_progress_hook",
]
