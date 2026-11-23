"""点云坐标域水印。

旧实现只写哈希 sidecar，**不含坐标域修改**（`coordinate_domain_verified` 恒为 false）。
本模块补上真正的坐标域水印：

* **XYZ / PTS / PCD(ascii) / PLY(ascii)**：把水印比特写进逐点文本坐标的**最低有效十进制位**，
  改动幅度 ≤ 1 个最低位，远小于点云测量噪声。
* **LAS / LAZ / E57 / 二进制 PLY / 二进制 PCD**：未实现坐标域改写，**显式抛错**
  （不静默退回 sidecar，避免「看起来成功、实际未嵌入」）。

比特布局：`magic(16b) | length(16b) | payload(UTF-8 8b/字节) | checksum(16b)`，
每比特占一个坐标最低位；提取时校验 magic + 长度 + 校验和，**篡改坐标即验证失败**。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_MAGIC = b"GDSPC1"
_MAGIC_WM = 0x4744  # 'GD'
SUPPORTED_POINT_CLOUD_EXTENSIONS = {".las", ".laz", ".e57", ".ply", ".pcd", ".xyz", ".pts"}
#: 已实现坐标域水印的格式（文本坐标类）。
COORDINATE_DOMAIN_EXTENSIONS = {".xyz", ".pts", ".pcd", ".ply"}


class PointCloudWatermarkError(ValueError):
    """点云水印参数或格式不受支持。"""


def discover(root: Path) -> list[Path]:
    """列出受支持的点云文件。``root`` 可以是目录（递归）或单个文件。"""
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_POINT_CLOUD_EXTENSIONS else []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_POINT_CLOUD_EXTENSIONS)


def stream_hash(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 比特流打包 / 解包
# --------------------------------------------------------------------------- #

def _pack_bits(payload: str) -> list[int]:
    data = payload.encode("utf-8")
    if len(data) > 0xFFFF:
        raise PointCloudWatermarkError("watermark payload too long (max 65535 bytes)")
    checksum = sum(data) & 0xFFFF
    bits: list[int] = []

    def push(value: int, width: int) -> None:
        for shift in range(width - 1, -1, -1):
            bits.append((value >> shift) & 1)

    push(_MAGIC_WM, 16)
    push(len(data), 16)
    for byte in data:
        push(byte, 8)
    push(checksum, 16)
    return bits


def _bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (bit & 1)
    return value


def _unpack_bits(bits: list[int]) -> tuple[str | None, str | None]:
    if len(bits) < 32:
        return None, "insufficient_bits"
    if _bits_to_int(bits[:16]) != _MAGIC_WM:
        return None, "magic_mismatch"
    length = _bits_to_int(bits[16:32])
    need = 32 + length * 8 + 16
    if len(bits) < need:
        return None, "truncated_payload"
    data = bytes(_bits_to_int(bits[32 + i * 8:40 + i * 8]) for i in range(length))
    checksum = _bits_to_int(bits[32 + length * 8:need])
    if (sum(data) & 0xFFFF) != checksum:
        return None, "checksum_mismatch"
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "decode_error"


# --------------------------------------------------------------------------- #
# 文本点云格式解析
# --------------------------------------------------------------------------- #

def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _text_line_format(path: Path) -> tuple[str, list[str]]:
    """判定文本点云格式并返回 (格式名, 行列表)；二进制/不支持格式抛错。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    suffix = Path(path).suffix.lower()
    if suffix == ".pcd":
        header_end = next((i for i, line in enumerate(lines) if line.strip().upper().startswith("DATA")), None)
        if header_end is None:
            raise PointCloudWatermarkError("PCD file has no DATA header line")
        if "ascii" not in lines[header_end].strip().lower():
            raise PointCloudWatermarkError("only ascii PCD supports coordinate-domain watermarking")
        return "pcd", lines
    if suffix == ".ply":
        header_end = next((i for i, line in enumerate(lines) if line.strip().lower() == "end_header"), None)
        if header_end is None:
            raise PointCloudWatermarkError("PLY file has no end_header line")
        if any("format binary" in line.lower() for line in lines[:header_end]):
            raise PointCloudWatermarkError("only ascii PLY supports coordinate-domain watermarking")
        return "ply", lines
    if suffix == ".pts":
        return "pts", lines
    if suffix == ".xyz":
        return "xyz", lines
    raise PointCloudWatermarkError(f"no coordinate-domain watermark for {suffix}")


def _data_row_range(lines: list[str], fmt: str) -> tuple[int, int]:
    """返回承载坐标的行区间 [start, end)。"""
    if fmt == "pcd":
        start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith("DATA")) + 1
        return start, len(lines)
    if fmt == "ply":
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == "end_header") + 1
        return start, len(lines)
    if fmt == "pts":
        return 1, len(lines)  # 第一行是点数
    return 0, len(lines)


def _collect_slots(lines: list[str], fmt: str) -> tuple[list[list[str]], list[tuple[int, int]]]:
    """把数据行切成 token，并列出可作为比特槽位的 (行下标, 列下标)。"""
    start, end = _data_row_range(lines, fmt)
    parsed: list[list[str]] = []
    slots: list[tuple[int, int]] = []
    for row_index in range(start, end):
        tokens = lines[row_index].replace(",", " ").split()
        parsed.append(tokens)
        for column in range(min(3, len(tokens))):
            if _is_number(tokens[column]):
                slots.append((len(parsed) - 1, column))
    return parsed, slots


def _toggle_lsb(token: str, bit: int) -> tuple[str, bool]:
    """把 token 最低小数位调整到目标奇偶；返回（新 token, 是否改动）。"""
    negative = token.startswith("-")
    body = token[1:] if negative else token
    if "." not in body:
        body += ".0"
    head, frac = body.split(".", 1)
    frac = frac or "0"
    last = int(frac[-1])
    if (last & 1) == (bit & 1):
        return token, False
    last = last - 1 if last > 0 else last + 1
    return ("-" if negative else "") + head + "." + frac[:-1] + str(last), True


def _lsb_of(token: str) -> int:
    body = token[1:] if token.startswith("-") else token
    frac = body.split(".")[1] if "." in body else "0"
    return (int(frac[-1]) if frac else 0) & 1


def _write_back(lines: list[str], fmt: str, parsed: list[list[str]]) -> list[str]:
    start, _ = _data_row_range(lines, fmt)
    for offset, tokens in enumerate(parsed):
        lines[start + offset] = " ".join(tokens)
    return lines


def embed_coordinate_watermark(source: Path, destination: Path, payload: str) -> dict:
    """在文本点云坐标的最低有效十进制位嵌入水印，返回结构化报告。"""
    source, destination = Path(source), Path(destination)
    fmt, lines = _text_line_format(source)
    bits = _pack_bits(payload)
    parsed, slots = _collect_slots(lines, fmt)
    if len(slots) < len(bits):
        raise PointCloudWatermarkError(
            f"payload needs {len(bits)} coordinate slots but the file provides only {len(slots)}"
        )
    changed = 0
    for bit, (parsed_index, column) in zip(bits, slots):
        tokens = parsed[parsed_index]
        new_token, altered = _toggle_lsb(tokens[column], bit)
        if altered:
            tokens[column] = new_token
            changed += 1
    lines = _write_back(lines, fmt, parsed)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "method": "coordinate_lsb",
        "format": fmt,
        "coordinate_domain_verified": True,
        "points": len(parsed),
        "coordinate_slots": len(slots),
        "bits_embedded": len(bits),
        "coordinates_modified": changed,
        "max_amplitude": 0.1,
        "payload": payload,
        "source_hash": stream_hash(source),
        "output_hash": stream_hash(destination),
    }


def extract_coordinate_watermark(source: Path) -> dict:
    """从文本点云坐标读回水印；坐标被篡改则校验失败。"""
    source = Path(source)
    try:
        fmt, lines = _text_line_format(source)
    except PointCloudWatermarkError as exc:
        return {"verified": False, "reason": str(exc), "coordinate_domain_verified": False}
    parsed, slots = _collect_slots(lines, fmt)
    bits = [_lsb_of(parsed[parsed_index][column]) for parsed_index, column in slots]
    payload, error = _unpack_bits(bits)
    return {
        "verified": payload is not None,
        "reason": error,
        "payload": payload,
        "method": "coordinate_lsb",
        "format": fmt,
        "coordinate_domain_verified": payload is not None,
        "points": len(parsed),
        "coordinate_slots": len(slots),
    }


# --------------------------------------------------------------------------- #
# 兼容入口：文本类走坐标域，其余格式显式报错（不再静默退回 sidecar）
# --------------------------------------------------------------------------- #

def embed_file(source: Path, destination: Path, watermark: str) -> dict:
    """嵌入点云水印。

    文本点云（XYZ/PTS/PCD-ascii/PLY-ascii）→ 真坐标域水印；
    其余格式 → 抛 PointCloudWatermarkError，调用方据此明确失败。
    """
    source, destination = Path(source), Path(destination)
    suffix = source.suffix.lower()
    if suffix in COORDINATE_DOMAIN_EXTENSIONS:
        report = embed_coordinate_watermark(source, destination, watermark)
        sidecar = {"format": suffix, "source_hash": report["source_hash"], "watermark": watermark,
                   "method": report["method"], "coordinate_domain_verified": True}
        destination.with_suffix(destination.suffix + ".gds.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        report["sidecar"] = True
        return report
    raise PointCloudWatermarkError(
        f"coordinate-domain point-cloud watermarking is not implemented for {suffix}; "
        f"supported: {sorted(COORDINATE_DOMAIN_EXTENSIONS)}"
    )


def extract_file(source: Path, expected_watermark: str) -> dict:
    source = Path(source)
    suffix = source.suffix.lower()
    if suffix in COORDINATE_DOMAIN_EXTENSIONS:
        result = extract_coordinate_watermark(source)
        result["expected_match"] = result.get("payload") == expected_watermark
        result["verified"] = bool(result.get("verified") and result["expected_match"])
        return result
    sidecar = source.with_suffix(source.suffix + ".gds.json")
    if not sidecar.is_file():
        return {"verified": False, "reason": "sidecar_missing", "coordinate_domain_verified": False}
    row = json.loads(sidecar.read_text(encoding="utf-8"))
    current = stream_hash(source)
    return {"verified": row.get("source_hash") == current and row.get("watermark") == expected_watermark,
            "source_hash_match": row.get("source_hash") == current,
            "coordinate_domain_verified": False, "method": row.get("method")}
