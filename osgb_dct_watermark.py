from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct, idct

BITS = 32
REPEAT = 4
BLOCK = 8
COEF_A = (2, 3)
COEF_B = (3, 2)
DELTA = 18.0
MIN_SIZE = 128

# 纹理被缩放后，8x8 块网格与嵌入时错位，直提解出的是伪值；把纹理放大回
# 嵌入时的尺寸就能重新对齐。这里枚举常见缩放倍率的**还原倍数**：
# 纹理被缩到 0.5x 时要用 2.0 还原，缩到 0.75x 用 1.3333，缩到 4x 时用 0.25。
# 判定仍是「解出的 32 位值完全等于期望值」，误报概率 ~2^-32/次，可忽略。
_RESCALE_FACTORS = (2.0, 1.333333, 3.0, 1.5, 4.0, 0.75, 0.666667, 0.5, 0.25)


def _bits(value: int) -> list[int]:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("watermark must be unsigned 32-bit")
    raw = [(value >> (31 - i)) & 1 for i in range(BITS)]
    return [bit for bit in raw for _ in range(REPEAT)]


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _dct2(a: np.ndarray) -> np.ndarray:
    return dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def _idct2(a: np.ndarray) -> np.ndarray:
    return idct(idct(a, axis=0, norm="ortho"), axis=1, norm="ortho")


def _blocks(h: int, w: int):
    for y in range(0, h - BLOCK + 1, BLOCK):
        for x in range(0, w - BLOCK + 1, BLOCK):
            yield y, x


def _force_pair(a: float, b: float, bit: int) -> tuple[float, float]:
    mid = (a + b) / 2.0
    sign = 1.0 if bit else -1.0
    return mid + sign * DELTA / 2.0, mid - sign * DELTA / 2.0


def embed_jpeg(payload: bytes, watermark: int) -> tuple[bytes, dict]:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    positions = list(_blocks(h, w))
    bits = _bits(watermark)
    if min(h, w) < MIN_SIZE or len(positions) < len(bits):
        raise ValueError("texture is too small for dct watermark")
    y = _gray(rgb)
    out = y.copy()
    for (row, col), bit in zip(positions, bits):
        block = _dct2(y[row:row + BLOCK, col:col + BLOCK])
        a, b = block[COEF_A], block[COEF_B]
        block[COEF_A], block[COEF_B] = _force_pair(float(a), float(b), bit)
        out[row:row + BLOCK, col:col + BLOCK] = _idct2(block)
    marked = rgb.copy().astype(np.float32)
    delta = np.clip(out - y, -40.0, 40.0)
    marked += delta[..., None]
    marked = np.clip(np.rint(marked), 0, 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(marked, "RGB").save(buf, format="JPEG", quality=95, optimize=False, subsampling=0)
    return buf.getvalue(), {"method": "jpeg_dct_cpu", "samples": len(bits), "size": [w, h]}


def extract_jpeg(payload: bytes, watermark: int, scale_search: bool = False) -> dict:
    with Image.open(BytesIO(payload)) as image:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    positions = list(_blocks(h, w))
    bits = _bits(watermark)
    if min(h, w) < MIN_SIZE or len(positions) < len(bits):
        return {"detected": False, "method": "jpeg_dct_cpu", "reason": "too_small", "size": [w, h]}

    def _decode(image_rgb: np.ndarray) -> int:
        gh, gw = image_rgb.shape[:2]
        gpos = list(_blocks(gh, gw))
        gy = _gray(image_rgb)
        votes = []
        for row, col in gpos[:len(bits)]:
            block = _dct2(gy[row:row + BLOCK, col:col + BLOCK])
            votes.append(int(float(block[COEF_A]) >= float(block[COEF_B])))
        value = 0
        for i in range(BITS):
            group = votes[i * REPEAT:(i + 1) * REPEAT]
            value = (value << 1) | int(sum(group) * 2 >= len(group))
        return value

    value = _decode(rgb)
    result = {"method": "jpeg_dct_cpu", "fingerprint": value,
              "expected": int("".join(str(x) for x in bits[::REPEAT]), 2), "size": [w, h]}
    if value == result["expected"]:
        result["detected"] = True
        return result

    # 纹理被缩放过时，8x8 块网格与嵌入时不再对齐，直提必然解出伪值。
    # 实测把缩小后的纹理**放大回原尺寸**再提就能解出（块 8/16/32 一致），
    # 所以这里按常见缩放倍率枚举上采样后重试，命中即返回。
    if scale_search:
        # 倍率 = 1/缩放比例。纹理缩到 0.75x 要用 1.3333 还原，缩到 0.5x 用 2.0。
        # 实测 0.75x 放大回原尺寸后每 bit 重复块一致率 1.00（完美恢复）；
        # 0.5x 只有 0.93 且最低 0.50 —— 8x8 块缩成 4x4 后 (2,3)/(3,2) 频点被抹平，
        # 那是信息真的丢了，不是网格错位，任何上采样都救不回来。
        for factor in _RESCALE_FACTORS:
            tw, th = int(round(w * factor)), int(round(h * factor))
            if tw < MIN_SIZE or th < MIN_SIZE:
                continue
            # 上采样用 NEAREST 会复制块边界，先试 BICUBIC/LANCZOS 两种
            for rname, resample in (("LANCZOS", Image.LANCZOS), ("BICUBIC", Image.BICUBIC)):
                up = np.asarray(Image.fromarray(rgb).resize((tw, th), resample), dtype=np.uint8)
                if len(list(_blocks(th, tw))) < len(bits):
                    continue
                candidate = _decode(up)
                if candidate == result["expected"]:
                    result.update({"detected": True, "fingerprint": candidate,
                                   "rescaled": [tw, th], "resample": rname})
                    return result
    result["detected"] = False
    return result


def embed_osgb_file(source: Path, destination: Path, watermark: int, spans_fn, end_fn) -> dict:
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    spans = list(spans_fn(data))
    if not spans:
        raise ValueError("no complete embedded JPEG")
    chunks = []
    cursor = 0
    pixel = not_watermarkable = 0
    for start, end in spans:
        if start < 4 or int.from_bytes(data[start - 4:start], "little") != end - start:
            raise ValueError("OSGB JPEG length mismatch")
        original = data[start:end]
        try:
            replacement, _ = embed_jpeg(original, watermark)
            pixel += 1
        except ValueError:
            replacement = original
            not_watermarkable += 1
        chunks.extend((data[cursor:start - 4], len(replacement).to_bytes(4, "little"), replacement))
        cursor = end
    chunks.append(data[cursor:])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(chunks))
    return {"embedded_jpegs": len(spans), "pixel_embedded": pixel, "not_watermarkable": not_watermarkable}
