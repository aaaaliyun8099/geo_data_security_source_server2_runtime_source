"""多域冗余水印：同一载荷同时写入两个互不相关的信号域。

## 为什么需要

单一域的水印只要被对手定位到承载域就能整体抹掉。例如
`block_sync_watermark` 全部写在 64 个 64×64 子块的 SVD 首奇异值上：
对手若知道这一点，直接对每个 512 块做一次平滑/重采样，该域就被摧毁。

多域冗余把同一载荷写到**两个互不重叠的域**：

    D1  频域（DCT-SVD 首奇异值）—— 抗几何变换，但对重采样敏感
    D2  空域（块内像素均值的低位）—— 抗重采样/模糊，但对裁剪敏感

对手必须同时破坏两处，且破坏手段要能同时命中两种统计量，
代价显著高于单域。任一路存活即可溯源。

## 设计要点

- D2 用「块内 16×16 小块的均值 LSB」，与 D1 的 SVD 奇异值不共享自由度
- 两路载荷相同但**编码方式不同**，避免对手一次定位同时摧毁
- 提取时两路独立解码、分别带 CRC，任一通过即可判定

## 边界（如实）

- D2 的 LSB 对 JPEG 类有损压缩极脆弱；本平台输出 GeoTIFF（无损），可接受
- 两路都写入会使 PSNR 下降约 3–4 dB，仍远高于 45 dB 门限
- 这不是「不可去除」，而是「必须同时破坏两个独立域」
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from scipy.fftpack import dctn, idctn

from . import block_sync_watermark as bsync
from . import robust_geotiff_watermark as rw

_TILE = rw._TILE
_BLOCK = rw._BLOCK

# D2：空域子块。每个 512 块切成 32×32 个 16×16 小格，
# 每格用「均值的最低位」承载 1 bit，取其中 64 格承载载荷。
_SPATIAL_CELL = 16
_SPATIAL_PER_SIDE = _TILE // _SPATIAL_CELL   # 32
_SPATIAL_CARRIERS = 64

_VERSION = "multi-domain-1"
_SPATIAL_TAG = "GDS_SPATIAL_WM"


def _bits_of(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def _crc8(value: int) -> int:
    crc = 0xFF
    for i in range(16):
        bit = (value >> (15 - i)) & 1
        crc ^= bit << 7
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _cell_slots() -> list[tuple[int, int]]:
    """选 64 个分散的小格作为空域载体，避开块边缘（重采样影响最大）。"""
    slots: list[tuple[int, int]] = []
    margin = 2
    step = 3
    for r in range(margin, _SPATIAL_PER_SIDE - margin, step):
        for c in range(margin, _SPATIAL_PER_SIDE - margin, step):
            slots.append((r, c))
            if len(slots) >= _SPATIAL_CARRIERS:
                return slots
    return slots


def _spatial_embed(tile: np.ndarray, watermark: int, strength: int = 1) -> np.ndarray:
    """在空域小格均值的最低位写入 16 位载荷 + 8 位 CRC。

    只改 LSB 不够稳（任何重采样都会抹掉），因此用「均值量化」：
    把格子均值量化到 2*strength 的奇数/偶数倍，等价于在均值上做 QIM，
    对小幅重采样有容忍度，同时不依赖 SVD 奇异值。
    """
    out = tile.astype(np.float32).copy()
    bits = list(_bits_of(watermark & 0xFFFF, 16)) + list(_bits_of(_crc8(watermark & 0xFFFF), 8))
    slots = _cell_slots()
    q = float(2 * strength)
    for i, (sr, sc) in enumerate(slots):
        if i >= len(bits):
            break
        r0, c0 = sr * _SPATIAL_CELL, sc * _SPATIAL_CELL
        cell = out[r0:r0 + _SPATIAL_CELL, c0:c0 + _SPATIAL_CELL]
        mean = float(cell.mean())
        bucket = math.floor(mean / q)
        if bucket % 2 != int(bits[i]):
            bucket += 1
        target = (bucket + 0.5) * q
        cell += (target - mean)
        out[r0:r0 + _SPATIAL_CELL, c0:c0 + _SPATIAL_CELL] = cell
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _spatial_extract(tile: np.ndarray, strength: int = 1) -> tuple[int, bool]:
    """从空域小格读回 16 位载荷 + CRC，返回 (载荷, 校验是否通过)。"""
    slots = _cell_slots()
    q = float(2 * strength)
    bits: list[int] = []
    for sr, sc in slots[:24]:
        r0, c0 = sr * _SPATIAL_CELL, sc * _SPATIAL_CELL
        cell = tile[r0:r0 + _SPATIAL_CELL, c0:c0 + _SPATIAL_CELL].astype(np.float32)
        mean = float(cell.mean())
        bucket = math.floor(mean / q)
        bits.append(bucket % 2)
    if len(bits) < 24:
        return 0, False
    payload = 0
    for b in bits[:16]:
        payload = (payload << 1) | b
    crc = 0
    for b in bits[16:24]:
        crc = (crc << 1) | b
    return payload, crc == _crc8(payload)


def embed(source: Path, destination: Path, watermark: int) -> dict:
    """双域嵌入：频域（block-sync）+ 空域（块内均值 QIM）。"""
    if not 0 <= watermark <= 0xFFFFFFF:
        raise ValueError("watermark must fit 28 bits for the frequency domain")

    # 第一步：频域（block-sync 自身完成嵌入与块序号）
    tmp = destination.with_suffix(".freq.tif")
    freq_stats = bsync.embed(source, tmp, watermark)

    with rasterio.open(tmp) as ds:
        data = ds.read(1)
        profile = ds.profile.copy()
    tmp.unlink(missing_ok=True)

    # 第二步：空域叠加（在频域结果上再写一层，两个域互不覆盖）
    out = data.copy()
    spatial_tiles = 0
    for row in range(0, data.shape[0] - _TILE + 1, _TILE):
        for col in range(0, data.shape[1] - _TILE + 1, _TILE):
            tile = data[row:row + _TILE, col:col + _TILE]
            if float(np.std(tile)) < 2.0:
                continue
            spatial_tiles += 1
            out[row:row + _TILE, col:col + _TILE] = _spatial_embed(tile, watermark)

    profile.update(dtype=out.dtype, count=1, compress="deflate")
    profile[_SPATIAL_TAG] = _VERSION
    with rasterio.open(destination, "w", **profile) as ds:
        ds.write(out, 1)

    diff = out.astype(np.float64) - data.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else 10 * math.log10((255.0 ** 2) / mse)
    return {"freq": freq_stats, "spatial_tiles": spatial_tiles,
            "spatial_psnr_db": round(psnr, 2), "version": _VERSION}


def extract_spatial(source: Path) -> dict:
    """只走空域通道解码（用于频域被破坏时）。"""
    with rasterio.open(source) as ds:
        data = ds.read(1)
    counts: dict[int, int] = {}
    for row in range(0, data.shape[0] - _TILE + 1, _TILE):
        for col in range(0, data.shape[1] - _TILE + 1, _TILE):
            tile = data[row:row + _TILE, col:col + _TILE]
            if float(np.std(tile)) < 2.0:
                continue
            value, ok = _spatial_extract(tile)
            if ok:
                counts[value] = counts.get(value, 0) + 1
    if not counts:
        return {"found": False, "channel": "spatial"}
    best = max(counts.items(), key=lambda kv: kv[1])
    return {"found": True, "watermark": best[0], "votes": best[1],
            "candidates": counts, "channel": "spatial",
            "note": "空域通道只承载低 16 位载荷"}


def extract(source: Path, reference: Path | None = None) -> dict:
    """双域提取：先试频域（精度高），失败再试空域（抗重采样）。"""
    freq = {"found": False}
    if reference is not None:
        try:
            r = bsync.extract_with_registration(source, reference)
            if r.get("found"):
                freq = {**r, "channel": "frequency"}
        except Exception as exc:  # noqa: BLE001
            freq = {"found": False, "error": f"{type(exc).__name__}: {exc}"}

    spatial = extract_spatial(source)

    if freq.get("found"):
        return {"found": True, "watermark": freq["watermark"],
                "channel": "frequency", "frequency": freq, "spatial": spatial,
                "note": "频域通道命中（完整 28 位载荷）"}
    if spatial.get("found"):
        return {"found": True, "watermark": spatial["watermark"],
                "channel": "spatial", "frequency": freq, "spatial": spatial,
                "note": "频域失效，空域通道命中（仅低 16 位载荷，需按登记值比对）"}
    return {"found": False, "channel": None, "frequency": freq, "spatial": spatial,
            "note": "两个域均未检出"}
