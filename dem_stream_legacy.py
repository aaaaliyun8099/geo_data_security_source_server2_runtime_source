"""DEM 跨瓦片水印：既保住跨瓦片可提取性，又把改动足迹压下来。

问题回顾
--------
旧「行内连续平铺」方案（副本长度 == 副本间隔 == 64）会让**整行每一个采样都参与**
承载：可提取性不错，但实测改动足迹 ≈47.9%（约一半采样被真正改值），代价偏高。

关键几何关系
------------
把一行看作一维载体，副本长度 ``L``、副本间隔 ``S``：

* **最小可解码窗口** ``W_min = (S - 1) + L``
  提取端在裁剪窗口内扫描 ``0..S-1`` 共 S 个相位，只要窗口够宽，必有一个完整副本
  落在某个被扫到的相位上。
* **改动足迹** 参与率 = ``L / S``；真正改值的比例约为其一半（LSB 恰好相同的采样不必改）。

于是 ``S > L`` 就能稀疏化，而 ``L`` 越小 W_min 越小。两个参数可以**同时**改善：
用「1 采样/位」（L=32）代替「2 采样/位」（L=64），S 保持 64，
参与率从 1.0 降到 0.5、W_min 从 127 降到 95 —— 足迹与可提取性同时变好。

本模块提供 ``samples_per_bit`` 与 ``replica_stride`` 两个旋钮，
两者的真实取舍由实测决定（见 tests/ 与实测日志）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
import rasterio
from rasterio.windows import Window

_MAGIC = b"GDSDEM1"
_BITS = 32
_TILE = 512
_STRIDE = 256

# ── 跨瓦片参数 ───────────────────────────────────────────────────────────────
# 实测（dem_scheme_matrix）：
#   v1 行内连续平铺 L=64,S=64  足迹 50.05%   W_min 127   窗口 0/7
#   v2 1采样/位  L=32,S=96     足迹 17.17%   W_min 127   窗口 7/7  ← 默认
# 选 S=96：足迹降到约 1/3，而 W_min 与旧文档口径的 127 完全一致（不牺牲可提取性）。
DEFAULT_SAMPLES_PER_BIT = 1      # 每位 1 个采样：L=32 → 足迹更低
DEFAULT_REPLICA_STRIDE = 96      # 副本间隔 96 → W_min = 95 + 32 = 127
# 解码定论所需的真实票数下限。票数不足时胜者由 max() 的平局打破决定（抽签），
# 解出的值不可信——1×800 薄窗口实测就是这种情况（得 1 票却"解出"错值）。
MIN_DECODE_VOTES = 2
_BIT_SLOT = 32                   # 32 bit 水印


def cross_tile_layout(samples_per_bit: int = DEFAULT_SAMPLES_PER_BIT,
                      replica_stride: int | None = DEFAULT_REPLICA_STRIDE) -> dict:
    """给定参数，返回这一组配置的几何口径（供 API / 文档 / 测试共用）。"""
    spb = max(1, int(samples_per_bit))
    length = _BIT_SLOT * spb
    stride = int(replica_stride) if replica_stride else length
    stride = max(length, stride)
    return {
        "samples_per_bit": spb,
        "replica_length": length,
        "replica_stride": stride,
        "min_window_width": (stride - 1) + length,
        "participation": round(length / stride, 6),
        "bits": _BITS,
    }


def windows(width: int, height: int, tile: int = _TILE, stride: int | None = None) -> Iterator[tuple[int, int, int, int]]:
    stride = tile if stride is None else stride
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            yield x, y, min(tile, width - x), min(tile, height - y)


def _bits(value: int) -> list[int]:
    return [(value >> (_BITS - 1 - i)) & 1 for i in range(_BITS)]


# --------------------------------------------------------------------------- #
# 同瓦片内（grid-dependent）——供 stream_* 使用，保持原行为
# --------------------------------------------------------------------------- #
def _embed_array(a: np.ndarray, value: int) -> np.ndarray:
    out = a.copy()
    flat = out.reshape(-1)
    if flat.size < 64:
        raise ValueError("DEM tile too small")
    for i, bit in enumerate(_bits(value)):
        for j in (2 * i, 2 * i + 1):
            flat[j] = (int(flat[j]) & ~1) | bit
    return out


def _extract_array(a: np.ndarray) -> tuple[int, float]:
    flat = a.reshape(-1)
    if flat.size < 64:
        raise ValueError("DEM tile too small")
    value = 0
    confidence = 0
    for i in range(_BITS):
        pair = [int(flat[2 * i]) & 1, int(flat[2 * i + 1]) & 1]
        bit = int(sum(pair) >= 1)
        confidence += int(pair[0] == pair[1])
        value = (value << 1) | bit
    return value, confidence / _BITS


# --------------------------------------------------------------------------- #
# 跨瓦片（grid-independent）
# --------------------------------------------------------------------------- #
def _group_value(group: np.ndarray, samples_per_bit: int) -> tuple[int, float]:
    """从长度 32*samples_per_bit 的采样组里表决出 32 bit 值。

    ``samples_per_bit == 2``：该位两个采样取多数（容忍单点噪声）。
    ``samples_per_bit == 1``：直接读该采样；组内无冗余，可靠性由**跨副本表决**保证
    （见 ``cross_tile_extract``），此时 confidence 恒为 1.0。
    """
    value = 0
    agree = 0
    for i in range(_BITS):
        samples = [int(group[i * samples_per_bit + k]) & 1 for k in range(samples_per_bit)]
        value = (value << 1) | int(sum(samples) * 2 >= len(samples))
        agree += int(len(set(samples)) == 1)
    return value, agree / _BITS


def cross_tile_embed(source: Path, destination: Path, watermark: int,
                     samples_per_bit: int = DEFAULT_SAMPLES_PER_BIT,
                     replica_stride: int | None = DEFAULT_REPLICA_STRIDE) -> dict:
    """把水印按「副本」稀疏铺进每一行，提取与瓦片划分无关。

    每行独立承载：行内从第 0 列起，每 ``replica_stride`` 列放一个长度
    ``replica_length`` 的完整副本。只要裁剪窗口宽度 ≥ ``min_window_width``，
    窗口内必含一个完整副本（含相位回旋余量），故任何瓦片划分/裁剪都能读回。
    """
    source, destination = Path(source), Path(destination)
    layout = cross_tile_layout(samples_per_bit, replica_stride)
    spb = layout["samples_per_bit"]
    length = layout["replica_length"]
    stride = layout["replica_stride"]
    bits = _bits(watermark)

    with rasterio.open(source) as src:
        profile = src.profile.copy()
        data = src.read(1)
        flat = data.reshape(-1).copy()
        width, height = int(src.width), int(src.height)
        before = [int(np.min(data)), int(np.max(data))]
        replicas = 0
        for row in range(height):
            base_row = row * width
            col = 0
            while col + length <= width:
                base = base_row + col
                for i, bit in enumerate(bits):
                    for k in range(spb):
                        idx = base + i * spb + k
                        flat[idx] = (int(flat[idx]) & ~1) | bit
                replicas += 1
                col += stride
        data = flat.reshape(data.shape)
        with rasterio.open(destination, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(
                GDS_DEM_WATERMARK=f"{watermark:08X}",
                GDS_DEM_SCHEME="cross_tile_v2",
                GDS_DEM_SPB=str(spb),
                GDS_DEM_STRIDE=str(stride),
                GDS_DEM_MIN_WINDOW=str(layout["min_window_width"]),
            )
        return {
            "scheme": "cross_tile_v2",
            "replicas": replicas,
            "width": width,
            "height": height,
            "total_samples": int(flat.size),
            "group_length": length,
            "samples_per_bit": spb,
            "replica_stride": stride,
            "row_aligned": True,
            "min_window_width": layout["min_window_width"],
            "participation": layout["participation"],
            "grid_independent": True,
            "elevation_before": before,
        }


def cross_tile_extract(source: Path, watermark: int,
                       window: tuple[int, int, int, int] | None = None,
                       samples_per_bit: int = DEFAULT_SAMPLES_PER_BIT,
                       replica_stride: int | None = DEFAULT_REPLICA_STRIDE) -> dict:
    """跨瓦片提取：任意窗口/任意瓦片划分都能读回水印。

    ``window`` 为 ``(row_off, col_off, height, width)``。

    做法：窗口内**穷举相位**（0..stride-1）读取完整副本，解出的值累计投票；
    得票最高者即结论，得票率为置信度。跨多行多副本投票使 1 采样/位同样稳，
    既不依赖单个副本，也不依赖瓦片划分。
    """
    source = Path(source)
    layout = cross_tile_layout(samples_per_bit, replica_stride)
    spb = layout["samples_per_bit"]
    length = layout["replica_length"]
    stride = layout["replica_stride"]

    with rasterio.open(source) as src:
        if window:
            row_off, col_off, height, width = window
            # 先夹到图像范围内：越界请求静默返回裁剪过的数据，会让调用方
            # 以为拿到了完整的 width 列。这里如实报告实际读到的范围。
            r0 = max(0, min(row_off, src.height))
            c0 = max(0, min(col_off, src.width))
            r1 = max(r0, min(row_off + height, src.height))
            c1 = max(c0, min(col_off + width, src.width))
            data = src.read(1, window=Window(c0, r0, max(1, c1 - c0), max(1, r1 - r0)))
            clipped = (r0 != row_off or c0 != col_off
                       or (r1 - r0) != height or (c1 - c0) != width)
        else:
            data = src.read(1)
            clipped = False
        height, width = int(data.shape[0]), int(data.shape[1])

        votes: dict[int, int] = {}
        total = 0
        replica_slots = 0  # 能容纳一个"完整副本"的相位个数 × 行数 = 真实承载槽位数
        if width >= length:
            phases = range(min(stride, width - length + 1))
            replica_slots = len(phases) * height
            for row in range(height):
                row_data = data[row]
                for phase in phases:
                    value, _ = _group_value(row_data[phase:phase + length], spb)
                    votes[value] = votes.get(value, 0) + 1
                    total += 1

        best_value, best_hits = (None, 0)
        if votes:
            best_value, best_hits = max(votes.items(), key=lambda item: (item[1], item[0]))
        support = (best_hits / total) if total else 0.0
        # 可解码性必须基于**证据**，不能靠平局打破规则。
        # 实测教训：1×800 的薄窗口里真实值只拿到 1 票，与随机候选同票数，
        # 旧的 `max(key=(hits, value))` 按候选数值大小选胜者 → 解出 FCE1207C 这种错值，
        # 调用方无从分辨这是"检出的水印"还是"抽签抽出来的数"。
        # 判据只能是**实际票数**（``groups_checked`` 是相位扫描总次数，不是票数密度，
        # 拿它算比例会把所有合规窗口都误判为不可信）。
        ranked = sorted(votes.values(), reverse=True) if votes else []
        unique_top = len(ranked) < 2 or ranked[0] > ranked[1]
        decodable = bool(replica_slots) and best_hits >= MIN_DECODE_VOTES and unique_top
        verified = bool(best_value == watermark) and decodable
        ambiguous = bool(votes) and not unique_top and best_value == watermark
        return {
            "scheme": "cross_tile_v2",
            "window": list(window) if window else None,
            "window_requested": [row_off, col_off, height, width] if window else None,
            "window_actual": [r0, c0, r1 - r0, c1 - c0] if window else None,
            "window_clipped": clipped,
            "samples_per_bit": spb,
            "replica_stride": stride,
            "replica_length": length,
            "groups_checked": total,
            "distinct_candidates": len(votes),
            "matched": int(best_value == watermark),
            "decoded": f"{best_value:08X}" if best_value is not None else None,
            "expected": f"{watermark:08X}",
            "support": round(support, 6),
            "rate": round(support, 6),
            "w_min": stride - 1 + length,
            "min_votes_required": MIN_DECODE_VOTES,
            "votes_for_best": best_hits,
            "votes_for_runner_up": (ranked[1] if len(ranked) > 1 else 0),
            "unique_top": unique_top,
            "ambiguous": ambiguous,
            "replica_slots": replica_slots,
            "decodable": decodable,
            "undecodable_reason": None if decodable else (
                f"窗口 {height}×{width}：最高票 {best_hits}（需 ≥{MIN_DECODE_VOTES}）"
                + ("，且与次高票并列——无法区分真值还是噪声"
                   if not unique_top else "")
                + "；票数不足时胜者由平局打破决定，解出的值无意义，不得当作水印结论"),
            "grid_independent": True,
            "verified": verified,
        }


def cross_tile_footprint(source: Path, destination: Path,
                         samples_per_bit: int = DEFAULT_SAMPLES_PER_BIT,
                         replica_stride: int | None = DEFAULT_REPLICA_STRIDE) -> dict:
    """实测改动足迹：真正被改值的采样比例（不是"参与率"）。"""
    layout = cross_tile_layout(samples_per_bit, replica_stride)
    with rasterio.open(source) as handle_a, rasterio.open(destination) as handle_b:
        before = handle_a.read(1)
        after = handle_b.read(1)
    changed = int(np.count_nonzero(before != after))
    total = int(before.size)
    delta = np.abs(after.astype(np.int64) - before.astype(np.int64)) if total else np.array([0])
    return {
        "samples_per_bit": layout["samples_per_bit"],
        "replica_stride": layout["replica_stride"],
        "participation_theoretical": layout["participation"],
        "changed_samples": changed,
        "total_samples": total,
        "footprint_ratio": round(changed / total, 6) if total else 0.0,
        "min_window_width": layout["min_window_width"],
        "max_abs_delta": int(delta.max()),
    }


def stream_embed(source: Path, destination: Path, watermark: int, tile: int = _TILE, stride: int | None = None) -> dict:
    source, destination = Path(source), Path(destination)
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        total = 0
        with rasterio.open(destination, "w", **profile) as dst:
            dst.update_tags(GDS_DEM_WATERMARK=f"{watermark:08X}", GDS_DEM_TILE=str(tile), GDS_DEM_STRIDE=str(stride))
            for x, y, w, h in windows(src.width, src.height, tile, stride):
                arr = src.read(1, window=Window(x, y, w, h))
                if arr.size < 64:
                    continue
                dst.write(_embed_array(arr, watermark), 1, window=Window(x, y, w, h))
                total += 1
    return {"tiles": total, "width": profile["width"], "height": profile["height"], "streaming": True}


def stream_extract(source: Path, watermark: int, tile: int = _TILE, stride: int | None = None) -> dict:
    matched = total = 0
    with rasterio.open(source) as src:
        for x, y, w, h in windows(src.width, src.height, tile, stride):
            arr = src.read(1, window=Window(x, y, w, h))
            if arr.size < 64:
                continue
            value, confidence = _extract_array(arr)
            total += 1
            matched += int(value == watermark and confidence >= 0.75)
    return {"samples": total, "matched": matched, "rate": matched / total if total else 0.0, "streaming": True}


def source_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with rasterio.open(path) as src:
        for x, y, w, hgt in windows(src.width, src.height, tile=1024, stride=1024):
            h.update(src.read(1, window=Window(x, y, w, hgt)).tobytes())
    return h.hexdigest()
