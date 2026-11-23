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


def _band_rows(width: int, dtype_itemsize: int, target_bytes: int = 96 * 1024 * 1024) -> int:
    """按目标内存挑选行带高度，保证单次读取不超预算。

    水印是**行内独立**的（每行从第 0 列起铺副本），所以按行带切分与整幅处理
    在数学上完全等价；差别只在内存占用与 IO 次数。
    """
    per_row = max(1, width * dtype_itemsize)
    rows = max(1, target_bytes // per_row)
    return int(min(rows, 4096))


def _replica_starts(width: int, length: int, stride: int) -> np.ndarray:
    """一行内所有能放下完整副本的起始列。"""
    if width < length:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, width - length + 1, stride, dtype=np.int64)


def _bit_view(array: np.ndarray) -> np.ndarray:
    """按位视图：与传入数组共享内存，使浮点栅格也能做 LSB 读写。

    Float32/Float64 DEM 的水印位在尾数末位。若先 int() 再写，改的是高程数值本身
    （四舍五入），不是最低位；按同一块内存的整数视图写入，只动这一位，数值变化在 1 ULP 内。
    整型栅格原样返回，原有整型路径完全不变。
    """
    if array.dtype.kind == "f":
        target = np.int32 if array.dtype.itemsize == 4 else np.int64
        return array.view(target)
    return array


def _apply_bits(band: np.ndarray, starts: np.ndarray, bits: list[int], spb: int) -> int:
    """把 32 bit 水印写入 band 每一行的所有副本位置，返回本 band 的副本数。

    ``band`` 形状 ``(rows, width)``。所有行的副本起始列相同，因此每个 bit 只需
    一次 fancy-index 赋值即可覆盖整个 band —— 避免逐行逐副本的 Python 循环。
    """
    if starts.size == 0:
        return 0
    offsets = np.arange(spb, dtype=np.int64)
    rows = band.shape[0]
    for i, bit in enumerate(bits):
        cols = (starts[:, None] + (i * spb + offsets)[None, :]).ravel()
        picked = band[:, cols]
        band[:, cols] = (picked & ~1) | bit
    return int(rows * starts.size)


def cross_tile_embed(source: Path, destination: Path, watermark: int,
                     samples_per_bit: int = DEFAULT_SAMPLES_PER_BIT,
                     replica_stride: int | None = DEFAULT_REPLICA_STRIDE) -> dict:
    """把水印按「副本」稀疏铺进每一行，提取与瓦片划分无关。

    每行独立承载：行内从第 0 列起，每 ``replica_stride`` 列放一个长度
    ``replica_length`` 的完整副本。只要裁剪窗口宽度 ≥ ``min_window_width``，
    窗口内必含一个完整副本（含相位回旋余量），故任何瓦片划分/裁剪都能读回。

    **流式实现**：按行带读取、就地改 LSB、写回，单次内存占用与行宽成正比，
    与整幅像素数无关。旧实现 ``src.read(1)`` + ``reshape(-1).copy()`` 需要
    「像素数 × 2 字节 × 2」的连续内存（10 亿像素级 DEM 要 15 GiB），
    在沙箱 8 GiB 上限下会抛 ``RasterioIOError: Read failed``。
    """
    source, destination = Path(source), Path(destination)
    layout = cross_tile_layout(samples_per_bit, replica_stride)
    spb = layout["samples_per_bit"]
    length = layout["replica_length"]
    stride = layout["replica_stride"]
    bits = _bits(watermark)

    with rasterio.open(source) as src:
        profile = src.profile.copy()
        width, height = int(src.width), int(src.height)
        starts = _replica_starts(width, length, stride)
        band_rows = _band_rows(width, np.dtype(src.dtypes[0]).itemsize)
        replicas = 0
        before_min, before_max = None, None
        with rasterio.open(destination, "w", **profile) as dst:
            for row_off in range(0, height, band_rows):
                rows = min(band_rows, height - row_off)
                band = src.read(1, window=Window(0, row_off, width, rows))
                if band.dtype.kind == "f":
                    # NaN/Inf 不能进 int()：Float32 DEM 常见 NoData，直接转换会抛 ValueError。
                    finite = band[np.isfinite(band)]
                    band_min = int(finite.min()) if finite.size else 0
                    band_max = int(finite.max()) if finite.size else 0
                else:
                    band_min, band_max = int(band.min()), int(band.max())
                before_min = band_min if before_min is None else min(before_min, band_min)
                before_max = band_max if before_max is None else max(before_max, band_max)
                replicas += _apply_bits(_bit_view(band), starts, bits, spb)
                dst.write(band, 1, window=Window(0, row_off, width, rows))
            dst.update_tags(
                GDS_DEM_WATERMARK=f"{watermark:08X}",
                GDS_DEM_SCHEME="cross_tile_v2",
                GDS_DEM_SPB=str(spb),
                GDS_DEM_STRIDE=str(stride),
                GDS_DEM_MIN_WINDOW=str(layout["min_window_width"]),
                GDS_DEM_STREAMING="row_band",
                GDS_DEM_BAND_ROWS=str(band_rows),
            )
        return {
            "scheme": "cross_tile_v2",
            "replicas": replicas,
            "width": width,
            "height": height,
            "total_samples": int(width) * int(height),
            "group_length": length,
            "samples_per_bit": spb,
            "replica_stride": stride,
            "row_aligned": True,
            "min_window_width": layout["min_window_width"],
            "participation": layout["participation"],
            "grid_independent": True,
            "elevation_before": [before_min, before_max],
            "streaming": True,
            "band_rows": band_rows,
            "band_bytes": band_rows * width * np.dtype(profile["dtype"]).itemsize,
        }


def _group_values(slab: np.ndarray, samples_per_bit: int) -> np.ndarray:
    """向量化表决：``slab`` 形状 ``(rows, length)`` → 每行一个 32 bit 值。

    与 ``_group_value`` 逐行版语义完全一致，但用 numpy 一次算完整块。
    ``samples_per_bit == 1`` 时直接取 LSB；``== 2`` 时两位取多数。
    """
    lsb = slab & 1
    if samples_per_bit > 1:
        usable = (slab.shape[1] // samples_per_bit) * samples_per_bit
        grouped = lsb[:, :usable].reshape(slab.shape[0], _BITS, samples_per_bit)
        bits = (grouped.sum(axis=2) * 2 >= samples_per_bit).astype(np.int64)
    else:
        bits = lsb[:, :_BITS].astype(np.int64)
    weights = (1 << np.arange(_BITS - 1, -1, -1, dtype=np.int64))
    return bits @ weights


def cross_tile_extract(source: Path, watermark: int,
                       window: tuple[int, int, int, int] | None = None,
                       samples_per_bit: int = DEFAULT_SAMPLES_PER_BIT,
                       replica_stride: int | None = DEFAULT_REPLICA_STRIDE) -> dict:
    """跨瓦片提取：任意窗口/任意瓦片划分都能读回水印。

    ``window`` 为 ``(row_off, col_off, height, width)``。

    做法：窗口内**穷举相位**（0..stride-1）读取完整副本，解出的值累计投票；
    得票最高者即结论，得票率为置信度。跨多行多副本投票使 1 采样/位同样稳，
    既不依赖单个副本，也不依赖瓦片划分。

    **按相位打分（2026-10-20 修正）**：水印副本只从固定相位起铺（第 0、S、2S… 列），
    其余相位切出的 32 采样是**原始 LSB 的随机组合**。旧实现把所有相位的票混在
    一起统计，在 NoData 占比高的 DEM 上会被伪值淹没 —— 实测 SRTM_NE（33.9% 为
    ``-32768``）里 36 个相位整窗落在 NoData 区，全部解出 ``00000000``，
    合计 72,000 票，压过真实值的 2,000 票。

    修正后先按相位聚合，取**单相位内一致性最高**的那个相位作结论：
    真值所在相位会让几乎每一行都解出同一个值，伪相位则各自分散。
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
            clipped = (r0 != row_off or c0 != col_off
                       or (r1 - r0) != height or (c1 - c0) != width)
        else:
            r0, c0 = 0, 0
            r1, c1 = int(src.height), int(src.width)
            clipped = False
        height, width = max(0, r1 - r0), max(0, c1 - c0)

        total = 0
        replica_slots = 0  # 能容纳一个"完整副本"的相位个数 × 行数 = 真实承载槽位数
        # phase -> (该相位最高票的值, 票数, 次高票数, 不同候选数)
        per_phase: dict[int, tuple[int, int, int, int]] = {}
        phases = range(min(stride, width - length + 1)) if width >= length else range(0)
        if width >= length:
            replica_slots = len(phases) * height
            phase_arr = np.fromiter(phases, dtype=np.int64, count=len(phases))
            band_rows = _band_rows(width, np.dtype(src.dtypes[0]).itemsize)
            for row_off in range(r0, r1, band_rows):
                rows = min(band_rows, r1 - row_off)
                band = src.read(1, window=Window(c0, row_off, width, rows))
                band_bits = _bit_view(band)
                for phase in phase_arr:
                    slab = band_bits[:, phase:phase + length]
                    if slab.shape[0] == 0:
                        continue
                    values = _group_values(slab, spb)
                    uniq, counts = np.unique(values, return_counts=True)
                    order = np.argsort(counts)[::-1]
                    top_n = int(counts[order[0]])
                    second_n = int(counts[order[1]]) if order.size > 1 else 0
                    top_v = int(uniq[order[0]])
                    total += int(values.size)
                    # 跨行带合并：同一相位在不同行带的结果累加
                    if int(phase) in per_phase:
                        pv, ph, ps, pc = per_phase[int(phase)]
                        if pv == top_v:
                            per_phase[int(phase)] = (pv, ph + top_n, max(ps, second_n), pc + int(uniq.size))
                        elif top_n > ph:
                            per_phase[int(phase)] = (top_v, top_n, ph, pc + int(uniq.size))
                        else:
                            per_phase[int(phase)] = (pv, ph, max(ps, top_n), pc + int(uniq.size))
                    else:
                        per_phase[int(phase)] = (top_v, top_n, second_n, int(uniq.size))

        # 选「单相位内一致性最高」的相位作为结论
        best_phase = None
        best_value, best_hits, runner_up, distinct = None, 0, 0, 0
        if per_phase:
            best_phase = max(per_phase, key=lambda p: (per_phase[p][1], -p))
            best_value, best_hits, runner_up, distinct = per_phase[best_phase]
        support = (best_hits / total) if total else 0.0
        # 可解码性必须基于**证据**，不能靠平局打破规则。
        # 实测教训：1×800 的薄窗口里真实值只拿到 1 票，与随机候选同票数，
        # 旧的 `max(key=(hits, value))` 按候选数值大小选胜者 → 解出 FCE1207C 这种错值，
        # 调用方无从分辨这是"检出的水印"还是"抽签抽出来的数"。
        # 判据只能是**实际票数**（``groups_checked`` 是相位扫描总次数，不是票数密度，
        # 拿它算比例会把所有合规窗口都误判为不可信）。
        unique_top = best_hits > runner_up
        decodable = bool(replica_slots) and best_hits >= MIN_DECODE_VOTES and unique_top
        verified = bool(best_value == watermark) and decodable
        ambiguous = bool(per_phase) and not unique_top and best_value == watermark
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
            "distinct_candidates": distinct,
            "matched": int(best_value == watermark),
            "decoded": f"{best_value:08X}" if best_value is not None else None,
            "expected": f"{watermark:08X}",
            "support": round(support, 6),
            "rate": round(support, 6),
            "w_min": stride - 1 + length,
            "min_votes_required": MIN_DECODE_VOTES,
            "votes_for_best": best_hits,
            "votes_for_runner_up": runner_up,
            "best_phase": best_phase,
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
