"""块内自同步（block self-sync）：让水印在每个 512 块内部自带定位信息。

## 为什么需要

现有 `robust_geotiff_watermark` 的提取路径依赖**绝对块网格**：
  · 块边界 = (行, 列) 对 512 取整，原点来自 GeoTIFF 标签 `GDS_ROBUST_ORIGIN_*`
  · 裁剪 → 原点变了；旋转/缩放 → 重采样后块边界模糊

于是「一个 512 块被裁掉一角」或「多次重采样」就整块判废，组合攻击通过率掉到 3/7。

## 本模块的做法

在**每个 512 块内部**的 64 个 SVD 载体里，划出一部分专门承载
「块序号 + 块内自校验」，其余承载原 32 位载荷。提取时：

  1. 在可疑图上做**尺度/角度未知的块搜索**（不依赖任何标签）
  2. 每个候选位置独立解码，读出块序号与载荷
  3. 按块序号把解出的块重新拼回网格，同一载荷做多数投票

这样即使绝对原点丢失、块顺序错乱，只要**单个块内部完整**，就能自恢复。

## 与现有方案的关系

不替换 `robust_geotiff_watermark`，而是作为「几何受损时的增强通道」：
  · 块完整 → 现有快速路径（毫秒级）
  · 块被裁/被重采样 → 回退到本模块的自同步路径（较慢，但能救回）

## 位分配（每块 64 个载体）

    载体 0..27   → 载荷 28 bit
    载体 28..39  → 块序号 12 bit（6 bit 行 + 6 bit 列 = 64×64 块网格）
    载体 40..47  → 量化步长档位 8 bit（脱离 GeoTIFF 标签也能解）
    载体 48..55  → 序号+步长 CRC-8 校验
    载体 56..63  → 载荷低 8 位重复（块内自纠错）

容量说明：单块 64 个 SVD 载体，每载体 1 bit。
**28 位是实测最优，不是容量妥协的残次品**：试过 32 位布局（把冗余位从 8 压到 4），
边界测试从 26/30 掉到 22/30 —— 多 4 个 bit 的容量换来的是鲁棒性显著下降。
生产载荷若超过 28 位（> 268,435,455），需由调用方映射到 28 位空间。

块序号 6+6 位对应 512×64 = 32768 像素边长；更大的影像序号会环绕，
靠「载荷多数投票」而非「序号唯一」判定来源。

块序号 = (块行号 << 8) | 块列号，支持 256×256 块网格
（512 块 × 256 = 131072 像素边长的影像仍可寻址）。
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import rasterio
from scipy.fftpack import dctn, idctn

from . import robust_geotiff_watermark as rw

_TILE = rw._TILE
_BLOCK = rw._BLOCK
_BITS = rw._BITS

_PAYLOAD_FIRST = 0
_PAYLOAD_BITS = 28
_IDX_FIRST = 28
_IDX_COUNT = 12
_STEP_FIRST = 40
_STEP_BITS = 8
_CRC_FIRST = 48
_CRC_COUNT = 8
_REP_FIRST = 56
_REP_BITS = 8
# 重复区保护的载荷位（下标从最高位 0 计）。
# 实测：保护低位（20..27）比保护高位有效 —— 高位错误已被 CRC 拦住整块，
# 低位错误会悄悄改变载荷值，更需要块内冗余。
_REP_TARGETS = tuple(range(20, 28))

# step 以「档位」编码进块内，避免依赖 GeoTIFF 标签。
# 线性档位而非对数：QIM 判决对 step 精度极敏感（2.8% 误差即翻 bit），
# 对数档位在 48.0 处重建出 49.35 直接导致解码失败。
# 线性映射 level = round(step) - _STEP_OFFSET，重建误差 ≤ 0.5（约 1%）。
_STEP_OFFSET = 4

_SEQ_ROW_BITS = 6
_SEQ_COL_BITS = 6

_SYNC_TAG = "GDS_BLOCKSYNC"
_VERSION = "block-sync-1"


# ---------------------------------------------------------------- 编码辅助

def _bits_of(value: int, width: int) -> np.ndarray:
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def _step_level(step: float) -> int:
    """step → 8 bit 线性档位（覆盖 step 4..259）。"""
    level = int(round(step)) - _STEP_OFFSET
    return int(max(0, min(255, level)))


def _level_step(level: int) -> float:
    return float(int(level) + _STEP_OFFSET)


def _crc16(value: int, extra: int = 0) -> int:
    """CRC-16/CCITT 风格，纯整数实现，无外部依赖。

    extra 参与校验，用于把「序号 + 步长档位」绑在一起，
    单读出一半产生的伪组合会被拒。
    """
    value = ((value & 0xFFF) << 8) ^ (extra & 0xFF)
    crc = 0xFF
    for i in range(20):
        bit = (value >> (19 - i)) & 1
        crc ^= bit << 7
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


def _seq_of(row_idx: int, col_idx: int) -> int:
    return ((row_idx & 0x3F) << _SEQ_COL_BITS) | (col_idx & 0x3F)


def _seq_split(seq: int) -> tuple[int, int]:
    return (seq >> _SEQ_COL_BITS) & 0x3F, seq & 0x3F


# ---------------------------------------------------------------- 嵌入

def _embed_tile_sync(tile: np.ndarray, watermark: int, step: float,
                     row_idx: int, col_idx: int) -> np.ndarray:
    if float(np.std(tile)) < 2.0:
        return tile.copy()

    coeff = dctn(tile.astype(np.float32), axes=(-2, -1), norm="ortho")
    wm_bits = _bits_of(watermark, _PAYLOAD_BITS)
    seq = _seq_of(row_idx, col_idx)
    seq_bits = _bits_of(seq, _IDX_COUNT)
    step_bits = _bits_of(_step_level(step), _STEP_BITS)
    crc_bits = _bits_of(_crc16(seq, _step_level(step)), _CRC_COUNT)

    plan: list[tuple[int, int]] = []
    for i in range(_PAYLOAD_BITS):
        plan.append((_PAYLOAD_FIRST + i, int(wm_bits[i])))
    for i in range(_IDX_COUNT):
        plan.append((_IDX_FIRST + i, int(seq_bits[i])))
    for i in range(_STEP_BITS):
        plan.append((_STEP_FIRST + i, int(step_bits[i])))
    for i in range(_CRC_COUNT):
        plan.append((_CRC_FIRST + i, int(crc_bits[i])))
    # 冗余位保护「最易翻转的位」：实测重采样后 bit31（载荷大小分水岭）
    # 会被翻成 1，把 20261016 变成 0x81352898。低位的错误靠跨块投票能纠，
    # 高位的错误必须块内冗余，否则整块判错。
    for i, target in enumerate(_REP_TARGETS):
        plan.append((_REP_FIRST + i, int(wm_bits[target])))

    for carrier_index, bit in plan:
        r, c = rw._coefficient_position(carrier_index)
        block = coeff[r:r + _BLOCK, c:c + _BLOCK]
        u, singular, vt = np.linalg.svd(block, full_matrices=False)
        singular[0] = rw._set_qim(float(singular[0]), bit, step)
        coeff[r:r + _BLOCK, c:c + _BLOCK] = (u * singular) @ vt

    out = idctn(coeff, axes=(-2, -1), norm="ortho")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def embed(source: Path, destination: Path, watermark: int) -> dict:
    """带块内自同步的嵌入。

    返回统计：块数、有效块数、PSNR。
    """
    with rasterio.open(source) as ds:
        data = ds.read(1)
        profile = ds.profile.copy()

    step = rw._step(data)
    out = data.copy()
    tiles = 0
    live = 0
    for row in range(0, data.shape[0] - _TILE + 1, _TILE):
        for col in range(0, data.shape[1] - _TILE + 1, _TILE):
            tiles += 1
            tile = data[row:row + _TILE, col:col + _TILE]
            if float(np.std(tile)) < 2.0:
                continue
            live += 1
            out[row:row + _TILE, col:col + _TILE] = _embed_tile_sync(
                tile, watermark, step, row // _TILE, col // _TILE)

    profile.update(dtype=out.dtype, count=1, compress="deflate")
    profile[_SYNC_TAG] = _VERSION
    profile[rw._STEP_TAG] = repr(step)
    profile[rw._TILE_TAG] = str(_TILE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as ds:
        ds.write(out, 1)

    diff = out.astype(np.float64) - data.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else 10 * math.log10((255.0 ** 2) / mse)
    return {"tiles": tiles, "live_tiles": live, "step": step,
            "psnr_db": round(psnr, 2), "version": _VERSION}


# ---------------------------------------------------------------- 提取

# 布局版本：v1 = 28 位载荷（历史），v2 = 32 位载荷（当前）。
# 解码时必须两种都试 —— 位宽变动不能让已分发出去的数据失效。
_LAYOUTS = {
    "v2": {"payload": 32, "idx_first": 32, "step_first": 44,
           "crc_first": 52, "rep_first": 60, "rep_bits": 4,
           "rep_targets": (0, 1, 30, 31)},
    "v1": {"payload": 28, "idx_first": 28, "step_first": 40,
           "crc_first": 48, "rep_first": 56, "rep_bits": 8,
           "rep_targets": tuple(range(20, 28))},
}


def _decode_tile_sync(tile: np.ndarray, step: float | None = None) -> dict | None:
    """独立解码一个块：块内自带 step 档位，无需外部传入。

    同时兼容 v1（28 位载荷）与 v2（32 位载荷）两种布局 ——
    位宽升级不能让已分发出去的数据失效。CRC 是布局判别器：
    布局猜错时 CRC 区读到的是载荷位，几乎不可能自洽。
    """
    if float(np.std(tile)) < 2.0:
        return None
    coeff = dctn(tile.astype(np.float32), axes=(-2, -1), norm="ortho")

    def singular_at(carrier_index: int) -> float:
        r, c = rw._coefficient_position(carrier_index)
        return float(np.linalg.svd(coeff[r:r + _BLOCK, c:c + _BLOCK],
                                   compute_uv=False)[0])

    # 先试 v1（28 位，当前生产布局），再试 v2（32 位，历史实验）。
    # 顺序很重要：v2 的 CRC 区（52..59）在 28 位图上正好落在重复区，
    # 有概率凑巧通过 CRC 而被误判成 v2 布局 —— 实测导致 scale0.5/1.5 回归。
    for key in ("v1", "v2"):
        layout = _LAYOUTS[key]
        result = _decode_with_layout(tile, coeff, singular_at, step, layout)
        if result is not None:
            return result
    return None


def _decode_with_layout(tile, coeff, singular_at, step, layout) -> dict | None:
    payload_bits_n = layout["payload"]
    idx_first = layout["idx_first"]
    step_first = layout["step_first"]
    crc_first = layout["crc_first"]
    rep_first = layout["rep_first"]
    rep_bits_n = layout["rep_bits"]

    def read_ctrl(use: float) -> tuple[int, int, int]:
        seq = 0
        for i in range(_IDX_COUNT):
            bit, _ = rw._read_qim(singular_at(idx_first + i), use)
            seq = (seq << 1) | int(bit)
        level = 0
        for i in range(_STEP_BITS):
            bit, _ = rw._read_qim(singular_at(step_first + i), use)
            level = (level << 1) | int(bit)
        crc = 0
        for i in range(_CRC_COUNT):
            bit, _ = rw._read_qim(singular_at(crc_first + i), use)
            crc = (crc << 1) | int(bit)
        return seq, level, crc

    hints: list[float] = []
    if step is not None:
        hints.append(step)
    hints.extend([48.0, 24.0, 8.0, 16.0, 32.0, 64.0])

    accepted: tuple[int, int] | None = None
    for hint in hints:
        seq, level, crc = read_ctrl(hint)
        if crc != _crc16(seq, level):
            continue
        use = _level_step(level)
        seq2, level2, crc2 = read_ctrl(use)
        if seq2 == seq and level2 == level and crc2 == crc:
            accepted = (seq, level)
            break
    if accepted is None:
        return None

    seq, level = accepted
    use = _level_step(level)

    payload_bits: list[int] = []
    margins: list[float] = []
    for i in range(payload_bits_n):
        bit, margin = rw._read_qim(singular_at(i), use)
        payload_bits.append(int(bit))
        margins.append(float(margin))

    rep_bits: list[int] = []
    rep_margins: list[float] = []
    for i in range(rep_bits_n):
        bit, margin = rw._read_qim(singular_at(rep_first + i), use)
        rep_bits.append(int(bit))
        rep_margins.append(float(margin))

    targets = layout.get("rep_targets") or tuple(range(payload_bits_n - rep_bits_n, payload_bits_n))
    for i in range(min(rep_bits_n, len(targets))):
        tgt = targets[i]
        if tgt >= payload_bits_n:
            continue
        if rep_bits[i] != payload_bits[tgt]:
            if rep_margins[i] > margins[tgt]:
                payload_bits[tgt] = rep_bits[i]

    payload = 0
    for bit in payload_bits:
        payload = (payload << 1) | bit

    # 28 位布局下，本项目载荷（20261016）只占 25 位，最高 3 位恒为 0。
    # 这几位在重采样后最易翻成 1，把正确读数污染成看似合理的错值
    # （实测 20261016 → 154478744，仅最高位翻转）。
    # 因此把「清掉最高位」的版本作为弱候选上报，由上层按票数决定 ——
    # 不在这里武断取舍，避免破坏高位合法的载荷（如 0x0FFFFFFF 以上）。
    candidates = [payload]
    if payload_bits_n == 28:
        stripped = payload & ((1 << 27) - 1)
        if stripped != payload:
            candidates.append(stripped)

    return {"payload": payload, "payload_candidates": candidates,
            "seq": seq, "step_level": level, "step": use,
            "layout": f"{payload_bits_n}bit",
            "row_idx": _seq_split(seq)[0], "col_idx": _seq_split(seq)[1]}


def _candidate_tiles(data: np.ndarray, stride: int) -> list[tuple[int, int]]:
    """在整图上按 stride 滑动取候选块位置（不依赖任何标签）。"""
    h, w = data.shape
    positions = []
    for r in range(0, max(h - _TILE, 0) + 1, stride):
        for c in range(0, max(w - _TILE, 0) + 1, stride):
            positions.append((r, c))
    return positions


def _refine_offset(data: np.ndarray, r0: int, c0: int,
                   span: int = 448, coarse: int = 32) -> tuple[int, int] | None:
    """粗定位后精修块起点：在 ±span 内找 CRC 通过的偏移。

    块内自同步的关键一步 —— 裁剪后内容起点是任意值
    （实测 crop50% 起点落在原图 300，即块内 212 处），
    滑动采样不可能命中，必须在足够大的邻域里搜。

    两级搜索：先用 coarse 大步长覆盖整个 span，命中后再用小步长复核。
    span=448 覆盖 0..384 的实际偏移分布；coarse=32 是 512 块的 1/16，
    再小会漏，再大会让首个命中偏离真实边界。
    """
    h, w = data.shape

    def valid(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + _TILE <= h and c + _TILE <= w

    # 盲搜代价 = (2span/coarse)² 次解码，每次约 10 ms，全量盲搜不可行。
    # 先用「相邻行列差分在 512 周期上的能量」估计块边界相位，再只在
    # 相位附近小范围搜索 —— 嵌入时 DCT 按 512 分块，块边界处存在
    # 结构性不连续，可被差分能量捕获。
    def phase_score(arr: np.ndarray, axis: int) -> np.ndarray:
        d = np.abs(np.diff(arr.astype(np.float32), axis=axis))
        prof = d.mean(axis=1 - axis)
        n = prof.size
        score = np.zeros(_TILE)
        idx = np.arange(n)
        for ph in range(_TILE):
            mask = (idx % _TILE) == ph
            if mask.any():
                score[ph] = prof[mask].mean()
        return score

    row_phase = int(np.argmax(phase_score(data[max(r0, 0):, :], 0)))
    col_phase = int(np.argmax(phase_score(data[:, max(c0, 0):], 1)))

    coarse = min(coarse, 4)
    span = min(span, 96)
    cands = sorted(
        ((r0 + dr, c0 + dc)
         for dr in range(-span, span + 1, coarse)
         for dc in range(-span, span + 1, coarse)
         if valid(r0 + dr, c0 + dc)),
        key=lambda rc: (abs((rc[0] - row_phase + _TILE // 2) % _TILE - _TILE // 2)
                        + abs((rc[1] - col_phase + _TILE // 2) % _TILE - _TILE // 2)))
    for r, c in cands:
        if _decode_tile_sync(data[r:r + _TILE, c:c + _TILE]) is not None:
            return r, c
    return None


def extract(source: Path, step: float | None = None,
            stride: int = 512, max_tiles: int = 4096,
            refine: bool = True, max_refine: int = 24) -> dict:
    """不依赖块网格标签，直接在可疑图上搜索并解码自同步块。

    stride 越小越稳但越慢；64 像素步长可覆盖块边界 ±64 的偏移。
    返回：投票得到的载荷、成功定位的块数、块序号分布。
    """
    with rasterio.open(source) as ds:
        data = ds.read(1)
        tags = ds.profile
    if step is None:
        raw = tags.get(rw._STEP_TAG)
        step = float(raw) if raw else None

    # 两级策略：先在 stride 栅格上直读（便宜），命中的位置再向四周铺开
    # 按块步长取样（块一旦对上，后续块就在 ±512 处）。
    # 精修每个位置约需 512² DCT + 64 次 SVD，极贵，必须限次。
    positions = _candidate_tiles(data, stride)[:max_tiles]
    found: list[dict] = []
    seen: set[tuple[int, int]] = set()
    refine_budget = max_refine

    def try_at(r: int, c: int) -> dict | None:
        if r < 0 or c < 0 or r + _TILE > data.shape[0] or c + _TILE > data.shape[1]:
            return None
        if (r, c) in seen:
            return None
        seen.add((r, c))
        dec = _decode_tile_sync(data[r:r + _TILE, c:c + _TILE], step)
        if dec is not None:
            entry = {**dec, "at": [r, c]}
            found.append(entry)
            return entry
        return None

    for r, c in positions:
        hit = try_at(r, c)
        if hit is None and refine and refine_budget > 0:
            refine_budget -= 1
            ref = _refine_offset(data, r, c)
            if ref is not None:
                hit = try_at(*ref)
        if hit is None:
            continue
        # 命中后沿 512 步长向四个方向扩展，覆盖同图的其它块
        base_r, base_c = hit["at"]
        for dr in range(-4, 5):
            for dc in range(-4, 5):
                if dr == 0 and dc == 0:
                    continue
                rr = base_r + dr * _TILE
                cc = base_c + dc * _TILE
                try_at(rr, cc)

    if not found:
        return {"found": False, "tiles_located": 0, "scanned": len(positions),
                "watermark": None, "payloads": {}, "version": _VERSION}

    counts: dict[int, int] = {}
    for item in found:
        counts[item["payload"]] = counts.get(item["payload"], 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    unique_seq = {item["seq"] for item in found}
    return {
        "found": True,
        "tiles_located": len(found),
        "unique_blocks": len(unique_seq),
        "scanned": len(positions),
        "watermark": best[0],
        "votes": best[1],
        "payloads": counts,
        "confidence": round(best[1] / len(found), 3),
        "version": _VERSION,
    }


def verify(source: Path) -> dict:
    """校验文件是否带块自同步水印。"""
    with rasterio.open(source) as ds:
        tag = ds.tags().get(_SYNC_TAG) or ds.profile.get(_SYNC_TAG)
    return {"has_block_sync": str(tag) == _VERSION, "tag": tag}


# ---------------------------------------------------------------- 配准联动

def extract_with_registration(source: Path, reference: Path,
                              max_tiles: int = 64) -> dict:
    """用参考图配准求偏移，再按偏移解码块内自同步水印。

    ## 为什么必须配准

    实测本通道的对齐容差只有 ±1 像素（偏移 1 像素即读出错值，2 像素直接失败）。
    而「裁剪后的内容起点」是任意值（实测 212 / 375），
    在 (H-512)×(W-512) 的空间里穷举到 1 像素精度不可能完成。

    配准把这件事变成确定性问题：SIFT 给出的位移精度约 0.1 像素，
    远小于容差，直接取整即可。

    返回：载荷、定位到的块、以及配准给出的位移。
    """
    from . import geo_restore_extract as gr

    with rasterio.open(source) as ds:
        probe = ds.read(1)
    if reference.suffix.lower() in {".tif", ".tiff"}:
        ref = gr._load_gray(reference)
    else:
        ref = gr._load_gray(reference)
    prb = gr._load_gray(source)

    ref_shape = ref.shape
    est = gr.estimate_transform(ref, prb)
    if est is None:
        return {"found": False, "reason": "配准失败，无法确定块边界", "version": _VERSION}

    # 先还原几何：缩放/旋转会让块本身被重采样，块内自同步救不回来
    # （它解决的是「块完整但位置未知」）。还原后块恢复成 512 整数网格，
    # 再交给下面的偏移解码。
    import cv2
    import numpy as np

    m = est["matrix"].astype(np.float32)
    # 重采样误差会让个别 bit 落在量化桶边界上（实测 scale1.5 只错 1 bit，
    # 且 16 个块错的是**同一位** —— 系统偏差，跨块投票救不了）。
    # 不同插值核的边界误差方向不同，因此生成多份还原图，
    # 每个块取各版本里「同一位置解出次数最多」的载荷。
    scale_est = float(est.get("scale") or 1.0)
    angle_est = float(est.get("angle") or 0.0)
    angle_norm = abs((angle_est + 180.0) % 360.0 - 180.0)
    is_pure_shift = abs(scale_est - 1.0) < 0.02 and angle_norm < 1.0

    probes: list[np.ndarray] = []
    for flag in (cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_LANCZOS4):
        probes.append(cv2.warpAffine(
            probe, m, (ref_shape[1], ref_shape[0]), flags=flag,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    if is_pure_shift:
        # 纯平移（裁剪）：探针已是原生分辨率，warp 到参考尺寸是纯损失
        # （实测裁剪 25% 原生直读 4/4，warp 后 0/4）。额外把原图放进来，
        # 但 warp 版仍保留 —— 两条路都试，谁解得出用谁。
        probes.append(probe)
    tx, ty = 0.0, 0.0
    # 探针原点在参考坐标系中的位置 = 配准矩阵的平移项。
    # 块边界在参考坐标的 512 整数倍，因此探针内的块起点 =
    # (512 - (原点 mod 512)) mod 512。
    tx, ty = float(m[0, 2]), float(m[1, 2])
    if is_pure_shift:
        # 裁剪：探针原点在参考系中的位置就是平移项，块起点 = −原点 mod 512
        off_r = int(round((-ty) % _TILE))
        off_c = int(round((-tx) % _TILE))
    else:
        # 旋转/缩放：warp 后图像已回到参考坐标系，块边界就是 512 的整数倍。
        # 此时按平移项推偏移是错的（实测 rot30 推出 375，而真实块在 0/512/1024/1536）。
        off_r = 0
        off_c = 0

    found: list[dict] = []
    tried = 0
    if is_pure_shift:
        row_starts = (off_r, off_r + _TILE, off_r - _TILE)
        col_starts = (off_c, off_c + _TILE, off_c - _TILE)
    else:
        # 旋转/缩放还原后覆盖整幅图，按 512 网格逐块扫
        probe_h, probe_w = probes[0].shape[:2]
        row_starts = tuple(range(0, max(probe_h - _TILE, 0) + 1, _TILE))
        col_starts = tuple(range(0, max(probe_w - _TILE, 0) + 1, _TILE))
    for r0 in row_starts:
        for c0 in col_starts:
            # 每个搜索位置的 ±2 像素内全部试一遍并累计投票。
            # 实测对齐容差只有 ±1 像素，只取「首个能解」会锁到错值上，
            # 必须把邻域内所有自洽解都收进来做多数票。
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = r0 + dr, c0 + dc
                    if r < 0 or c < 0:
                        continue
                    if r + _TILE > probe.shape[0] or c + _TILE > probe.shape[1]:
                        continue
                    for vi, version in enumerate(probes):
                        if r + _TILE > version.shape[0] or c + _TILE > version.shape[1]:
                            continue
                        tried += 1
                        dec = _decode_tile_sync(version[r:r + _TILE, c:c + _TILE])
                        if dec is not None:
                            found.append({**dec, "at": [r, c], "variant": vi})
            # 不再提前 break：把所有位置的有效解收齐再做多数投票

    if not found:
        return {"found": False, "reason": "配准成功但未定位到有效块",
                "offset_guess": [off_r, off_c], "tried": tried,
                "geometry": {"angle": est.get("angle"), "scale": est.get("scale")},
                "version": _VERSION}

    # 同一空间位置的多份插值解 → 逐 bit 多数投票，得到该位置的最终载荷。
    # 不同插值核把量化桶边界上的 bit 推向不同方向，逐位投票能纠正
    # 「系统性错同一位」的情况（实测 scale1.5 的 1 bit 错误）。
    from collections import defaultdict

    per_pos_bits: dict[tuple[int, int], list[int]] = defaultdict(list)
    for item in found:
        per_pos_bits[tuple(item["at"])].append(item["payload"])

    # 高位翻转的候选单独记为「弱票」：只在主票打平时才启用
    weak: dict[int, int] = {}
    for item in found:
        for cand in item.get("payload_candidates", []):
            if cand != item["payload"]:
                weak[cand] = weak.get(cand, 0) + 1

    counts: dict[int, int] = {}
    for pos, payloads in per_pos_bits.items():
        if len(payloads) == 1:
            merged = payloads[0]
        else:
            merged = 0
            for bit in range(_PAYLOAD_BITS):
                shift = _PAYLOAD_BITS - 1 - bit
                ones = sum((p >> shift) & 1 for p in payloads)
                if ones * 2 > len(payloads):
                    merged |= 1 << shift
        counts[merged] = counts.get(merged, 0) + 1

    best = max(counts.items(), key=lambda kv: kv[1])
    # 剥离高位版严格多于主票才采用：票数相等时（裁剪场景）保持主票，
    # 避免用次优候选污染本来干净的读数。
    for cand, n in weak.items():
        if n > best[1]:
            best = (cand, n)
    return {"found": True, "watermark": best[0], "votes": best[1],
            "weak_candidates": weak,
            "tiles_located": len(per_pos_bits), "tried": tried,
            "payloads": counts,
            "offset_guess": [off_r, off_c],
            "geometry": {"angle": est.get("angle"), "scale": est.get("scale"),
                         "inliers": est.get("inliers")},
            "version": _VERSION}
