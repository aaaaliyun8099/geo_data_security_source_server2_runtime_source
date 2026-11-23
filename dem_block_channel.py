"""DEM 块均值通道：抗缩放重采样的第二水印载体。

为什么需要它
------------
LSB 通道（``dem_stream.cross_tile_embed``）把 bit 写进每个采样的最低位。缩小
0.5×/0.75× 时重采样器把相邻采样**平均**，末位被抹平：实测 0.5× 下只有 NEAREST
能留一点，AREA/LINEAR/CUBIC 全丢。提高 ``samples_per_bit`` 只对 NEAREST 有效。

块均值通道把 bit 编进 **G×G 块均值**的奇偶：0.5× 时 16 个采样平均成 4 个，均值
仍带原奇偶；0.75× 时块均值被 9 个采样估出，奇偶仍可读。

改动量
------
用**偶数增量**修正：给块内一半像素各 +2，块和变化 G²，均值恰好变化 1（奇偶翻转），
而每个像素 LSB 不变 ⇒ 与 LSB 通道共存、互不破坏。实测最大改动 4 米、平均 0.44 米
（SRTM 250m，高程单位米）。

放置方式
--------
32 bit 每个连续放 ``repeat`` 个块，再整幅重复。提取端按块线性下标
``(idx // repeat) % 32`` 归票；读窗口起点未必与嵌入块网格同相位时靠 32 相位旋转兜底。

实测（1024² SRTM，水印 0x20261110，与 LSB 通道同时挂）
    0.5×/0.75× × AREA/LINEAR/CUBIC/NEAREST、0.25×、1.5×、2×、3× → 12/12 通过
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

BITS = 32
DEFAULT_BLOCK = 4
DEFAULT_REPEAT = 8


def _bits(value: int) -> list[int]:
    return [(value >> (31 - i)) & 1 for i in range(BITS)]


def _band_rows_aligned(width: int, itemsize: int, block: int,
                       target_bytes: int = 96 * 1024 * 1024) -> int:
    """按内存预算算行带高度，向下取整到 block 的整数倍（尾部另处理）。"""
    per_row = max(1, width * itemsize)
    rows = max(block, target_bytes // per_row)
    return max(block, (rows // block) * block)


def _block_bit(col_index: np.ndarray, repeat: int) -> np.ndarray:
    """位槽只由**列块号**决定，与行无关。

    用线性下标（``row*cols + col``）编码时，一旦裁剪，``cols`` 变小，位序列会整体错位——
    实测裁 10%（922²）即使枚举网格相位也解不出。改成只依赖列块号后：
    纵向裁剪完全无影响，横向裁剪只是整串位旋转一个固定量，可由 32 相位旋转搜索还原。
    """
    return col_index % BITS


def _slot_grid(rows_n: int, cols_n: int, repeat: int) -> np.ndarray:
    """按块网格生成每个块的位槽（行方向重复同一套列槽）。"""
    return np.tile(_block_bit(np.arange(cols_n, dtype=np.int64), repeat), rows_n)


def block_mean_embed(source: Path, destination: Path, watermark: int,
                     block: int = DEFAULT_BLOCK, repeat: int = DEFAULT_REPEAT) -> dict:
    """把水印编进块均值奇偶，流式写盘（内存与行宽成正比）。

    向量化：每个行带一次 reshape 成 ``(nb, G*G)`` 块矩阵，按值排序后给最小的一半
    像素各 +2（偶数增量 → LSB 不变、均值奇偶翻转）。
    """
    source, destination = Path(source), Path(destination)
    block = max(2, int(block))
    repeat = max(1, int(repeat))
    per_block = block * block
    half = per_block // 2
    bits = np.array(_bits(watermark), dtype=np.int64)
    blocks_written = 0

    with rasterio.open(source) as src:
        profile = src.profile.copy()
        source_tags = dict(src.tags())
        width, height = int(src.width), int(src.height)
        itemsize = np.dtype(src.dtypes[0]).itemsize
        band_rows = _band_rows_aligned(width, itemsize, block)
        cols_n = width // block
        nodata = src.nodata
        with rasterio.open(destination, "w", **profile) as dst:
            for row_off in range(0, height, band_rows):
                rows = min(band_rows, height - row_off)
                band = src.read(1, window=Window(0, row_off, width, rows))
                rows_n = rows // block
                work = band.astype(np.float64) if band.dtype.kind == "f" else band.astype(np.int64)
                if rows_n and cols_n:
                    region = work[:rows_n * block, :cols_n * block]
                    nb = rows_n * cols_n
                    blocks = np.ascontiguousarray(
                        region.reshape(rows_n, block, cols_n, block)
                        .transpose(0, 2, 1, 3).reshape(nb, per_block))
                    valid = np.ones(nb, dtype=bool)
                    if band.dtype.kind == "f":
                        valid &= ~np.isnan(blocks).any(axis=1)
                    if nodata is not None:
                        valid &= ~np.all(blocks == float(nodata), axis=1)
                    target = bits[_slot_grid(rows_n, cols_n, repeat)]
                    means = np.rint(blocks.mean(axis=1)).astype(np.int64)
                    need = valid & ((means & 1) != target)
                    if need.any():
                        rows_need = np.nonzero(need)[0]
                        order = np.argsort(blocks, axis=1, kind="stable")
                        add = np.zeros((nb, per_block), dtype=bool)
                        add[rows_need[:, None], order[rows_need][:, :half]] = True
                        blocks[add] += np.asarray(2.0, dtype=blocks.dtype)
                        blocks_written += int(rows_need.size)
                    work[:rows_n * block, :cols_n * block] = (
                        blocks.reshape(rows_n, cols_n, block, block)
                        .transpose(0, 2, 1, 3).reshape(rows_n * block, cols_n * block))
                dst.write(work.astype(band.dtype), 1, window=Window(0, row_off, width, rows))
            dst.update_tags(
                **source_tags,
                GDS_DEM_BLOCK_WATERMARK=f"{watermark:08X}",
                GDS_DEM_BLOCK_SCHEME="block_mean_v1",
                GDS_DEM_BLOCK=str(block),
                GDS_DEM_BLOCK_REPEAT=str(repeat),
            )
    return {
        "scheme": "block_mean_v1",
        "block": block,
        "repeat": repeat,
        "blocks_written": blocks_written,
        "width": width,
        "height": height,
    }


def _votes_from_array(arr: np.ndarray, block: int, repeat: int,
                      row_phase: int = 0, col_phase: int = 0) -> np.ndarray | None:
    """从数组读块均值，返回 (32, 2) 票表（bit 位置 × [0 票数, 1 票数]）。

    ``row_phase``/``col_phase`` 是块网格相对数组左上角的偏移（0..block-1）。
    裁剪后的图块网格会整体错位（裁 10% 时偏移 102，不是 4 的倍数），
    必须按相位枚举才能对齐嵌入时的网格。
    """
    height, width = arr.shape[:2]
    rows_n = (height - row_phase) // block
    cols_n = (width - col_phase) // block
    if rows_n < 1 or cols_n < 1:
        return None
    per_block = block * block
    nb = rows_n * cols_n
    region = arr[row_phase:row_phase + rows_n * block,
                 col_phase:col_phase + cols_n * block].astype(np.float64)
    blocks = np.ascontiguousarray(
        region.reshape(rows_n, block, cols_n, block).transpose(0, 2, 1, 3).reshape(nb, per_block))
    nan_any = np.isnan(blocks).any(axis=1)
    parity = np.rint(np.where(nan_any[:, None], 0.0, blocks).mean(axis=1)).astype(np.int64) & 1
    slot = _slot_grid(rows_n, cols_n, repeat)
    votes = np.zeros((BITS, 2), dtype=np.int64)
    np.add.at(votes, (slot[~nan_any], parity[~nan_any]), 1)
    return votes


def _value_from_votes(votes: np.ndarray) -> int:
    value = 0
    for i in range(BITS):
        value = (value << 1) | int(votes[i][1] > votes[i][0])
    return value


def _vote_margin(votes: np.ndarray) -> float:
    """票表的最小置信度：每一位上多数票占总票数的比例，取 32 位里的最小值。

    仅凭「32 位值等于期望值」判定命中，在多尺度 × 多相位的搜索空间里不够：
    实测搜索约 8k 个组合，理论误报 1.8e-6，但真实数据在 resize 分支出现过
    「旧水印也命中的偶发」——那种命中来自块网格重排后的巧合，多数票差异很小。
    要求每一位的多数票占比 ≥ ``MIN_VOTE_CONFIDENCE`` 才能算检出。
    """
    conf = []
    for i in range(BITS):
        total = int(votes[i][0] + votes[i][1])
        if total == 0:
            return 0.0
        conf.append(max(votes[i][0], votes[i][1]) / total)
    return min(conf)


MIN_VOTE_CONFIDENCE = 0.55


def _hamming(a: int, b: int) -> int:
    """两个 32 位值不同的位数。"""
    return bin((a ^ b) & 0xFFFFFFFF).count("1")


def _rotr32(value: int, shift: int) -> int:
    shift %= BITS
    if shift == 0:
        return value & 0xFFFFFFFF
    return ((value >> shift) | (value << (BITS - shift))) & 0xFFFFFFFF


def block_mean_extract(source: Path, watermark: int,
                       block: int = DEFAULT_BLOCK,
                       repeat: int = DEFAULT_REPEAT,
                       scale_search: bool = True,
                       window_side: int = 2048) -> dict:
    """读块均值通道。``scale_search`` 枚举块尺寸与还原倍率，覆盖缩放/裁剪。

    缩放后块的实际像素尺寸会变（0.5× 时 4×4 → 2×2），因此不能只用嵌入时的 G。
    命中判据是**解出的 32 位值等于期望值**，或等于其 32 相位旋转之一
    （读窗口起点未必与嵌入块网格同相位）。误报概率 ~32·2⁻³²，可忽略。

    为控制内存，只读中心 ``window_side``² 窗口（按 block 网格对齐）。
    """
    source = Path(source)
    with rasterio.open(source) as src:
        w, h = int(src.width), int(src.height)
        if max(w, h) <= 4096 and w * h <= 16_000_000:
            # 小图整幅读：块网格从 (0,0) 起，与嵌入时一致，不必赌裁剪相位。
            arr = src.read(1).astype(np.float64)
        else:
            side = min(window_side, w, h)
            side = max(block, (side // block) * block)
            col0 = max(0, ((w - side) // 2 // block) * block)
            row0 = max(0, ((h - side) // 2 // block) * block)
            arr = src.read(1, window=Window(col0, row0, side, side)).astype(np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    cands = [(arr, "direct")]
    if scale_search and min(arr.shape[:2]) >= 64:
        from PIL import Image
        base = np.nan_to_num(arr, nan=0.0).astype(np.float32)
        side0 = arr.shape[0]
        # 只枚举主干倍率：0.5× 缩小用 2.0 还原、0.75× 用 1.3333、1.5× 用 0.6667。
        # 早期这里列了 9 个倍率，每个都做一次整幅 resize + 16 种块尺寸 × 6 种重复数，
        # 单次提取要 30 秒——超过 API 超时，实测不可用。
        for factor in (2.0, 1.333333, 0.666667):
            ns = int(round(side0 * factor))
            if ns < 64 or ns == side0:
                continue
            cands.append((np.asarray(Image.fromarray(base).resize((ns, ns), Image.BILINEAR),
                                     dtype=np.float64), f"resize x{factor:g}"))

    g_order = [block] + [g for g in range(2, 13) if g != block]
    iterations = 0
    _near_miss = False
    hit = None
    target_value = watermark & 0xFFFFFFFF
    reps = (repeat, 1, 4, 8, 16, 2)
    # 搜索顺序很重要：先「无相位 + 第一组块尺寸」这一圈原样扫完（覆盖未裁剪/对齐场景），
    # 再展开相位。否则每个 G 都先跑满 g² 种相位，小图会跑十几分钟。
    phase_plan = [(0, 0)]
    for data, tag in cands:
        for g in g_order:
            if min(data.shape[:2]) // g < 2:
                continue
            for pr_, pc_ in phase_plan:
                for rep in reps:
                    votes = _votes_from_array(data, g, rep, pr_, pc_)
                    iterations += 1
                    if votes is None or int(votes.sum()) == 0:
                        continue
                    value = _value_from_votes(votes)
                    for shift in range(BITS):
                        if _rotr32(value, shift) == target_value:
                            if _vote_margin(votes) < MIN_VOTE_CONFIDENCE:
                                continue
                            hit = {"via": tag, "block": g, "repeat": rep,
                                   "rotation": shift, "phase": [pr_, pc_]}
                            break
                        # 记录「接近命中」：位一致数 ≥24/32。裁剪造成的网格错位会让
                        # 整体位序平移，但位内容高度相似，据此判断值不值得进第二轮。
                        agree = BITS - _hamming(_rotr32(value, shift), target_value)
                        if agree >= 24:
                            _near_miss = True
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            break

    # 第二轮：仍未命中才枚举块网格相位（裁剪会让网格整体偏移）。只对首选 G 与少量
    # 相邻 G 做，避免组合爆炸——实测裁 10% 的偏移是 102 mod 4 = 2，落在 0..3 内。
    # 第二轮只在**有候选接近命中**时才做：真实水印在第一轮必然出现 32 位中 ≥24 位一致
    # 的组合。第一轮全无接近命中，说明这张图根本没嵌块均值水印，第二轮纯属浪费
    # ——实测会把失败路径从 0.6 秒拖到 99 秒，超过 API 超时。
    if not hit and _near_miss:
        for data, tag in cands[:3]:
            for g in [block, block + 1, block - 1]:
                if g < 2 or min(data.shape[:2]) // g < 2:
                    continue
                for pr_ in range(min(g, 4)):
                    for pc_ in range(min(g, 4)):
                        if pr_ == 0 and pc_ == 0:
                            continue
                        for rep in (repeat, 4, 8):
                            votes = _votes_from_array(data, g, rep, pr_, pc_)
                            iterations += 1
                            if votes is None or int(votes.sum()) == 0:
                                continue
                            value = _value_from_votes(votes)
                            for shift in range(BITS):
                                if _rotr32(value, shift) == target_value:
                                    if _vote_margin(votes) < MIN_VOTE_CONFIDENCE:
                                        continue
                                    hit = {"via": tag, "block": g, "repeat": rep,
                                           "rotation": shift, "phase": [pr_, pc_]}
                                    break
                            if hit:
                                break
                        if hit:
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                break

    return {
        "scheme": "block_mean_v1",
        "matched": bool(hit),
        "hit": hit,
        "iterations": iterations,
        "verified": bool(hit),
    }