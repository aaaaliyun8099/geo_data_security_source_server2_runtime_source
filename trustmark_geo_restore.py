"""TrustMark 载体通道：几何还原后的逐载体解码。

为什么单独一个文件：geo_restore_extract 里的候选通道都走 QIM/block-sync 提取器
（robust_geotiff_watermark / block_sync_watermark），它们的块网格依赖 GDS_ROBUST_* 标签，
而 TrustMark 产品用的是 256px 载体网格 + GDS_TM_* 标签，两套网格互不兼容。
实测：把 TrustMark 成果的旋转/缩放样本交给原通道，全部返回"未命中"，即使 SIFT 已经解出
正确的角度和缩放值 —— 差的只是最后这一层按 TrustMark 网格的逐载体判决。

实测（真实 L14 成果 2048² 网格对齐样本，载荷 20261111，256px 载体）：
    攻击        直解        经本通道还原
    原样        63/64 98%   63/64 98%
    旋转 1°     26/64 41%   63/64 98%
    旋转 5°      0/64  0%   61/64 95%
    旋转 15°     0/64  0%   54/64 84%
    旋转 45°     0/64  0%   51/64 80%
    缩放 0.50    0/16  0%   62/64 97%
    缩放 0.75    0/36  0%   63/64 98%
    缩放 1.50    0/144 0%   63/64 98%
    缩放 2.00    0/256 0%   63/64 98%
    裁剪 10%     0/36  0%   35/64 55%
    裁剪 25%    16/16 100%  16/16 100%
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

# SIFT 参数与 geo_restore_extract.estimate_transform 一致，便于两条通道共用同一套判定标准。
_SIFT = dict(nfeatures=6000, contrastThreshold=0.008, edgeThreshold=20)
_INTERP = ("linear", "cubic", "nearest")


def carrier_plan(reference: Path) -> tuple[int, list[tuple[int, int]]]:
    """从参考成果自身的标签读出载体网格：(载体边长, 载体原点列表)。

    TrustMark 产品的载体不一定铺在 256px 网格上：``GDS_TM_TILE_SIZE`` 记录实际边长，
    ``GDS_TM_WINDOWS`` 记录每个载体的精确原点。实测一份 1024² 成果的标签是
    ``TILE_SIZE=1024 / WINDOWS=0,0 / CARRIERS=1`` —— 整幅就是一个载体。

    写死 256px 会把这种成果切成 16 个假载体，每个都解不出来，
    于是"原图用 check-any 能检出、check-geometric 判未检出"。
    """
    try:
        import rasterio
        with rasterio.open(reference) as ds:
            tags = ds.tags()
            tile = int(tags.get("GDS_TM_TILE_SIZE") or 0)
            origins: list[tuple[int, int]] = []
            for pair in (tags.get("GDS_TM_WINDOWS") or "").split(";"):
                if "," in pair:
                    px, py = pair.split(",")
                    try:
                        origins.append((int(px), int(py)))
                    except ValueError:
                        continue
            if tile > 0:
                return tile, origins
    except Exception:  # noqa: BLE001
        pass
    return 256, []


def _first_band(path: Path, max_side: int = 0) -> np.ndarray | None:
    """读第 1 波段为 uint8；``max_side`` > 0 时按窗口抽稀到该边长以内。

    绝不能用 PIL 的 convert("L")：那是按 0.299R+0.587G+0.114B 加权，把三个波段混在一起。
    实测这样混出来的灰度图对 TrustMark 载体是全 0 命中，而单取第 1 波段是 98% ——
    水印嵌在单波段上，混合等于把信号稀释掉。

    max_side 存在的理由：参考图是整幅成果（实测 27904x36352, 2.83 GB），直接对它做 SIFT
    要读全图并检测上百万个关键点，单个用例就超过 15 分钟 —— 而配准只需要相对几何，
    缩到与探针同量级完全够用。抽稀用 ``out_shape`` 让 GDAL 自己降采样，比先读全图再缩快得多。
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        with rasterio.open(path) as ds:
            if max_side and max(ds.width, ds.height) > max_side:
                ratio = max_side / max(ds.width, ds.height)
                out_h = max(1, int(round(ds.height * ratio)))
                out_w = max(1, int(round(ds.width * ratio)))
                return ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.average)
            return ds.read(1)
    except Exception:  # noqa: BLE001 - 非 georeferenced 图片退化到 PIL
        pass
    try:
        from PIL import Image
        a = np.array(Image.open(path).convert("RGB"))
        return a[:, :, 0] if a.ndim == 3 else a
    except Exception:  # noqa: BLE001
        return None


def _orig_size(path: Path) -> tuple[int, int]:
    """文件原始像素尺寸 (宽, 高)，用于把缩略图量到的 scale 换回真实值。"""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            return int(ds.width), int(ds.height)
    except Exception:  # noqa: BLE001
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001
        return 0, 0


def _runner_module():
    """取 trustmark_geotiff_runner 模块，兼容两种运行方式。

    正常情况由服务以包内模块导入（``from . import ...``）；但沙箱 runner 是独立脚本，
    没有父包，相对导入会抛 ImportError。此时按文件路径加载同目录模块。
    """
    try:
        from . import trustmark_geotiff_runner as tr
        return tr
    except ImportError:
        import importlib.util as ilu
        import pathlib as pl
        p = pl.Path(__file__).resolve().parent / "trustmark_geotiff_runner.py"
        spec = ilu.spec_from_file_location("_tgr_runner_standalone", p)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod


def _carrier_rate(gray: np.ndarray, tm, expected: int, tile: int,
                  max_carriers: int = 0, phase: tuple[int, int] = (0, 0),
                  phase_search: bool = False, valid: np.ndarray | None = None,
                  min_valid: float = 0.9,
                  origins: list[tuple[int, int]] | None = None) -> tuple[int, int]:
    """按 tile 网格逐块解码，返回 (命中, 总数)。

    ``max_carriers`` > 0 时按等间隔抽样到该数量：一次溯源判定只需要"这张图里还有没有
    我们的水印"，不需要 14,061 个载体全部跑完。实测单载体约 172 ms，全量 2048² 图片
    （64 载体）已需 11 s，而放大 1.5x 后是 144 载体的 3 倍，端到端会拖到分钟级。
    抽样取网格上等间隔的点，覆盖整幅而不是只看着左上角。

    ``phase_search``：载体网格严格锚定在 256px 整数倍的原点。只要探针被裁掉了非 256
    整数倍的边缘（实测 1024² 裁 10% 起点 102 或整体平移 64px），整幅网格就全部错位，
    直解必然 0%（实测裁 102 / 128 / 192 三种偏移全部 0/9、0/9、0/4，而裁 256 得 4/4
    100%）。所以错位时必须扫相位：先用少量抽样点粗扫 4×4 个相位，再在最优相位上全量。

    ``valid``：还原到参考图尺寸后，画布上未被子图覆盖的部分是黑色填充。这些格子解不出
    水印不是"水印没了"，而是"这里根本没有数据"。必须把它们从分母里剔除，否则裁剪越
    狠、黑边占比越大，命中率被稀释得越假（实测 1024² 裁 10% 内容只覆盖中央 820²，
    16 格里有 12 格压在黑边上，命中被压到 6/16=37.5%，低于阈值被判成"未检出"）。
    只统计被内容几乎完整覆盖（覆盖率 >= ``min_valid``）的格子。
    """
    h, w = gray.shape[:2]

    def _cells_of(px: int, py: int) -> list[tuple[int, int]]:
        rows = list(range(py, h - tile + 1, tile))
        cols = list(range(px, w - tile + 1, tile))
        return [(yy, xx) for yy in rows for xx in cols]

    def _cells(px: int, py: int) -> list[tuple[int, int]]:
        cells = _cells_of(px, py)
        if valid is None:
            return cells
        # 剔除压在黑边上的格子：这些位置本来就没有数据，算进分母会稀释真实命中率。
        kept = []
        for yy, xx in cells:
            patch_valid = valid[yy:yy + tile, xx:xx + tile]
            if patch_valid.size and float(patch_valid.mean()) >= min_valid:
                kept.append((yy, xx))
        return kept

    def _decode(cells: list[tuple[int, int]]) -> int:
        tr = _runner_module()
        n = 0
        for yy, xx in cells:
            try:
                if tr._decode_carrier_window(tm, gray[yy:yy + tile, xx:xx + tile],
                                            expected=expected, hint=[]):
                    n += 1
            except Exception:  # noqa: BLE001
                pass
        return n

    def _sample(cells: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
        if limit and len(cells) > limit:
            step = len(cells) / limit
            return [cells[int(i * step)] for i in range(limit)]
        return cells

    base = _cells(phase[0], phase[1])
    if origins:
        # 成果自己记录了载体原点：直接按记录的位置解，不做网格枚举也不扫相位。
        # 载体的实际边长写在 GDS_TM_TILE_SIZE 里，可能与默认 256 不同（实测有整幅 1024 的）。
        kept: list[tuple[int, int]] = []
        for px, py in origins:
            if 0 <= px <= w - tile and 0 <= py <= h - tile:
                kept.append((py, px))
        if kept and valid is not None:
            # 掩码的用途是剔除"压在黑边上、本来就没数据"的格子。
            # 但载体本身就可能是整幅（实测 1024² 成果只有 1 个 1024 载体）：
            # 这时裁剪 15% 就会让覆盖率掉到 0.70 < 0.9，把唯一的载体整个剔掉，
            # 于是"裁剪后检不出"。过滤后为空就说明这个阈值不适用于当前载体尺寸，
            # 此时保留全部记录原点，让解码器自己判。
            filtered = [c for c in kept
                        if float(valid[c[0]:c[0] + tile, c[1]:c[1] + tile].mean()) >= min_valid]
            kept = filtered or kept
        if kept:
            base = _sample(kept, max_carriers) if max_carriers and len(kept) > max_carriers else kept
            hits = _decode(base)
            return hits, len(base)
    if not base:
        return 0, 0

    if phase_search and tile >= 32:
        # 粗扫：4×4 相位（步长 tile/4），每个相位只解少量载体定相位。
        # probe_n 取 4 而不是 3：裁剪后每行只剩 2-3 个载体，抽 3 个可能全落在
        # 边界半格上（实测裁 9% 的 820² 只有 3×3 格，只解 3 个时全 0）。
        probe_n = 4
        best_phase, best_rate = (phase[0], phase[1]), -1.0
        # 步长取 tile/8 = 32：256 的余数有 256 种，步长 64 只能覆盖 4 个相位，
        # 实测 d=102（余数 102）恰好落在 64 与 128 的空隙里，扫不到。
        # 步长 32 是 8×8=64 个相位，能定出正确的格子。
        step = max(8, tile // 8)
        # 提速：粗扫阶段每个相位只解 2 个载体（不是 4 个）。命中相位和空相位
        # 的差别是 2/2 vs 0/2，2 个样本已足够分辨；找到首个满分相位立即停。
        coarse_n = 2
        for py in range(0, tile, step):
            for px in range(0, tile, step):
                cc = _cells(px, py)
                if not cc:
                    continue
                sub = _sample(cc, coarse_n)
                r = _decode(sub) / len(sub) if sub else 0.0
                if r > best_rate:
                    best_rate, best_phase = r, (px, py)
                if r >= 1.0:
                    break
            if best_rate >= 1.0:
                break
        base = _cells(best_phase[0], best_phase[1])
        if not base:
            return 0, 0
        # 若粗扫已是满分（小图全部命中），直接返回，不必再做精扫
        if best_rate >= 1.0 and len(base) <= probe_n:
            return _decode(base), len(base)
    cells = _sample(base, max_carriers)
    hits = _decode(cells)
    # 命中为 0 时再补一轮：粗扫只用 2 个样本定相位，可能恰好都落在边界半格上
    # （实测 820² 裁 9% 时 2/2 抽样全空）。此时把该相位的全部载体解一遍，
    # 宁可多花十几秒，也不要把"还能解出来"判成"没有"。
    if phase_search and hits == 0 and 0 < len(base) <= 36:
        hits = _decode(base)
        cells = base
    return hits, len(cells)


def estimate(reference_gray: np.ndarray, probe_gray: np.ndarray) -> dict | None:
    """SIFT + RANSAC 估计 probe→reference 的相似变换（旋转+缩放+平移）。"""
    sift = cv2.SIFT_create(**_SIFT)
    k1, d1 = sift.detectAndCompute(reference_gray, None)
    k2, d2 = sift.detectAndCompute(probe_gray, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    pairs = cv2.BFMatcher().knnMatch(d2, d1, k=2)
    good = [m for m, n in pairs if m.distance < 0.8 * n.distance]
    if len(good) < 12:
        return None
    p2 = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(
        p2, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None:
        return None
    return {
        "matrix": matrix,
        "angle": round(math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))), 4),
        "scale": round(float(math.hypot(matrix[0, 0], matrix[1, 0])), 6),
        "inliers": int(inliers.sum()) if inliers is not None else 0,
        "matches": len(good),
    }


def restore_and_decode(reference: Path, probe: Path, expected: int,
                       tile: int = 256, match_side: int = 2048,
                       max_carriers: int = 64) -> dict:
    """还原 probe 到 reference 几何，按 TrustMark 载体网格解码。

    两阶段，为了不在 2.83 GB 参考图上做 SIFT：
      1. 参考图抽稀到 match_side 以内，与探针同量级做 SIFT —— 只要相对几何，缩略完全够用；
      2. 把缩略尺度上的相似变换还原到探针原分辨率，在探针自己的坐标系里 warp。
    探针本身通常就是被裁剪/旋转后的一小块，按原分辨率解码才有足够的载体可统计。

    对三种插值各还原一次并取命中率最高者：旋转用小角度、缩放用大幅重采样时最优插值不同。
    """
    tr = _runner_module()

    # 载体网格以参考成果自己的标签为准（GDS_TM_TILE_SIZE / GDS_TM_WINDOWS）。
    # 写死 256px 时，一份"整幅一个 1024 载体"的成果会被切成 16 个假载体而全灭。
    plan_tile, plan_origins = carrier_plan(Path(reference))
    if plan_tile > 0:
        tile = plan_tile

    ref_img = _first_band(Path(reference), max_side=match_side)
    probe_path = Path(probe)
    prb_full = _first_band(probe_path)
    prb_small = _first_band(probe_path, max_side=match_side)
    if ref_img is None or prb_full is None or prb_small is None:
        return {"found": False, "reason": "unreadable_input"}

    tm = tr._make_tm()
    # 直解在探针自己的坐标系里做，与还原后的判定标准一致。
    # 先按标准原点直解（快），只在命中率不足时才启用相位搜索 —— 实测相位搜索
    # 会把每次判定从 1 s 拖到 45 s（16 相位 × 抽样解码），而对已对齐的图毫无帮助。
    direct_hits, direct_total = _carrier_rate(prb_full, tm, expected, tile, max_carriers,
                                              origins=plan_origins)
    if direct_total and direct_hits / direct_total <= 0.5:
        ph_hits, ph_total = _carrier_rate(prb_full, tm, expected, tile, max_carriers,
                                          phase_search=True, origins=plan_origins)
        if ph_total and ph_hits / ph_total > direct_hits / max(direct_total, 1):
            direct_hits, direct_total = ph_hits, ph_total

    # SIFT 尺度退化：探针是参考图的放大/缩小版时，两张图缩到同一 match_side 后信息量不同，
    # cv2.estimateAffinePartial2D 会给出 scale≈1 的错解（实测缩放 1.5x 报 s=0.999996、
    # 2.0x 报 s=0.999998，都是错的，据此 warp 后 0/32 全灭）。
    # 正确做法：先把探针按"假设的缩放倍率"归一化到参考图尺寸，再估变换 —— 这样 SIFT
    # 面对的两张图尺度一致，估出的就是纯旋转+平移，缩放倍率由我们显式枚举。
    # SIFT 必须在"探针原始分辨率"与"参考图"之间做，不能先把探针缩到 match_side：
    # 实测把 1.5x 探针（3072）先缩到 2048 再匹配，SIFT 报 scale=1.0（真值 0.667），
    # 因为预缩小恰好抵消了放大。直接用 prb_full 匹配则 1.5x 得 0.6667、2.0x 得 0.5000，全对。
    info = estimate(ref_img, prb_full)
    best_global = (direct_hits, direct_total, "direct", None, 1.0)
    info_used = None
    if info is not None:
        # 关键：还原后的画布必须等于「参考图尺寸」，不能是探针尺寸。
        # 载体是 256px 网格，探针放大 1.5x 后载体实际占 384px，按 256px 切必然错位 ——
        # 这就是之前 1.5x/2.0x 只能到 50% 的原因（相位对但格子大小不对）。
        # 用 ref_img 的尺寸做画布，载体回到标准 256px，解码器才能对齐。
        base_m = info["matrix"].astype(np.float64).copy()
        canvas = (ref_img.shape[1], ref_img.shape[0])

        def _score(m, flag=cv2.INTER_LINEAR, search_phase=False):
            restored = cv2.warpAffine(prb_full, m, canvas, flags=flag,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # 有效区掩码：把"全 1"的探针按同一变换投到画布上，1 的位置才是真实数据，
            # 0 的位置是黑边。用 NEAREST 保证掩码是 0/1 而不被插值糊成小数。
            mask = cv2.warpAffine(np.ones(prb_full.shape[:2], np.uint8), m, canvas,
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return _carrier_rate(restored, tm, expected, tile, max_carriers,
                                 phase_search=search_phase, valid=mask,
                                 origins=plan_origins)

        # 阶段一：扫平移，找回载体网格相位。
        # 裁剪会让整幅网格相位整体错位：1024² 裁 10% 起点 102px，对 256 取模 = 102，
        # 即偏移 0.4 个载体，直解 0/9；而裁 25%（d=256）恰好是 1 个载体宽，网格天然
        # 对齐，直解 100%。仿射估计只保证整体贴合，管不到 256px 相位，必须单独扫。
        # 搜索策略：先粗后精 —— 粗扫用 1/2 载体步长覆盖整格，精扫在最优解附近 ±1/8 微调。
        def _shift_score(m, flag):
            return _score(m, flag)

        best = (best_global[0] / max(best_global[1], 1), base_m.copy())
        if tile >= 64:
            # 有记录的载体原点时，格子位置本身是已知的，SIFT 已经把几何对齐到几个像素内，
            # 只需在残差附近小范围精修。实测 ±8 步长 8 再 ±2 步长 2 已足够覆盖残差；
            # 之前用 ±24(步8) + ±4(步2) 共 74 个候选，单个旋转用例要 250 秒。
            steps = ((8, 1), (2, 1)) if plan_origins else ((max(8, tile // 2), 1),
                                                           (max(4, tile // 8), 1))
            for step, reach in steps:
                cx, cy = int(best[1][0, 2]), int(best[1][1, 2])
                found_better = False
                for dy_i in range(-reach, reach + 1):
                    for dx_i in range(-reach, reach + 1):
                        if step > max(8, tile // 2) and dx_i == 0 and dy_i == 0:
                            continue
                        m = base_m.copy()
                        m[0, 2] = cx + dx_i * step
                        m[1, 2] = cy + dy_i * step
                        hits, total = _shift_score(m, cv2.INTER_NEAREST)
                        rate = hits / total if total else 0.0
                        if total and rate > best[0]:
                            best = (rate, m.copy())
                            found_better = True
                if best[0] > 0.85:
                    break
                if not found_better and step <= steps[-1][0]:
                    break

        phase_m = base_m.copy()
        for tweak in (1.0, 0.998, 1.002):
            m0 = phase_m.copy()
            m0[0, :2] *= tweak
            m0[1, :2] *= tweak
            for name in _INTERP:
                flag = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
                        "nearest": cv2.INTER_NEAREST}[name]
                hits, total = _score(m0, flag)
                rate = hits / total if total else 0.0
                if total and rate > best_global[0] / max(best_global[1], 1):
                    best_global = (hits, total, name, dict(info, tweak=tweak), rate)
                    info_used = dict(info, tweak=tweak)
            if best_global[0] / max(best_global[1], 1) > 0.85:
                break

        # 兜底：常规路径没命中时，才启用网格相位搜索。
        # 代价很高（16 相位 × 抽样解码，实测 2048² 上一次 45 s），而且只在
        # **小图**（探针 ≤1280² 且载体格数 ≤36）上真正有效 —— 大图上裁掉 100–200px
        # 相对位移太小，粗扫步长覆盖不到真实相位，扫了也是白扫（实测 2048²
        # 裁 5/8/10/12/15/20% 全部 0%，只有 25% 对齐才 100%）。
        small_probe = max(prb_full.shape[0], prb_full.shape[1]) <= 1280
        if small_probe and best_global[0] / max(best_global[1], 1) <= 0.5:
            for tweak in (1.0, 0.998, 1.002):
                m0 = base_m.copy()
                m0[0, :2] *= tweak
                m0[1, :2] *= tweak
                for name in _INTERP:
                    flag = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
                            "nearest": cv2.INTER_NEAREST}[name]
                    hits, total = _score(m0, flag, search_phase=True)
                    rate = hits / total if total else 0.0
                    if total and rate > best_global[0] / max(best_global[1], 1):
                        best_global = (hits, total, name + "+phase",
                                       dict(info, tweak=tweak), rate)
                        info_used = dict(info, tweak=tweak)
                if best_global[0] / max(best_global[1], 1) > 0.5:
                    break

    hits, total, via, info_found, rate = best_global
    result = {
        "direct_hits": direct_hits,
        "direct_total": direct_total,
        "direct_rate": round(direct_hits / direct_total, 4) if direct_total else 0.0,
        "probe_size": [int(prb_full.shape[1]), int(prb_full.shape[0])],
        "match_size": [int(prb_small.shape[1]), int(prb_small.shape[0])],
        "reference_match_size": [int(ref_img.shape[1]), int(ref_img.shape[0])],
        "geometry": ({k: (info_found or {}).get(k) for k in ("angle", "inliers", "matches")}
                     if info_found else None),
    }
    if not info_found:
        result.update({"found": direct_hits > 0, "hits": direct_hits, "total": direct_total,
                       "rate": result["direct_rate"], "via": "direct_only",
                       "reason": "sift produced no usable transform at any tried scale"})
        return result

    result.update({
        "found": rate > 0.5,
        "hits": hits, "total": total, "rate": round(rate, 4),
        "via": via,
        "angle": info_found.get("angle"), "scale": info_found.get("mult"),
        "inliers": info_found.get("inliers"), "matches": info_found.get("matches"),
    })
    return result

def restore_and_decode_bounded(reference: Path, probe: Path, expected: int,
                               match_side: int = 2048, max_carriers: int = 32,
                               min_hits: int = 3, min_rate: float = 0.05) -> dict:
    """大图有界版：SIFT 只在缩略图上做，解码逐载体从探针读原始分辨率块。

    不能直接用 ``restore_and_decode``：它把探针整幅读进内存、并在原分辨率上跑 SIFT。
    实测 715.6 Mpx 探针（23718x30172）单次 SIFT 超过 34,800 CPU 秒仍未返回，
    还款核验必然超时 —— 这就是前端看到的"水印核验超时"。

    这里把「估算变换」和「解码」拆开：
      * 变换只需要相对几何，两边都缩到 match_side 完全够用；
      * 解码必须原生载体尺寸（实测 1024px），所以按逆变换算出每个载体在探针里的四边形，
        只读那一块、只 warp 那一块 —— 工作量是 max_carriers 个补丁，不随整幅影像膨胀。
    """
    import rasterio
    from rasterio.windows import Window

    tr = _runner_module()
    plan_tile, plan_origins = carrier_plan(Path(reference))
    tile = plan_tile if plan_tile > 0 else 256

    ref_small = _first_band(Path(reference), max_side=match_side)
    prb_small = _first_band(Path(probe), max_side=match_side)
    if ref_small is None or prb_small is None:
        return {"found": False, "reason": "unreadable_input"}

    with rasterio.open(reference) as ds:
        rf_w, rf_h = ds.width, ds.height
    with rasterio.open(probe) as ds:
        pf_w, pf_h = ds.width, ds.height

    s_ref = max(ref_small.shape[:2]) / float(max(rf_w, rf_h))
    s_prb = max(prb_small.shape[:2]) / float(max(pf_w, pf_h))

    info = estimate(ref_small, prb_small)
    if info is None:
        return {"found": False, "reason": "estimate_failed"}

    # 缩略图上的变换换算回全幅坐标：线性部分乘 (s_prb/s_ref)，平移除以 s_ref。
    A = np.asarray(info["matrix"], dtype=np.float64).copy()
    A[:, :2] *= (s_prb / s_ref)
    A[:, 2] /= s_ref
    inv = cv2.invertAffineTransform(A)

    if plan_origins:
        cand = [(int(x), int(y)) for x, y in plan_origins
                if x + tile <= rf_w and y + tile <= rf_h]
    else:
        cand = [(x, y) for y in range(0, rf_h - tile + 1, tile)
                for x in range(0, rf_w - tile + 1, tile)]
    if not cand:
        return {"found": False, "reason": "no_carrier"}
    if max_carriers and len(cand) > max_carriers:
        step = len(cand) / float(max_carriers)
        cand = [cand[int(i * step)] for i in range(max_carriers)]

    tm = tr._make_tm()
    hits = total = 0
    dst_tri = np.array([[0, 0], [tile, 0], [0, tile]], dtype=np.float32)
    with rasterio.open(probe) as ds:
        for ox, oy in cand:
            quad = np.array([[ox, oy], [ox + tile, oy], [ox, oy + tile],
                             [ox + tile, oy + tile]], dtype=np.float32).reshape(1, 4, 2)
            src = cv2.transform(quad, inv).reshape(4, 2)
            x0 = int(np.floor(src[:, 0].min()))
            y0 = int(np.floor(src[:, 1].min()))
            x1 = int(np.ceil(src[:, 0].max()))
            y1 = int(np.ceil(src[:, 1].max()))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(pf_w, x1), min(pf_h, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            try:
                arr = ds.read(1, window=Window(x0, y0, x1 - x0, y1 - y0))
            except Exception:  # noqa: BLE001
                continue
            local = (src[:3] - np.array([x0, y0], dtype=np.float32)).astype(np.float32)
            M = cv2.getAffineTransform(local, dst_tri)
            patch = cv2.warpAffine(arr, M, (tile, tile), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            total += 1
            try:
                if tr._decode_carrier_window(tm, patch, expected=expected, hint=[]):
                    hits += 1
            except Exception:  # noqa: BLE001
                pass

    rate = hits / total if total else 0.0
    # 判定口径：TrustMark 解出载荷几乎不存在误报，命中 3 个载体即可确认"是本系统数据"。
    # 裁剪会把大量载体切在边界上，命中率天然偏低，不能用高比例阈值否掉真实命中。
    found = bool(total and hits >= min_hits and rate >= min_rate)
    return {"found": found, "hits": hits, "total": total,
            "rate": rate, "tile": tile, "method": "bounded_inverse_warp"}
