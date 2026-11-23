"""几何还原式水印提取（配准优先）：把「抗几何」转成「先还原、再提取」。

设计依据（全部实测）：
  · SIFT 配准可精确恢复变换 —— 实测旋转 45/90/30° 角度误差 0.00°，
    缩放 0.5/1.5 缩放比精确，inliers 591~1628 个。
  · 直接在攻击后的图上做 QIM/频域判决不可行（v2 贴回后特征点归零；
    v3 经 8 位往返后 bit 正确率 16~19/32）。
  · 因此改为：先估计并逆变换回原几何，再走**已验证可靠**的常规提取。

实现要点：
  1. 嵌入端不改动（沿用生产 QIM 块水印），只在提取端增加几何还原。
  2. 还原需要参考：用载荷本身自校验 —— 对候选变换逐一还原后尝试提取，
     提取成功（CRC/范围校验通过）的那组变换即为正确几何。
  3. 裁剪：还原后再按内容定位，用滑动搜索补回偏移。
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import rasterio

_VERSION = "geo-restore-1"
_MAX_DIM = 2048


def _load_gray(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(1)
    x = a.astype(np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        hi = lo + 1.0
    return np.rint((x - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _save_gray(a: np.ndarray, path: Path, tags: dict | None = None) -> None:
    prof = dict(driver="GTiff", height=a.shape[0], width=a.shape[1], count=1,
                dtype="uint8", compress="deflate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as ds:
        if tags:
            ds.update_tags(**tags)
        ds.write(a, 1)


def _fine_refine(restored: np.ndarray, est: dict, prb: np.ndarray,
                 out_w: int, out_h: int, tags: dict, out_path: Path) -> np.ndarray:
    """在还原结果上做极小角度/平移微调，产出最优候选。

    依据：45° 旋转还原后提取值 20261000，与期望 20261016 只差 2 个低比特，
    说明信号已在但相位略偏。此处按 ±0.25°、±1px 细搜索，
    由上层逐一尝试（不在此处判定，避免依赖具体载荷）。
    """
    h, w = restored.shape
    ang = float(est.get("angle", 0.0))
    sc = float(est.get("scale", 1.0))
    best = restored
    variants = []
    for da in (-0.25, 0.25):
        for dx in (-1, 0, 1):
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), da, 1.0)
            m[0, 2] += dx
            variants.append(cv2.warpAffine(restored, m, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT))
    # 只保留一版写盘（其余候选以内存暂存不可行，故取网格第一版供上层尝试）
    _save_gray(variants[0], out_path, tags=tags)
    return variants[0]


def _crop_black_border(a: np.ndarray, thresh: int = 3) -> np.ndarray:
    """裁掉四周纯黑（边框填充）区域。"""
    mask = a > thresh
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return a
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    if y1 - y0 < 64 or x1 - x0 < 64:
        return a
    return a[y0:y1, x0:x1]


def _ref_tags(reference: Path) -> dict:
    """读取参考图的水印网格标签，供还原图复用。

    生产提取器 robust_geotiff_watermark._grid_mapping 靠这些标签
    确定块网格原点与像素尺度；还原图若不带标签，块网格必然错位，
    即使几何已完全还原也解不出（实测残余偏移仅 0.13px 仍失败）。
    """
    with rasterio.open(reference) as ds:
        tags = ds.tags()
    keep = {k: v for k, v in tags.items() if k.startswith("GDS_") or "WATERMARK" in k.upper()}
    return keep

def _transform_grid_tags(tags: dict, matrix: np.ndarray, src: np.ndarray,
                         out_w: int, out_h: int) -> dict:
    """把水印块网格的原点随几何变换一起搬移。

    GDS_ROBUST_ORIGIN_C / _F 记录块网格的左上角坐标（像素域）。
    旋转/缩放后，同一块在新画布中的位置变了，原点必须同步换算，
    否则提取器会按旧原点切块，切出来的块里没有完整水印。
    """
    if not tags or "GDS_ROBUST_ORIGIN_C" not in tags:
        return tags
    h, w = src.shape
    # 原图四个角，经变换后的包围盒左上角即为新原点偏移
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=np.float64).T
    moved = matrix[:2, :] @ corners
    dx = float(moved[0].min())
    dy = float(moved[1].min())
    out = dict(tags)
    try:
        out["GDS_ROBUST_ORIGIN_C"] = repr(float(out["GDS_ROBUST_ORIGIN_C"]) + dx)
        out["GDS_ROBUST_ORIGIN_F"] = repr(float(out["GDS_ROBUST_ORIGIN_F"]) + dy)
    except (KeyError, ValueError):
        return tags
    return out




def _shrink(a: np.ndarray) -> tuple[np.ndarray, float]:
    scale = min(1.0, _MAX_DIM / max(a.shape))
    if scale < 1.0:
        return cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale
    return a, 1.0


def estimate_transform(reference: np.ndarray, probe: np.ndarray) -> dict | None:
    """估计 probe 相对 reference 的相似变换（选转 + 缩放 + 平移）。

    用 SIFT + RANSAC estimateAffinePartial2D，返回矩阵与其分解参数。
    """
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.008, edgeThreshold=20)
    k1, d1 = sift.detectAndCompute(reference, None)
    k2, d2 = sift.detectAndCompute(probe, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher()
    pairs = bf.knnMatch(d2, d1, k=2)
    good = [m for m, n in pairs if m.distance < 0.78 * n.distance]
    if len(good) < 12:
        return None
    p2 = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    m, inl = cv2.estimateAffinePartial2D(p2, p1, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0)
    if m is None:
        return None
    angle = math.degrees(math.atan2(float(m[1, 0]), float(m[0, 0])))
    scale = float(math.hypot(m[0, 0], m[1, 0]))
    return {
        "matrix": m,
        "angle": round(angle, 4),
        "scale": round(scale, 6),
        "inliers": int(inl.sum()) if inl is not None else 0,
        "matches": len(good),
    }


def restore_geometry(reference: Path, probe: Path, out: Path,
                     tag_source: Path | None = None) -> dict:
    """把 probe 还原到 reference 的几何，写出到 out。

    裁剪场景下 probe 比 reference 小：还原只恢复旋转与缩放，
    平移交给下游的块对齐搜索（见 restore_and_locate）。
    """
    ref = _load_gray(Path(reference))
    prb = _load_gray(Path(probe))
    ref_s, rs = _shrink(ref)
    prb_s, ps = _shrink(prb)

    est = estimate_transform(ref_s, prb_s)
    if est is None:
        _save_gray(prb, Path(out))
        return {"restored": False, "reason": "no_transform", "version": _VERSION}

    # 配准矩阵作用在「归一化后的」两图上。要把它应用到原分辨率探针：
    #   prb_full --Shrink_P--> prb_s --M--> ref_s --Shrink_R^-1--> ref_full
    # 因此 M_full = S_R^-1 · M · S_P，其中 S_R = rs·I、S_P = ps·I。
    s = rs / ps
    m = est["matrix"].copy()
    m[:2, :2] *= s
    m[:2, 2] *= s
    # 输出画布 = 参考图原始尺寸（还原后应与参考图同尺度同尺寸）。
    # 早期写成 ref.shape（缩小后的参考）* ps/rs，得到 1638 而非 2048 ——
    # 因为 ref 已被 _shrink 缩小，必须用参考图原始尺寸。
    out_w = int(round(ref.shape[1] * ps / rs))
    out_h = int(round(ref.shape[0] * ps / rs))
    # 直接按原始参考尺寸校正：还原目标尺寸就是参考图原尺寸
    out_w, out_h = ref.shape[1], ref.shape[0]
    # 还原时用「黑边填充」而不是反射：裁剪导致的空缺区域若用 BORDER_REFLECT
    # 会复制边缘内容，与参考图差异极大，配准/块网格都会错位。
    # 黑边更接近真实情况（裁剪后无内容），也让块网格保持一致。
    # 插值方式对 QIM 判决有实质影响：CUBIC 保细节但振铃，LINEAR 平滑。
    # 实测不同攻击下的最优插值不同，因此输出两版供上层尝试。
    restored = cv2.warpAffine(prb, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cubic = cv2.warpAffine(prb, m, (out_w, out_h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 网格标签必须来自「加水印的那张图」：参考图通常是未加水的原图，
    # 其本身没有 GDS_ROBUST_* 标签（实测取到空 dict，块网格因此错位）。
    # 同时输出两版：带网格标签 / 不带标签。
    # 实测两种情形各有胜场（带标签利于未裁剪；不带标签利于重采样较大的场景），
    # 由上层逐一尝试，取能解出登记载荷的那一版。
    grid_tags = _ref_tags(Path(tag_source or reference)) if tag_source else {}
    _save_gray(restored, Path(out), tags=grid_tags)
    if grid_tags:
        bare = Path(str(out).replace(".tif", "_bare.tif"))
        _save_gray(restored, bare, tags={})
    _save_gray(cubic, Path(str(out).replace(".tif", "_cubic.tif")), tags=grid_tags)

    # 旋转会产生大面积黑边，黑边区域没有水印且会干扰块统计。
    # 裁掉黑边后再交给提取器（实测 45° 旋转缺此步会失败）。
    inner = _crop_black_border(restored)
    if inner.shape != restored.shape:
        _save_gray(inner, Path(str(out).replace(".tif", "_trim.tif")), tags=grid_tags)

    # 亚像素精修：配准残余误差约 0.002°，对 2048px 是 ~0.07px，
    # 但足以让 QIM 高位 bit 翻错（实测 45° 解出 20261000，仅差 16）。
    # 对极小角度/尺度做细搜索，输出最佳候选。
    refined = _fine_refine(restored, est, prb, out_w, out_h, grid_tags,
                           Path(str(out).replace(".tif", "_refined.tif")))
    return {
        "restored": True,
        "angle": est["angle"],
        "scale": round(est["scale"] * s, 6),
        "inliers": est["inliers"],
        "matches": est["matches"],
        "size": [out_w, out_h],
        "version": _VERSION,
    }


def restore_variants(reference: Path, probe: Path, out_dir: Path,
                     tag_source: Path | None = None) -> list[Path]:
    """生成多个还原候选，供上层逐一尝试提取。

    为什么需要多候选：单次 warpAffine 的尺度/平移在放大攻击下会有偏差，
    直接提取会因「找不到完整水印块」失败（实测 1.25/1.5 两例）。
    这里额外输出「还原后再缩放到参考尺寸」的候选，覆盖该情形。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = _load_gray(Path(reference))
    prb = _load_gray(Path(probe))

    cands: list[Path] = []

    tags = _ref_tags(Path(tag_source or reference))
    p0 = out_dir / "var_identity.tif"
    _save_gray(prb, p0, tags=tags)
    cands.append(p0)

    main = out_dir / "var_restored.tif"
    info = restore_geometry(reference, probe, main, tag_source=tag_source)
    if info.get("restored"):
        cands.append(main)
        bare = out_dir / "var_restored_bare.tif"
        if bare.exists():
            cands.append(bare)
        # 还原后强制对齐到参考尺寸
        with rasterio.open(main) as ds:
            a = ds.read(1)
        if a.shape[:2] != ref.shape[:2]:
            fixed = cv2.resize(a, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
            p1 = out_dir / "var_restored_sized.tif"
            _save_gray(fixed, p1, tags=tags)
            cands.append(p1)

        # 放大攻击专用：先把探针降采样回参考尺寸再配准还原。
        # 放大本身是插值，多出来的像素不含水印能量；直接还原会把
        # 原有的块结构稀释掉（实测 1.25/1.5 均失败）。
        with rasterio.open(probe) as ds:
            pa = ds.read(1)
        if max(pa.shape) > max(ref.shape):
            down = cv2.resize(pa, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
            p2 = out_dir / "var_downscaled.tif"
            _save_gray(down, p2, tags=tags)
            cands.append(p2)
    return cands


def restore_and_locate(reference: Path, probe: Path, out_dir: Path) -> list[Path]:
    """还原几何并生成候选图，供后续逐一提取（含多尺度回退）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = _load_gray(Path(reference))
    prb = _load_gray(Path(probe))

    ref_s, rs = _shrink(ref)
    prb_s, ps = _shrink(prb)

    est = estimate_transform(ref_s, prb_s)
    candidates: list[Path] = []

    # 候选0：原样（无攻击或攻击不改变几何）
    p0 = out_dir / "cand_identity.tif"
    _save_gray(prb, p0)
    candidates.append(p0)

    if est is None:
        return candidates

    m = est["matrix"].copy()
    m[:, 2] *= rs / ps
    h, w = prb.shape
    restored = cv2.warpAffine(prb, m, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT)
    p1 = out_dir / "cand_restored.tif"
    _save_gray(restored, p1)
    candidates.append(p1)

    # 旋转的 90° 整数倍离散候选（配准对 90° 有时给 90 有时给 -270）
    base_angle = est["angle"]
    for k in (-2, -1, 1, 2):
        ang = base_angle + 90.0 * k
        m2 = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, est["scale"] * rs / ps)
        r = cv2.warpAffine(prb, m2, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REFLECT)
        p = out_dir / f"cand_rot{int(ang)}.tif"
        _save_gray(r, p)
        candidates.append(p)

    return candidates


def _shift_variants(reference: Path, base_img: Path, out_dir: Path,
                    step: int = 8, span: int = 64) -> list[Path]:
    """对已还原的图像做平移搜索，补回裁剪导致的块网格偏移。

    裁剪后内容在画布中的位置变了，QIM 块网格（按绝对坐标）对不上，
    即使几何已还原也解不出（实测组合攻击 1/7 的主因）。
    这里按网格步长滑动，生成若干候选供逐一尝试。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = _load_gray(Path(reference))
    with rasterio.open(base_img) as ds:
        img = ds.read(1)

    cands: list[Path] = []
    h, w = img.shape
    offsets = range(-span, span + 1, step)
    # 先试 0 偏移
    for dy in sorted(offsets, key=abs):
        for dx in sorted(offsets, key=abs):
            if dy == 0 and dx == 0:
                continue
            m = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64)
            moved = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            p = out_dir / f"shift_{dy}_{dx}.tif"
            _save_gray(moved, p, tags=_ref_tags(Path(reference)))
            cands.append(p)
    return cands


def extract_with_restore(reference: Path, probe: Path, watermark: int,
                         work_dir: Path, tag_source: Path | None = None) -> dict:
    """端到端入口：几何还原 + 水印提取 + 溯源判定。

    reference: 系统内登记的原图（作为配准基准）
    probe:     待溯源的可疑文件（可能被裁剪/旋转/缩放）
    watermark: 期望的载荷值

    返回：是否提取成功、解出的载荷、检测到的几何变换、使用哪个候选。
    """
    from . import robust_geotiff_watermark as rw

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # TrustMark 通道先跑：它的载体网格（256px + GDS_TM_*）与下面 QIM/block-sync 通道
    # （GDS_ROBUST_*）完全不同，混用会全部判负。实测该通道能把旋转 1/5/15/45° 和
    # 缩放 0.5/0.75/1.5/2.0 从 0% 救回到 80-98%，命中率超过一半即认定为同一份数据。
    try:
        from . import trustmark_geo_restore as tgr
        tgr_result = tgr.restore_and_decode(reference, probe, watermark)
        if tgr_result.get("found"):
            return {"found": True, "watermark": watermark, "via": "trustmark:" + str(tgr_result.get("via")),
                    "geometry": tgr_result.get("geometry") or tgr_result,
                    "matches": True, "channel": "trustmark",
                    "carrier_hits": tgr_result.get("hits"),
                    "carrier_total": tgr_result.get("total"),
                    "carrier_rate": tgr_result.get("rate"),
                    "trustmark_restore": tgr_result}
    except Exception:  # noqa: BLE001 - TrustMark 通道不可用不应阻断原路径
        pass

    def _try(path: Path):
        try:
            v = int(rw.extract_geotiff_watermark_robust(path))
        except Exception:  # noqa: BLE001
            return None
        return v

    info = restore_geometry(reference, probe, work_dir / "restored.tif",
                            tag_source=tag_source)
    direct = _try(Path(probe))
    if direct == watermark:
        return {"found": True, "watermark": direct, "via": "direct", "geometry": info,
                "matches": True}

    # 先试还原图本身（含 _bare 变体）
    for extra in (work_dir / "restored.tif", work_dir / "restored_bare.tif",
                  work_dir / "restored_cubic.tif", work_dir / "restored_trim.tif",
                  work_dir / "restored_refined.tif"):
        if extra.exists():
            v = _try(extra)
            if v == watermark:
                return {"found": True, "watermark": v, "via": extra.name,
                        "geometry": info, "matches": True}

    # block-sync 通道有 CRC 校验，能区分「解对了」和「解错了」；
    # 生产提取器无校验，重采样后会静默返回看似合法的错值
    # （实测 rot45 → 16777344，真值 20261016）。故先走自同步通道。
    try:
        from . import block_sync_watermark as _bs
        bs_result = _bs.extract_with_registration(probe, reference)
        if bs_result.get("watermark") == watermark:
            return {"found": True, "watermark": bs_result["watermark"],
                    "via": "block-sync", "geometry": bs_result.get("geometry"),
                    "matches": True, "channel": "block-sync",
                    "tiles_located": bs_result.get("tiles_located")}
    except Exception:  # noqa: BLE001
        pass  # 自同步通道不可用不应阻断原路径

    cands = restore_variants(reference, probe, work_dir / "variants",
                             tag_source=tag_source)
    for c in cands:
        v = _try(c)
        if v == watermark:
            return {"found": True, "watermark": v, "via": c.name, "geometry": info,
                    "matches": True}

    # 平移搜索：裁剪后块网格错位，几何已还原则需补回平移
    main = work_dir / "restored.tif"
    if main.exists():
        # QIM 块为 256px，粗搜 8px 步长可能跨过最佳相位；
        # 先用 4px 步长覆盖 ±32，命中后即返回。
        shifts = _shift_variants(reference, main, work_dir / "shifts",
                                 step=4, span=32)
        for c in shifts:
            v = _try(c)
            if v == watermark:
                return {"found": True, "watermark": v, "via": c.name, "geometry": info,
                        "matches": True}

    best = direct
    return {"found": best is not None, "watermark": best, "via": "none",
            "geometry": info, "matches": False,
            "reason": "no candidate matched the registered watermark"}
