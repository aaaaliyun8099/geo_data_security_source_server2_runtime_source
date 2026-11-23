"""DEM 抗几何还原：SIFT 配准 + 整数值 LSB 提取。

与 ``trustmark_geo_restore`` 的关键差异（都是实测得出的硬约束）：

1. **DEM 水印位在整数值末位**，任何重采样都会把这些位抹掉。
   所以只有「旋转」和「放大」能还原 —— 放大是插值复制，末位有概率保住；
   缩小是面积平均，末位必然被抹平。实测：
     旋转 5/90/180°  还原后命中
     缩放 1.5x/3.0x  还原后命中
     缩放 0.5x/0.75x 还原后仍失败（不是策略问题，是信息已丢失）

2. **还原画布必须等于参考图尺寸**，且用 ``INTER_NEAREST``。
   用 ``INTER_LINEAR``/``INTER_CUBIC`` 会做插值，末位被算掉 → 全失败（实测三种插值全 0）。

3. DEM 是连续高程面，SIFT 特征丰富（1024² 块能出 4000+ 关键点），配准本身不难。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_NODATA_PROBE = (-32768, -9999, -999999)


def _is_nodata(a: np.ndarray, nodata) -> np.ndarray:
    if nodata is None:
        return np.zeros(a.shape, bool)
    return a == nodata


def _to_gray(a: np.ndarray, nodata=None) -> np.ndarray:
    """把高程面映射成 uint8，供 SIFT 使用。NaN/NoData 置 0。"""
    import numpy as np
    m = ~_is_nodata(a, nodata) & np.isfinite(a)
    if not m.any():
        return np.zeros(a.shape, np.uint8)
    vals = a[m].astype(np.float64)
    lo, hi = np.percentile(vals, [1, 99])
    if hi - lo < 1e-9:
        return np.zeros(a.shape, np.uint8)
    g = np.clip((a.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0) * 255.0
    g[~m] = 0.0
    return g.astype(np.uint8)


def _read_band(path: Path, max_side: int | None = None) -> tuple[np.ndarray, dict]:
    """读第 1 波段。max_side 用于抽样（只影响配准估计，不影响末位判定）。"""
    import rasterio
    from rasterio.enums import Resampling
    with rasterio.open(path) as ds:
        prof = {"width": ds.width, "height": ds.height, "dtype": ds.dtypes[0],
                "nodata": ds.nodata, "transform": ds.transform, "crs": ds.crs,
                "profile": ds.profile.copy(), "tags": ds.tags()}
        if max_side and max(ds.width, ds.height) > max_side:
            ratio = max_side / max(ds.width, ds.height)
            h = max(1, int(ds.height * ratio))
            w = max(1, int(ds.width * ratio))
            arr = ds.read(1, out_shape=(h, w), resampling=Resampling.average)
        else:
            arr = ds.read(1)
    return arr, prof


def _estimate(reference: Path, probe: Path, max_side: int = 2048) -> dict | None:
    """SIFT 估计 probe -> reference 的相似变换。

    注意：两边都必须用**原始分辨率**（或同一 max_side 抽样）做匹配，
    不能先把小尺寸的一边钉到大尺寸，否则尺度信息丢失。实测探针 512² 与
    参考 1024² 直接匹配得到 s=2.0000（正确），把探针先拉到 1024 则得 s=1.0（错误）。
    """
    import cv2

    ref_arr, ref_meta = _read_band(reference, max_side=max_side)
    prb_arr, prb_meta = _read_band(probe, max_side=max_side)

    ref_g = _to_gray(ref_arr, ref_meta.get("nodata"))
    prb_g = _to_gray(prb_arr, prb_meta.get("nodata"))

    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.005, edgeThreshold=20)
    k1, d1 = sift.detectAndCompute(ref_g, None)
    k2, d2 = sift.detectAndCompute(prb_g, None)
    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return None

    pairs = cv2.BFMatcher().knnMatch(d2, d1, k=2)
    good = [m for m, n in pairs if m.distance < 0.8 * n.distance]
    if len(good) < 12:
        return None

    p2 = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    p1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(p2, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None

    angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    scale = math.hypot(M[0, 0], M[1, 0])

    # 抽样比例要折算回原始分辨率坐标
    ref_ratio = ref_meta["width"] / ref_arr.shape[1]
    prb_ratio = prb_meta["width"] / prb_arr.shape[1]
    # probe(抽样) -> reference(抽样) 的 M，换算到原始像素坐标：
    #   p_ref_orig = ref_ratio * M * (p_prb_small) = ref_ratio * M / prb_ratio * p_prb_orig
    k = ref_ratio / prb_ratio
    M_orig = M.copy()
    M_orig[0, :] *= k
    M_orig[1, :] *= k

    return {"matrix_small": M, "matrix": M_orig,
            "angle": round(angle, 4), "scale": round(scale, 4),
            "inliers": int(inl.sum()) if inl is not None else 0,
            "matches": len(good),
            "ref_shape": (ref_meta["height"], ref_meta["width"]),
            "probe_shape": (prb_meta["height"], prb_meta["width"])}


def restore_and_decode(reference: Path, probe: Path, expected: int,
                       work_dir: Path | None = None, max_carriers: int | None = None) -> dict:
    """DEM 端到端：直解 -> SIFT 还原 -> 再解。

    返回结构对齐 ``trustmark_geo_restore.restore_and_decode``：found / decoded / via / rate / geometry。
    """
    import rasterio

    reference, probe = Path(reference), Path(probe)
    from . import dem_stream as D

    # 1) 直解
    try:
        r0 = D.cross_tile_extract(probe, expected)
        if r0.get("matched"):
            return {"found": True, "decoded": r0.get("decoded"), "via": "direct",
                    "hits": None, "total": None, "rate": None,
                    "detail": {"votes": r0.get("votes_for_best")},
                    "conclusion": "未变换，直接检出水印"}
    except Exception:  # noqa: BLE001
        r0 = {}

    # 2) 估计变换
    info = _estimate(reference, probe)
    if info is None:
        return {"found": False, "decoded": r0.get("decoded"), "via": "sift_failed",
                "hits": 0, "total": 0, "rate": 0.0,
                "detail": {"reason": "SIFT 无法估计变换"},
                "conclusion": "无法还原几何，未能判定"}

    # 3) 按参考图原始尺寸还原，必须 NEAREST（保住末位）
    import cv2
    prb_arr, prb_meta = _read_band(probe)          # 原始分辨率
    ref_h, ref_w = info["ref_shape"]
    rest = cv2.warpAffine(prb_arr, info["matrix"], (ref_w, ref_h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=prb_meta.get("nodata") or _NODATA_PROBE[0])

    ref_meta = rasterio.open(reference)
    try:
        prof = ref_meta.profile.copy()
    finally:
        ref_meta.close()
    prof.update(width=ref_w, height=ref_h, tiled=False, blockxsize=None, blockysize=None)

    work_dir = Path(work_dir) if work_dir else probe.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "dem_restored.tif"
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(rest.astype(prb_arr.dtype), 1)

    try:
        r1 = D.cross_tile_extract(out, expected)
        hit = bool(r1.get("matched"))
        return {"found": hit, "decoded": r1.get("decoded"), "via": "sift_restore",
                "hits": 1 if hit else 0, "total": 1, "rate": 1.0 if hit else 0.0,
                "geometry": {k: info[k] for k in ("angle", "scale", "inliers", "matches")},
                "detail": {"votes": r1.get("votes_for_best"),
                           "restored_file": str(out),
                           "note": "缩小超过 30% 会因重采样抹掉 LSB，还原后仍失败"},
                "conclusion": ("经几何还原后检出水印，判定为本系统流出" if hit else
                               "还原后仍未检出（缩小场景超出 LSB 水印能力边界）")}
    except Exception as e:  # noqa: BLE001
        return {"found": False, "decoded": None, "via": "sift_restore",
                "hits": 0, "total": 0, "rate": 0.0,
                "geometry": {k: info[k] for k in ("angle", "scale", "inliers", "matches")},
                "detail": {"error": f"{type(e).__name__}: {str(e)[:80]}"},
                "conclusion": "还原后提取异常，未能判定"}
