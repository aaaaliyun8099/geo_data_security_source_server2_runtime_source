"""降精度安全范围：客户可在前端选择降精度强度，但**必须保证水印仍可嵌入并可验证**。

问题
----
客户希望自己选"降多少"。但降精度是水印的**上游**（先降精度、后嵌水印），
降过头会：
  * 栅格：缩小到载体不足 → TrustMark 找不到可承载的 256 块，水印嵌不进去；
  * 矢量：道格拉斯-普克把要素抽成 0/1 个顶点 → 几何无处落水印；
  * 坐标精度：小数位截断到 0 会把相邻要素塌成同一点，同样失去载体。

因此不能只做 `gt=0, le=1` 这种形式校验，必须**按数据类型给出可用的最小/最大值**，
并在提交时用真实载体量算出该次请求是否可行（`422` + 可用的安全范围），而不是让它跑到一半才失败。

本模块只做**范围判定与说明**，不改水印算法。
"""
from __future__ import annotations

import math
from typing import Any

# ── 硬性下限：来自各水印算法本身，不可调 ────────────────────────────────────
# DEM 跨瓦片：行内每 64 采样放一个完整副本，窗口宽 ≥ 2*64-1=127 才能保证完整副本落进来。
DEM_CROSS_TILE_MIN_WIDTH = 127
# TrustMark 栅格载体：密集栅格（GeoTIFF/DOM/影像）的载体已由 512 改为 256（DENSE_RASTER_TILE），
# 因此整幅至少要放得下一个 256 块。原先写死 512 会把 256..511 px 的图误判为"不可降精度"，
# 而这类图实际可以正常嵌入并提取（实测 400×400、300×300 均嵌入成功、提取命中）。
RASTER_MIN_SIDE = 256
# 点云坐标域：LSB 需要足够的坐标槽位承载 payload（≈112 bit）。
# 点云：坐标域水印每比特占 1 个坐标槽；比特布局 = magic(16)+length(16)+payload(8/字节)+checksum(16)。
# 因此所需槽位 = 48 + 8 * 载荷字节数，不能用固定点数糊弄。
POINT_CLOUD_HEADER_SLOTS = 48
POINT_CLOUD_DEFAULT_PAYLOAD_BYTES = 8   # 默认水印文本 "20261016" 的长度
# 矢量：几何域/属性域都需要至少一个要素；1 个要素无法体现"多要素一致性"，因此要求 ≥ 2。
VECTOR_MIN_FEATURES = 2
# 坐标精度安全下限：0 位会把坐标压到整度，相邻要素可能塌陷成同一点，水印载体随之消失。
VECTOR_MIN_DECIMALS = 1
# 单要素至少要留够落水印的顶点数（道格拉斯-普克后）。
VECTOR_MIN_VERTICES_PER_FEATURE = 2


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def raster_bounds(width: int | None, height: int | None, *, cross_tile: bool = False) -> dict[str, Any]:
    """栅格降精度的可用范围。

    返回 ``min_scale`` = 能保住水印的最小缩放比例（0–1），以及对应的最小边长。
    """
    w, h = _as_int(width), _as_int(height)
    if not w or not h or w <= 0 or h <= 0:
        return {"known": False, "reason": "缺少栅格尺寸，无法给出降精度安全范围"}

    floor = DEM_CROSS_TILE_MIN_WIDTH if cross_tile else RASTER_MIN_SIDE
    min_side = min(w, h)
    if min_side < floor:
        return {
            "known": True, "feasible": False,
            "reason": f"源数据最短边 {min_side} px 已低于水印载体下限 {floor} px，不能再降精度",
            "min_side_required": floor, "source_short_side": min_side,
        }
    min_scale = round(floor / min_side, 6)
    # 用 ceil 而不是 int 截断：300*0.853333=255.9999，截断会给出 255 —— 低于载体下限 256，
    # 等于把"刚好合法"的边界算成不合法。
    min_width = max(floor, math.ceil(w * min_scale))
    min_height = max(floor, math.ceil(h * min_scale))
    return {
        "known": True, "feasible": True,
        "min_side_required": floor,
        "source_short_side": min_side,
        # 客户可选范围：最小缩放（再小就保不住水印）～ 1.0（不缩放）
        "min_scale": min_scale, "max_scale": 1.0,
        "min_width": min_width, "min_height": min_height,
        "note": _raster_note(floor, w, h, min_scale, cross_tile, min_width, min_height),
    }


def _raster_note(floor: int, w: int, h: int, min_scale: float, cross_tile: bool,
                 min_width: int | None = None, min_height: int | None = None) -> str:
    mw = min_width if min_width is not None else int(w * min_scale)
    mh = min_height if min_height is not None else int(h * min_scale)
    note = (f"该类型水印需要 ≥{floor}px 的载体；本图 {w}×{h}，故缩放比例不得小于 "
            f"{min_scale}（即不得缩到 {mw}×{mh} 以下）")
    if cross_tile:
        # 诚实标注：DEM 跨瓦片的 127px 是几何下限，实测并非 100% 命中
        # （有合规尺寸窗口漏检）。这是"可嵌入"的下限，不是"保证可提取"。
        note += ("。注意：DEM 跨瓦片路径实测命中率并非 100%，此下限只保证水印可嵌入，"
                 "不构成可提取性保证")
    return note


def vector_bounds(features: int | None, vertices: int | None = None,
                  *, tolerance_is_absolute: bool = True) -> dict[str, Any]:
    """矢量降精度（道格拉斯-普克 / 坐标精度）的可用范围。"""
    f = _as_int(features)
    if f is None:
        return {"known": False, "reason": "缺少要素数，无法给出降精度安全范围"}
    if f < VECTOR_MIN_FEATURES:
        return {
            "known": True, "feasible": False,
            "reason": f"源数据仅 {f} 个要素，低于水印下限 {VECTOR_MIN_FEATURES}，不能再降精度",
            "min_features_required": VECTOR_MIN_FEATURES, "source_features": f,
        }
    out: dict[str, Any] = {
        "known": True, "feasible": True,
        "min_features_required": VECTOR_MIN_FEATURES,
        "source_features": f,
        # DECIMALS 模式：小数位 0–15，但 0 位会把坐标压到整度，跨要素塌陷风险高 → 建议下限 1
        "decimals_min": 1, "decimals_max": 15,
    }
    v = _as_int(vertices)
    if v:
        out["source_vertices"] = v
        out["min_vertices_required"] = VECTOR_MIN_VERTICES_PER_FEATURE * f
        if tolerance_is_absolute:
            # 绝对容差没有普适上界（取决于坐标量级），给"顶点保留"口径而不是编一个数
            out["tolerance_note"] = (
                "绝对容差无普适上界；请以保留顶点数为准："
                f"简化后总顶点须 ≥ {out['min_vertices_required']}（每要素 ≥ {VECTOR_MIN_VERTICES_PER_FEATURE}）"
            )
    return out


def point_cloud_bounds(points: int | None, payload_bytes: int | None = None) -> dict[str, Any]:
    """点云坐标域水印的载体下限。

    真实约束来自 `point_cloud._pack_bits`：比特布局是
    `magic(16) | length(16) | payload(8/字节) | checksum(16)`，**每比特占一个坐标槽位**。
    所以所需槽位 `= 48 + 8 * 载荷字节数`；坐标槽位数 = 点数 × 每行坐标列数
    （ASCII 点云一般 3 列，故每点约 3 槽）。这里用保守口径：**每点按 1 槽**算下限。
    """
    p = _as_int(points)
    if p is None:
        return {"known": False, "reason": "缺少点数，无法给出降精度安全范围"}
    needed = POINT_CLOUD_HEADER_SLOTS + 8 * _as_int(payload_bytes or POINT_CLOUD_DEFAULT_PAYLOAD_BYTES)
    if p < needed:
        return {
            "known": True, "feasible": False,
            "reason": (f"点云仅 {p} 点，低于坐标域水印所需的 {needed} 个槽位"
                       f"（= 头部 48 + 载荷 8×{_as_int(payload_bytes or POINT_CLOUD_DEFAULT_PAYLOAD_BYTES)}；"
                       f"按每点至少 1 槽的保守口径）"),
            "slots_required": needed, "source_points": p,
        }
    return {
        "known": True, "feasible": True,
        "slots_required": needed, "source_points": p,
        "note": (f"坐标域水印需 ≥{needed} 个坐标槽位（每比特 1 槽）；"
                 f"本点云 {p} 点，按每点 ≥1 槽计可承载"),
    }


def bounds_for(data_type: str, *, width=None, height=None, features=None,
               vertices=None, points=None) -> dict[str, Any]:
    """按数据类型分派到对应范围计算。"""
    kind = (data_type or "").upper()
    if kind in {"DEM"}:
        return raster_bounds(width, height, cross_tile=True)
    if kind in {"GEOTIFF", "GTIFF", "TIF", "TIFF", "DOM"}:
        return raster_bounds(width, height)
    if kind in {"SHP", "DLG", "GEOJSON", "GPKG", "GEOPACKAGE"}:
        return vector_bounds(features, vertices)
    if kind == "POINT_CLOUD":
        return point_cloud_bounds(points)
    return {"known": False, "reason": f"数据类型 {data_type} 暂无降精度安全范围定义"}


def check_resample(width=None, height=None, *, cross_tile=False,
                   scale=None, target_width=None, target_height=None) -> dict[str, Any]:
    """判定一次 RESAMPLE 请求是否越过水印载体下限。"""
    bounds = raster_bounds(width, height, cross_tile=cross_tile)
    if not bounds.get("known"):
        return {"ok": True, "bounds": bounds, "reason": None}
    if not bounds.get("feasible"):
        return {"ok": False, "bounds": bounds, "reason": bounds["reason"]}

    floor = bounds["min_side_required"]
    w, h = _as_int(width), _as_int(height)
    target = None
    if target_width and target_height:
        target = (int(target_width), int(target_height))
    elif scale is not None:
        target = (int(w * float(scale)), int(h * float(scale)))
    if target is None:
        return {"ok": True, "bounds": bounds, "reason": None, "target": None}

    short = min(target)
    if short < floor:
        return {
            "ok": False, "bounds": bounds, "target": list(target),
            "reason": (
                f"降精度后最短边 {short} px 低于水印载体下限 {floor} px，"
                f"水印将无法嵌入；最小可用缩放比例 {bounds['min_scale']}"
                f"（对应 {bounds['min_width']}×{bounds['min_height']}）"
            ),
        }
    return {"ok": True, "bounds": bounds, "target": list(target), "reason": None}


def check_decimals(features=None, *, decimals=None) -> dict[str, Any]:
    """判定坐标精度是否越过下限。"""
    bounds = vector_bounds(features)
    if not bounds.get("known"):
        return {"ok": True, "bounds": bounds, "reason": None}
    if not bounds.get("feasible"):
        return {"ok": False, "bounds": bounds, "reason": bounds["reason"]}
    if decimals is None:
        return {"ok": True, "bounds": bounds, "reason": None}
    d = _as_int(decimals)
    if d is not None and d < bounds["decimals_min"]:
        return {
            "ok": False, "bounds": bounds,
            "reason": (f"坐标精度 {d} 位低于安全下限 {bounds['decimals_min']} 位；"
                       f"0 位会把坐标压到整度并可能使相邻要素塌陷，水印载体随之消失"),
        }
    return {"ok": True, "bounds": bounds, "reason": None}
