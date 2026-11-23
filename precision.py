from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "PrecisionError",
    "PRECISION_MODES",
    "PRECISION_METHODS",
    "reduce_geojson_precision",
    "douglas_peucker",
    "simplify_geojson_douglas_peucker",
    "resample_raster",
    "apply_precision_plan",
]

#: 前端可选降精度模式。NONE = 不降精度（默认）。
PRECISION_MODES = ("NONE", "DECIMALS", "DOUGLAS_PEUCKER", "RESAMPLE")

#: 重采样插值方法。
PRECISION_METHODS = ("nearest", "bilinear", "cubic", "average", "lanczos")


class PrecisionError(ValueError):
    """降精度参数非法，或该算子不适用于此数据类型。"""


def _round_coordinates(value: Any, decimals: int) -> Any:
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value):
            return [round(float(item), decimals) for item in value]
        return [_round_coordinates(item, decimals) for item in value]
    return value


def reduce_geojson_precision(source: Path, destination: Path, decimals: int) -> None:
    if decimals < 0 or decimals > 15:
        raise ValueError("decimals must be between 0 and 15")
    document = json.loads(Path(source).read_text(encoding="utf-8"))
    if document.get("type") == "FeatureCollection":
        features = document.get("features", [])
    elif document.get("type") == "Feature":
        features = [document]
    else:
        raise ValueError("GeoJSON must be a Feature or FeatureCollection")
    for feature in features:
        geometry = feature.get("geometry") or {}
        if "coordinates" in geometry:
            geometry["coordinates"] = _round_coordinates(geometry["coordinates"], decimals)
    Path(destination).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 道格拉斯-普克
# --------------------------------------------------------------------------- #

def _perpendicular_distance(point: tuple[float, float], start: tuple[float, float],
                            end: tuple[float, float]) -> float:
    """点到线段的垂距；起点终点重合时退化为点距。"""
    (px, py), (ax, ay), (bx, by) = point, start, end
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    # 线段参数化投影，t 夹到 [0,1]，保证是「线段」距离而非「无限直线」距离
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def douglas_peucker(points: list, tolerance: float) -> list:
    """经典道格拉斯-普克递归简化；保留首尾点。

    ``points`` 形如 ``[[x, y], ...]``（可带 z 分量，按前两维判定）。
    """
    if tolerance < 0:
        raise PrecisionError("tolerance must be >= 0")
    if len(points) <= 2:
        return [list(p) for p in points]
    start, end = points[0], points[-1]
    max_distance, index = -1.0, 0
    for i in range(1, len(points) - 1):
        distance = _perpendicular_distance((points[i][0], points[i][1]),
                                           (start[0], start[1]), (end[0], end[1]))
        if distance > max_distance:
            max_distance, index = distance, i
    if max_distance > tolerance:
        left = douglas_peucker(points[: index + 1], tolerance)
        right = douglas_peucker(points[index:], tolerance)
        return left[:-1] + right
    return [list(start), list(end)]


def _simplify_ring(ring: list, tolerance: float) -> list:
    """对环做简化；环若闭合（首尾相同）则保持闭合。"""
    if not ring or len(ring) < 4:
        return [list(p) for p in ring]
    closed = ring[0][:2] == ring[-1][:2]
    working = ring[:-1] if closed else ring
    if len(working) <= 2:
        return [list(p) for p in ring]
    simplified = douglas_peucker(working, tolerance)
    if closed and simplified[0][:2] != simplified[-1][:2]:
        simplified.append(list(simplified[0]))
    return simplified


def _simplify_geometry(geometry: dict, tolerance: float) -> None:
    kind = (geometry or {}).get("type")
    coordinates = (geometry or {}).get("coordinates")
    if coordinates is None:
        return
    if kind == "LineString":
        geometry["coordinates"] = douglas_peucker(coordinates, tolerance)
    elif kind in {"MultiLineString", "Polygon"}:
        geometry["coordinates"] = [_simplify_ring(part, tolerance) for part in coordinates]
    elif kind == "MultiPolygon":
        geometry["coordinates"] = [[_simplify_ring(ring, tolerance) for ring in polygon]
                                   for polygon in coordinates]
    elif kind == "Point":
        return
    else:
        raise PrecisionError(f"unsupported geometry type for simplification: {kind}")


def _count_vertices(document: dict) -> int:
    def walk(node: Any) -> int:
        if isinstance(node, list):
            if node and all(isinstance(item, (int, float)) for item in node):
                return 1
            return sum(walk(item) for item in node)
        return 0

    def geometry_vertices(geometry: dict) -> int:
        return walk((geometry or {}).get("coordinates") or [])

    if document.get("type") == "FeatureCollection":
        return sum(geometry_vertices(f.get("geometry") or {}) for f in document.get("features", []))
    return geometry_vertices(document.get("geometry") or document)


def simplify_geojson_douglas_peucker(source: Path, destination: Path, tolerance: float) -> dict:
    """对 GeoJSON 的线/面几何做道格拉斯-普克简化，返回统计报告。"""
    if tolerance < 0:
        raise PrecisionError("tolerance must be >= 0")
    document = json.loads(Path(source).read_text(encoding="utf-8"))
    if document.get("type") == "FeatureCollection":
        features = document.get("features", [])
    elif document.get("type") == "Feature":
        features = [document]
    else:
        raise PrecisionError("GeoJSON must be a Feature or FeatureCollection")
    before = _count_vertices(document)
    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("coordinates") is not None:
            _simplify_geometry(geometry, tolerance)
    after = _count_vertices(document)
    Path(destination).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "mode": "DOUGLAS_PEUCKER", "tolerance": tolerance,
        "vertices_before": before, "vertices_after": after,
        "vertices_removed": max(0, before - after),
        "reduction_ratio": round(1 - (after / before), 6) if before else 0.0,
        "features": len(features),
    }


# --------------------------------------------------------------------------- #
# 栅格重采样
# --------------------------------------------------------------------------- #

def resample_raster(source: Path, destination: Path, scale: float | None = None,
                    width: int | None = None, height: int | None = None,
                    method: str = "bilinear") -> dict:
    """栅格重采样（GeoTIFF / DEM 通用）。

    ``scale``（0<scale<=1 为降采样）与 ``width``/``height`` 二选一；同时给出时以显式尺寸为准。
    """
    import rasterio

    if width is None and height is None:
        if scale is None:
            raise PrecisionError("resample requires scale, or both width and height")
        if not 0 < scale <= 1:
            raise PrecisionError("scale must be in (0, 1] for downsampling")
    if method not in PRECISION_METHODS:
        raise PrecisionError(f"unsupported resampling method: {method}; use one of {list(PRECISION_METHODS)}")
    resampling = getattr(_resampling_enum(), method)

    with rasterio.open(source) as dataset:
        src_w, src_h = dataset.width, dataset.height
        dst_w = int(width) if width else max(1, int(round(src_w * float(scale))))
        dst_h = int(height) if height else max(1, int(round(src_h * float(scale))))
        data = dataset.read(out_shape=(dataset.count, dst_h, dst_w), resampling=resampling)
        profile = dataset.profile.copy()
        profile.update(width=dst_w, height=dst_h,
                       transform=dataset.transform * dataset.transform.scale((src_w / dst_w), (src_h / dst_h)))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination, "w", **profile) as sink:
            sink.write(data)

    return {
        "mode": "RESAMPLE", "method": method,
        "source_size": [src_w, src_h], "output_size": [dst_w, dst_h],
        "scale": round(dst_w / src_w, 6) if src_w else 0.0,
        "pixels_before": src_w * src_h, "pixels_after": dst_w * dst_h,
        "pixels_removed": max(0, src_w * src_h - dst_w * dst_h),
    }


def _resampling_enum():
    from rasterio.enums import Resampling  # 延迟导入：只有真正跑重采样时才需要
    return Resampling


def apply_precision_plan(source: Path, destination: Path, *, mode: str,
                         decimals: int | None = None, tolerance: float | None = None,
                         scale: float | None = None, width: int | None = None,
                         height: int | None = None, method: str = "bilinear",
                         data_type: str | None = None) -> dict | None:
    """按 ``mode`` 执行降精度并返回报告；``NONE`` 返回 ``None``（调用方直接用源文件）。

    这是**唯一**的降精度入口，保证「参数不适用就报错」而不是静默空转。
    """
    normalized = (mode or "NONE").upper()
    if normalized not in PRECISION_MODES:
        raise PrecisionError(f"unsupported precision_mode: {mode}; use one of {list(PRECISION_MODES)}")
    if normalized == "NONE":
        return None

    kind = (data_type or "").upper()
    vector_kinds = {"GEOJSON", "SHP", "GPKG", "GEOPACKAGE", "DLG"}
    raster_kinds = {"GEOTIFF", "GTIFF", "DOM", "DEM", "TIF", "TIFF"}

    if normalized == "DECIMALS":
        if decimals is None:
            raise PrecisionError("precision_mode=DECIMALS requires precision_decimals")
        if kind in {"GEOJSON"}:
            reduce_geojson_precision(source, destination, decimals)
            return {"mode": "DECIMALS", "decimals": decimals,
                    "applies_to": "GeoJSON coordinates (round)", "file": str(destination)}
        if kind in {"SHP", "DLG"}:
            return process_shapefile_geometry(source, destination, mode="DECIMALS", decimals=decimals)
        if kind in {"GPKG", "GEOPACKAGE"}:
            return process_geopackage_geometry(source, destination, mode="DECIMALS", decimals=decimals)
        raise PrecisionError(
            f"precision_mode=DECIMALS is only implemented for vector data "
            f"(GeoJSON/SHP/GPKG); got data_type={data_type}")

    if normalized == "DOUGLAS_PEUCKER":
        if tolerance is None:
            raise PrecisionError("precision_mode=DOUGLAS_PEUCKER requires douglas_peucker_tolerance")
        if kind in {"GEOJSON"}:
            return simplify_geojson_douglas_peucker(source, destination, tolerance)
        if kind in {"SHP", "DLG"}:
            return process_shapefile_geometry(source, destination, mode="DOUGLAS_PEUCKER", tolerance=tolerance)
        if kind in {"GPKG", "GEOPACKAGE"}:
            return process_geopackage_geometry(source, destination, mode="DOUGLAS_PEUCKER", tolerance=tolerance)
        raise PrecisionError(
            "precision_mode=DOUGLAS_PEUCKER is implemented for vector data "
            f"(GeoJSON/SHP/GPKG); got data_type={data_type}")

    if normalized == "RESAMPLE":
        if kind not in raster_kinds:
            raise PrecisionError(
                f"precision_mode=RESAMPLE is only implemented for raster data; got data_type={data_type}")
        return resample_raster(source, destination, scale=scale, width=width, height=height, method=method)

    raise PrecisionError(f"unhandled precision_mode: {normalized}")


# --------------------------------------------------------------------------- #
# 矢量几何层：SHP / GPKG 的道格拉斯-普克与坐标精度
#
# 约束：本机没有 GDAL/OGR（`import osgeo` 失败），因此几何读写走 pyshp（SHP）与
# sqlite3 + WKB（GPKG）。两边都自建最小 WKB 编解码，只支持平台实际会遇到的几何类型。
# --------------------------------------------------------------------------- #

_WKB_GEOM = {
    1: "Point", 2: "LineString", 3: "Polygon", 4: "MultiPoint",
    5: "MultiLineString", 6: "MultiPolygon",
}
_WKB_CODE = {v: k for k, v in _WKB_GEOM.items()}


def _read_wkb_point(buf: bytes, offset: int, ndim: int) -> tuple[list, int]:
    import struct

    values = struct.unpack_from("<" + "d" * ndim, buf, offset)
    return list(values[:2]), offset + 8 * ndim


def _read_wkb(buf: bytes, offset: int = 0) -> tuple[dict, int]:
    """最小 WKB 解析器：支持 Point/LineString/Polygon/Multi*（含 Z 分量，按 2 维简化）。"""
    import struct

    (order,) = struct.unpack_from("<B", buf, offset)
    endian = "<" if order == 1 else ">"
    (raw_type,) = struct.unpack_from(endian + "I", buf, offset + 1)
    offset += 5
    # 处理 ISO WKB 的 Z/M 标志（1000/2000/3000 偏移）
    has_z = raw_type >= 1000
    base = raw_type - 1000 * (raw_type // 1000)
    ndim = 3 if has_z else 2
    kind = _WKB_GEOM.get(base)
    if kind is None:
        raise PrecisionError(f"unsupported WKB geometry type: {raw_type}")

    def read_point(off):
        return _read_wkb_point(buf, off, ndim)

    if kind == "Point":
        point, offset = read_point(offset)
        return {"type": "Point", "coordinates": point}, offset
    if kind == "LineString":
        (count,) = struct.unpack_from(endian + "I", buf, offset)
        offset += 4
        coords = []
        for _ in range(count):
            point, offset = read_point(offset)
            coords.append(point)
        return {"type": "LineString", "coordinates": coords}, offset
    if kind == "Polygon":
        (rings,) = struct.unpack_from(endian + "I", buf, offset)
        offset += 4
        polygons = []
        for _ in range(rings):
            (points,) = struct.unpack_from(endian + "I", buf, offset)
            offset += 4
            ring = []
            for _ in range(points):
                point, offset = read_point(offset)
                ring.append(point)
            polygons.append(ring)
        return {"type": "Polygon", "coordinates": polygons}, offset
    # Multi*：子几何各自带完整的类型头，递归读取
    (count,) = struct.unpack_from(endian + "I", buf, offset)
    offset += 4
    parts = []
    for _ in range(count):
        sub, offset = _read_wkb(buf, offset)
        parts.append(sub)
    name = {4: "MultiPoint", 5: "MultiLineString", 6: "MultiPolygon"}[base]
    if name == "MultiPoint":
        return {"type": name, "coordinates": [item["coordinates"] for item in parts]}, offset
    if name == "MultiLineString":
        return {"type": name, "coordinates": [item["coordinates"] for item in parts]}, offset
    return {"type": name, "coordinates": [item["coordinates"] for item in parts]}, offset


def _write_wkb_point(out: bytearray, point, ndim: int) -> None:
    import struct

    values = list(point) + [0.0] * (ndim - len(point))
    out.extend(struct.pack("<" + "d" * ndim, *values[:ndim]))


def _write_wkb(geometry: dict, *, ndim: int = 2) -> bytes:
    import struct

    kind = geometry["type"]
    code = _WKB_CODE[kind] + (1000 if ndim == 3 else 0)
    out = bytearray()
    out.append(1)
    out.extend(struct.pack("<I", code))
    coordinates = geometry["coordinates"]
    if kind == "Point":
        _write_wkb_point(out, coordinates, ndim)
    elif kind == "LineString":
        out.extend(struct.pack("<I", len(coordinates)))
        for point in coordinates:
            _write_wkb_point(out, point, ndim)
    elif kind == "Polygon":
        out.extend(struct.pack("<I", len(coordinates)))
        for ring in coordinates:
            out.extend(struct.pack("<I", len(ring)))
            for point in ring:
                _write_wkb_point(out, point, ndim)
    else:
        out.extend(struct.pack("<I", len(coordinates)))
        sub_code = {"MultiPoint": 1, "MultiLineString": 2, "MultiPolygon": 3}[kind]
        for part in coordinates:
            if kind == "MultiPoint":
                inner = {"type": "Point", "coordinates": part}
            elif kind == "MultiLineString":
                inner = {"type": "LineString", "coordinates": part}
            else:
                inner = {"type": "Polygon", "coordinates": part}
            out.extend(_write_wkb(inner, ndim=ndim))
    return bytes(out)


def _round_geometry(geometry: dict, decimals: int) -> dict:
    def walk(node):
        if isinstance(node, (list, tuple)):
            if node and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in node):
                return [round(float(item), decimals) for item in node]
            return [walk(item) for item in node]
        return node

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def _simplify_geometry_wkb(geometry: dict, tolerance: float) -> dict:
    kind = geometry["type"]
    coords = geometry["coordinates"]
    if kind == "LineString":
        return {"type": kind, "coordinates": douglas_peucker(coords, tolerance)}
    if kind == "Polygon":
        return {"type": kind, "coordinates": [_simplify_ring(ring, tolerance) for ring in coords]}
    if kind == "MultiLineString":
        return {"type": kind, "coordinates": [douglas_peucker(part, tolerance) for part in coords]}
    if kind == "MultiPolygon":
        return {"type": kind,
                "coordinates": [[_simplify_ring(ring, tolerance) for ring in polygon] for polygon in coords]}
    if kind in {"Point", "MultiPoint"}:
        return geometry
    raise PrecisionError(f"unsupported geometry type for simplification: {kind}")


def _geometry_vertex_count(geometry) -> int:
    """统计坐标点数。支持 dict（GeoJSON 风格）与 pyshp 的 `__geo_interface__` 产出。"""
    if isinstance(geometry, dict):
        coordinates = geometry.get("coordinates")
    else:
        return 0

    def walk(node):
        if isinstance(node, (list, tuple)):
            if node and all(isinstance(item, (int, float)) for item in node):
                return 1
            return sum(walk(item) for item in node)
        return 0

    return walk(coordinates)


# --------------------------------------------------------------------------- #
# GPKG：几何以 blob 存在 features 表，需保留 GP 头与原 envelope
# --------------------------------------------------------------------------- #

def _gpkg_parts(blob: bytes) -> tuple[bytes, list[str], bool]:
    """拆 GPKG 几何 blob：返回（头部字节, 表名列表, 是否小端）。

    头部 = magic(2) + version(1) + flags(1) + srs_id(4) + envelope(N)
    """
    import struct

    if blob[:2] != b"GP":
        raise PrecisionError("not a GeoPackage geometry blob (missing 'GP' magic)")
    flags = blob[3]
    little = bool(flags & 0x01)
    indicator = (flags >> 1) & 0x07
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[indicator]
    header_len = 8 + env_len
    return blob[:8], [str(indicator)], little


def _gpkg_geometry_offset(blob: bytes) -> int:
    flags = blob[3]
    indicator = (flags >> 1) & 0x07
    env_len = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[indicator]
    return 8 + env_len


def _rebuild_gpkg_blob(blob: bytes, geometry: dict) -> bytes:
    """重算 envelope 后重建 GPKG blob（坐标变了，envelope 必须跟着变）。"""
    import struct

    offset = _gpkg_geometry_offset(blob)
    header = bytearray(blob[:offset])
    geometry_wkb = _write_wkb(geometry)
    flags = blob[3]
    indicator = (flags >> 1) & 0x07
    xs, ys = [], []

    def collect(node):
        if isinstance(node, list):
            if node and all(isinstance(item, (int, float)) for item in node):
                if len(node) >= 2:
                    xs.append(float(node[0]))
                    ys.append(float(node[1]))
                return
            for item in node:
                collect(item)

    collect(geometry["coordinates"])
    if indicator and xs:
        values = [min(xs), max(xs), min(ys), max(ys)]
        if indicator in (2, 4):
            values += [0.0, 0.0]
        if indicator == 4:
            values += [0.0, 0.0]
        header[8:8 + len(values) * 8] = struct.pack("<" + "d" * len(values), *values)
    return bytes(header) + geometry_wkb


def _iter_gpkg_feature_tables(connection) -> list[tuple[str, str]]:
    """返回 [(表名, 几何列名)]。"""
    rows = connection.execute(
        "SELECT table_name, column_name FROM gpkg_geometry_columns ORDER BY table_name"
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def process_geopackage_geometry(source: Path, destination: Path, *, mode: str,
                                decimals: int | None = None,
                                tolerance: float | None = None) -> dict:
    """对 GeoPackage 所有要素表的几何做降精度，返回统计报告。"""
    import shutil
    import sqlite3

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)

    connection = sqlite3.connect(destination)
    try:
        tables = _iter_gpkg_feature_tables(connection)
        if not tables:
            raise PrecisionError("GeoPackage contains no feature geometry columns")
        before_total = after_total = 0
        features = 0
        for table, geom_column in tables:
            rows = connection.execute(
                f'SELECT rowid, "{geom_column}" FROM "{table}" WHERE "{geom_column}" IS NOT NULL'
            ).fetchall()
            updates = []
            for rowid, blob in rows:
                offset = _gpkg_geometry_offset(blob)
                geometry, _ = _read_wkb(blob, offset)
                before_total += _geometry_vertex_count(geometry)
                if mode == "DOUGLAS_PEUCKER":
                    geometry = _simplify_geometry_wkb(geometry, tolerance)
                else:
                    geometry = _round_geometry(geometry, decimals)
                after_total += _geometry_vertex_count(geometry)
                updates.append((_rebuild_gpkg_blob(blob, geometry), rowid))
                features += 1
            connection.executemany(
                f'UPDATE "{table}" SET "{geom_column}" = ? WHERE rowid = ?', updates
            )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()

    report = {"mode": mode, "format": "GPKG", "features": features, "tables": [t for t, _ in tables],
              "vertices_before": before_total, "vertices_after": after_total}
    if mode == "DOUGLAS_PEUCKER":
        report.update({"tolerance": tolerance,
                       "vertices_removed": max(0, before_total - after_total),
                       "reduction_ratio": round(1 - (after_total / before_total), 6) if before_total else 0.0})
    else:
        report.update({"decimals": decimals})
    return report


# --------------------------------------------------------------------------- #
# SHP：用 pyshp 读写，几何与 DBF 属性一并保留
# --------------------------------------------------------------------------- #

def _shp_targets(source: Path) -> list[Path]:
    source = Path(source)
    if source.is_dir():
        return sorted(item for item in source.rglob("*.shp") if item.is_file())
    if source.suffix.lower() == ".shp":
        return [source]
    return []


def process_shapefile_geometry(source, destination, *, mode: str,
                               decimals: int | None = None,
                               tolerance: float | None = None) -> dict:
    """对 Shapefile 几何做降精度；``source`` 可为 .shp 或目录，``destination`` 为目录。"""
    import shapefile

    sources = _shp_targets(source)
    if not sources:
        raise PrecisionError("no .shp file found in source")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if Path(source).is_file() and destination.resolve() == Path(source).parent.resolve() and len(sources) == 1:
        pass  # 目标目录即源目录时不额外拷贝逻辑，直接写出同名文件由 EPSG 侧处理

    before_total = after_total = 0
    files = 0
    for shp in sources:
        relative = Path(shp.name) if Path(source).is_file() else shp.relative_to(Path(source))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        reader = shapefile.Reader(str(shp), encoding="gbk", encodingErrors="replace")
        writer = shapefile.Writer(str(target.with_suffix("")), encoding="gbk", encodingErrors="replace")
        writer.fields = list(reader.fields)[1:]
        for shape_record in reader.iterShapeRecords():
            geometry = shape_record.shape.__geo_interface__
            before_total += _geometry_vertex_count(geometry)
            if mode == "DOUGLAS_PEUCKER":
                geometry = _simplify_geometry_wkb(geometry, tolerance)
            else:
                geometry = _round_geometry(geometry, decimals)
            after_total += _geometry_vertex_count(geometry)
            writer.record(*shape_record.record)
            writer.shape(geometry)
        writer.close()
        files += 1
    report = {"mode": mode, "format": "SHP", "files": files, "layers": [p.name for p in sources],
              "vertices_before": before_total, "vertices_after": after_total}
    if mode == "DOUGLAS_PEUCKER":
        report.update({"tolerance": tolerance,
                       "vertices_removed": max(0, before_total - after_total),
                       "reduction_ratio": round(1 - (after_total / before_total), 6) if before_total else 0.0})
    else:
        report.update({"decimals": decimals})
    return report
