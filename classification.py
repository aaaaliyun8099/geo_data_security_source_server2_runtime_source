from __future__ import annotations

from enum import Enum


class Classification(str, Enum):
    DLG_LINEWORK = "DLG_LINEWORK"
    REMOTE_SENSING_RASTER = "REMOTE_SENSING_RASTER"
    THEMATIC_VECTOR = "THEMATIC_VECTOR"
    OBLIQUE_MESH = "OBLIQUE_MESH"
    ELEVATION_DEM = "ELEVATION_DEM"
    ORTHOIMAGE = "ORTHOIMAGE"
    POINT_CLOUD = "POINT_CLOUD"


CAPABILITIES = {
    Classification.DLG_LINEWORK: {"implemented": True, "data_types": ["SHP", "GPKG"], "watermark_domain": "encrypted_attribute",
                                   "precision_reduction": ["DOUGLAS_PEUCKER", "DECIMALS"]},
    Classification.REMOTE_SENSING_RASTER: {"implemented": True, "data_types": ["GEOTIFF"], "watermark_domain": "pixel",
                                            "precision_reduction": ["RESAMPLE"]},
    Classification.THEMATIC_VECTOR: {"implemented": True, "data_types": ["SHP", "GEOJSON"], "watermark_domain": "encrypted_attribute",
                                      "precision_reduction": ["DOUGLAS_PEUCKER", "DECIMALS"]},
    Classification.OBLIQUE_MESH: {"implemented": True, "data_types": ["OSGB", "TEXTURES"], "watermark_domain": "pixel"},
    # DEM：跨瓦片（稀疏副本）LSB，提取与瓦片划分无关，可裁剪后提取。
    # 几何：W_min = replica_stride - 1 + replica_length = 96 - 1 + 32 = 127。
    Classification.ELEVATION_DEM: {"implemented": True, "data_types": ["DEM"], "watermark_domain": "cross_tile_lsb",
                                   "scheme": "cross_tile_v2", "robustness": "grid_independent_extraction",
                                   "extract_after_crop": True, "min_extract_window_width": 127,
                                   "replica_length": 32, "replica_stride": 96,
                                   "measured_footprint_ratio": 0.171818,
                                   "measured_window_hit": "1521/1521 (9 sizes x 169 offsets, real DEM)"},
    # ORTHOIMAGE：QIM-DCT 抗压载体；抗压矩阵在单幅 2048x2048 场景实测 17/20
    # （未过：高斯噪声 sigma10、中心 80% 遮挡、强锐化）。单场景结果，不等于全量验收。
    Classification.ORTHOIMAGE: {"implemented": True, "data_types": ["GEOTIFF", "TEXTURES"], "watermark_domain": "pixel",
                                "robustness": "measured_matrix_17_of_20_single_scene",
                                "matrix_source": "docs/ORTHOIMAGE_抗压矩阵_实测.json",
                                "matrix_scope": "single 2048x2048 scene, not full-data acceptance",
                                "precision_reduction": ["RESAMPLE"],
                                "visible_watermark": True},
    # POINT_CLOUD：文本格式（XYZ/PTS/PCD-ascii/PLY-ascii）为坐标域 LSB；
    # 二进制格式（LAS/LAZ/E57/binary PLY/PCD）显式报错，不静默退回 sidecar。
    Classification.POINT_CLOUD: {"implemented": True, "data_types": ["POINT_CLOUD"], "watermark_domain": "coordinate_lsb",
                                 "coordinate_domain_verified": True,
                                 "coordinate_formats": [".xyz", ".pts", ".pcd", ".ply"],
                                 "binary_formats_unsupported": [".las", ".laz", ".e57"]},
}


# 简写兼容表：前端与调用方习惯用数据类型名当密级传（如 classification="DLG"），
# 而平台内部的密级枚举是业务语义名（DLG_LINEWORK）。两者不是同一概念，但人写起来
# 极易混用。这里做只读映射，把常见简写收进来，避免 422 把用户挡在门外。
# 映射必须唯一：一个简写只能指向一个密级，否则会有歧义，宁可不猜。
CLASSIFICATION_ALIASES = {
    "DLG": Classification.DLG_LINEWORK,
    "DLG_LINEWORK": Classification.DLG_LINEWORK,
    "SHP": Classification.DLG_LINEWORK,
    "GPKG": Classification.DLG_LINEWORK,
    "GEOPACKAGE": Classification.DLG_LINEWORK,
    "DOM": Classification.ORTHOIMAGE,
    "ORTHO": Classification.ORTHOIMAGE,
    "ORTHOIMAGE": Classification.ORTHOIMAGE,
    "DEM": Classification.ELEVATION_DEM,
    "ELEVATION_DEM": Classification.ELEVATION_DEM,
    "OSGB": Classification.OBLIQUE_MESH,
    "MESH": Classification.OBLIQUE_MESH,
    "OBLIQUE_MESH": Classification.OBLIQUE_MESH,
    "TEXTURES": Classification.OBLIQUE_MESH,
    "POINT_CLOUD": Classification.POINT_CLOUD,
    "LAS": Classification.POINT_CLOUD,
    "GEOJSON": Classification.THEMATIC_VECTOR,
    "THEMATIC_VECTOR": Classification.THEMATIC_VECTOR,
    "GEOTIFF": Classification.REMOTE_SENSING_RASTER,
    "REMOTE_SENSING_RASTER": Classification.REMOTE_SENSING_RASTER,
}


def parse_classification(value: str) -> Classification:
    """把调用方给的密级字符串解析成 Classification，兼容常见简写。

    顺序：先按枚举精确匹配，再查简写表。两者都不命中才报错，
    并在报错里同时列出枚举值与可用简写，让调用方知道该怎么改。
    """
    try:
        text = value.strip().upper()
    except AttributeError as exc:
        raise ValueError(f"unsupported classification: {value!r}") from exc

    try:
        return Classification(text)
    except ValueError:
        pass

    alias = CLASSIFICATION_ALIASES.get(text)
    if alias is not None:
        return alias

    accepted = ", ".join(item.value for item in Classification)
    raise ValueError(
        f"unsupported classification: {value!r}; accepted values: {accepted}; "
        f"常见简写也可接受：{', '.join(sorted(CLASSIFICATION_ALIASES))}")


def allowed_for_data_type(classification: Classification, data_type: str) -> bool:
    data_type = data_type.upper()
    allowed = {
        "SHP": {Classification.DLG_LINEWORK, Classification.THEMATIC_VECTOR},
        # GPKG 是一等数据类型，与 SHP 同属矢量载体；缺这个键会让 GPKG 永远判不通过。
        "GPKG": {Classification.DLG_LINEWORK, Classification.THEMATIC_VECTOR},
        "GEOPACKAGE": {Classification.DLG_LINEWORK, Classification.THEMATIC_VECTOR},
        "GEOJSON": {Classification.THEMATIC_VECTOR},
        "GEOTIFF": {Classification.REMOTE_SENSING_RASTER, Classification.ELEVATION_DEM, Classification.ORTHOIMAGE},
        "GTIFF": {Classification.REMOTE_SENSING_RASTER, Classification.ELEVATION_DEM, Classification.ORTHOIMAGE},
        "TEXTURES": {Classification.OBLIQUE_MESH, Classification.ORTHOIMAGE},
        "OSGB": {Classification.OBLIQUE_MESH},
        "OSGB_TEXTURES": {Classification.OBLIQUE_MESH},
        "DEM": {Classification.ELEVATION_DEM},
        "POINT_CLOUD": {Classification.POINT_CLOUD},
    }
    return classification in allowed.get(data_type, set())
