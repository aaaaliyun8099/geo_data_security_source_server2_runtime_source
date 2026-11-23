"""四类数据统一溯源分派。

每一类走不同的水印通道，能力边界各不相同。本模块的职责只是**分派 + 统一返回结构**，
不做能力夸大：返回里的 ``capability`` 字段如实写明该类通道能扛什么、扛不住什么。

分派依据（按优先级）：
  1. 扩展名 .osgb / .osg        -> OSGB 纹理 DCT
  2. 扩展名 .gpkg / .sqlite     -> GPKG 属性字段 _wm
  3. 扩展名 .tif/.tiff/.img 等   -> 看标签：
        GDS_TM_PAYLOAD  有值      -> TrustMark 载体（DOM 成果）
        GDS_DEM_WATERMARK 有值    -> DEM 整数值 LSB
        否则                      -> 栅格通用（尝试 TrustMark，再尝试 DEM）
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_RASTER_EXT = {".tif", ".tiff", ".img", ".vrt", ".jp2", ".png", ".bmp", ".webp"}
# PNG/BMP/WEBP 是"像素域容器"，与 TIFF 同一套 TrustMark 载体；但没有地理标签，
# 分类时不能因为读不到 rasterio 标签就判成未知。见 classify()。
_PIXEL_CONTAINER_EXT = {".png", ".bmp", ".webp"}


def classify(path: Path) -> tuple[str, dict]:
    """返回 (类别, 附加信息)。类别取值：OSGB / GPKG / RASTER_TRUSTMARK / RASTER_DEM / RASTER_UNKNOWN。"""
    ext = path.suffix.lower()
    if ext in {".osgb", ".osg"}:
        return "OSGB", {}
    # 纹理图片本身就是水印载体：嵌入时按张处理的 JPEG/PNG，提取时按同一通道读。
    # 判成 UNKNOWN 会让前端拿到"不支持的文件类型：.jpg"，而水印其实就在里面。
    # PNG 例外：它同时可能是"DOM 成果转存的无损容器"（TrustMark 载体还在），
    # 所以 PNG 交给下面的栅格分支按标签/像素域判定，不在这里固定成纹理。
    if ext in {".jpg", ".jpeg"}:
        return "OSGB_TEXTURE", {}
    if ext in {".gpkg", ".sqlite", ".db"}:
        return "GPKG", {}
    if ext in _RASTER_EXT:
        try:
            import rasterio
            with rasterio.open(path) as ds:
                tags = {k.upper(): v for k, v in ds.tags().items()}
        except Exception:  # noqa: BLE001
            tags = {}
        tm = tags.get("GDS_TM_PAYLOAD")
        dem = tags.get("GDS_DEM_WATERMARK")
        if tm:
            return "RASTER_TRUSTMARK", {"tag": tm, "tags": tags}
        if dem:
            return "RASTER_DEM", {"tag": dem, "tags": tags}
        return "RASTER_UNKNOWN", {"tags": tags}
    # 目录 -> 当作 OSGB 数据集
    if path.is_dir():
        return "OSGB", {}
    return "UNKNOWN", {}


def _capability_text(kind: str) -> str:
    return {
        "RASTER_TRUSTMARK": ("TrustMark 256px 载体 + SIFT 配准还原；"
                             "抗任意角度旋转、缩放 0.5–2.0x、裁剪 10–25%（实测 12/12）"),
        "RASTER_DEM": ("整数值末位 LSB + SIFT 配准还原；"
                       "旋转与放大可还原，缩小超过 30% 因重采样抹掉 LSB 无法还原（实测 7/10，"
                       "0.5x/0.75x 失败）"),
        "GPKG": ("属性字段 _wm，逐要素唯一；几何任意变换不影响，"
                 "字段被删除则失效（实测 5/5 几何变换，删除字段 0%）"),
        "OSGB": ("内嵌 JPEG 纹理 DCT 系数；只改节点矩阵/重编码 JPEG 仍可提取，"
                 "纹理转 PNG 或缩放 0.5x 失效（实测 4/6）"),
    }.get(kind, "未知通道")


def check_any(path: Path, expected: int, reference: Path | None, work_dir: Path) -> dict:
    kind, info = classify(path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    base = {"data_class": kind, "channel": kind, "capability": _capability_text(kind),
            "expected": expected}

    if kind == "OSGB":
        return {**base, **_check_osgb(path, expected)}
    if kind == "OSGB_TEXTURE":
        # 单张图片可能是两种载体，按顺序都试：
        #   ① OSGB 外置纹理（JPEG DCT，嵌入方式是 JPEG_DCT_CPU_PER_MEMBER）
        #   ② 像素域 TrustMark（PNG 无损转存后 DCT 系数会没，但 TrustMark 载体还在）
        # 先 DCT 再 TrustMark：纹理成果数量更多，且 DCT 判定是毫秒级。
        r = _check_osgb(path, expected)
        if r.get("found"):
            return {**base, **r}
        r2 = _check_raster_trustmark(path, expected, reference, work_dir)
        if r2.get("found"):
            return {**base, "data_class": "RASTER_TRUSTMARK", "channel": "RASTER_TRUSTMARK",
                    "capability": _capability_text("RASTER_TRUSTMARK"), **r2}
        return {**base, **r, "detail": {"texture": r.get("detail"), "trustmark": r2.get("detail")}}
    if kind == "GPKG":
        return {**base, **_check_gpkg(path, expected)}
    if kind in ("RASTER_TRUSTMARK", "RASTER_UNKNOWN"):
        r = _check_raster_trustmark(path, expected, reference, work_dir)
        if r.get("found") or kind == "RASTER_TRUSTMARK":
            return {**base, **r}
        # 标签缺失且 TrustMark 失败，再试 DEM 通道
        r2 = _check_raster_dem(path, expected, reference, work_dir)
        if r2.get("found"):
            return {**base, "data_class": "RASTER_DEM", "channel": "RASTER_DEM",
                    "capability": _capability_text("RASTER_DEM"), **r2}
        # 两条通道都没命中时，若 TrustMark 走的是"无参考图，探针自身网格直解"的有界判定，
        # 就把它的覆盖率结论返回，而不是压成一句"均未检出"：
        # 前端要能区分"确实没有水印"和"缺参考图所以判不了"，并拿到 coverage。
        if r.get("via") == "direct_self":
            return {**base, **r}
        return {**base, **r2, "detail": "TrustMark 与 DEM 通道均未检出"}
    if kind == "RASTER_DEM":
        return {**base, **_check_raster_dem(path, expected, reference, work_dir)}
    return {**base, "found": False,
            "conclusion": f"不支持的文件类型：{path.suffix or '目录'}"}


# ---------------- 各通道实现 ----------------

def _sandbox_reachable(path: Path) -> bool:
    """判断路径对沙箱账户（gds_sbx）是否可读。

    TrustMark 通道在受限账户 gds_sbx 内执行，它看不到管理员 home 下的临时目录
    （实测 C:\\Users\\Administrator\\... 传入后返回 ``unreadable_input``）。
    所以 DET 通道只对工作区（V:/E:）内的文件有效；其他路径必须提前说明，
    否则用户会拿到一个含糊的"未检出"。
    """
    try:
        import os
        roots = [os.environ.get("GDS_SANDBOX_ROOT", ""), "V:\\geo_data_security",
                 "E:\\geo_data_security"]
        p = str(path.resolve()).lower()
        for r in roots:
            if r and p.startswith(r.lower().rstrip("\\")):
                return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _check_raster_trustmark(path: Path, expected: int, reference: Path | None, work: Path) -> dict:
    """TrustMark 通道必须走 trustmark_engine 的子进程，不能直接 import。

    `trustmark` 包只装在 .trustmark-deploy-venv 里，服务主进程跑在 .deploy-venv，
    直接 ``from . import trustmark_geotiff_runner`` 会 ModuleNotFoundError（实测 500）。
    engine 通过 GDS_TRUSTMARK_PYTHON 指向正确的解释器，并用 taskkill /F /T 管进程树。
    """
    from .trustmark_engine import TrustMarkGeoTIFFEngine, TrustMarkEngineUnavailableError

    engine_ok = True
    engine_err = ""
    # 构造函数本身即做可用性探测（python 能否启动、runner 是否存在），
    # 没有单独的 check_available()。
    try:
        engine = TrustMarkGeoTIFFEngine()
    except TrustMarkEngineUnavailableError as e:
        engine_ok = False
        engine_err = str(e)[:100]
        engine = None

    direct_r: dict | None = None
    if engine_ok and engine is not None:
        # 先做直解（全量逐载体，engine 内部按 GDS_TRUSTMARK_CARRIER_TIMEOUT 控时）
        try:
            r = engine.extract_carriers(path, expected, workers=None)
            direct_r = r
            if r.get("matched") or (r.get("detected") and not r.get("fast_fail")):
                return {"found": True, "decoded": r.get("recorded") or f"{expected:08X}",
                        "via": "direct",
                        "detail": {"coverage": r.get("coverage"),
                                   "meets_bar": r.get("coverage_meets_bar"),
                                   "carriers": r.get("carriers_checked"),
                                   "matched_carriers": r.get("matched")},
                        "conclusion": "检出水印：该栅格为本系统加过水印的成果"}
        except Exception:  # noqa: BLE001
            pass

    # 需要几何还原：必须走 engine 子进程（torch 只在 trustmark venv 里）
    ref = reference or path
    # 没有独立参考图时，ref 就是探针自己 —— 自己配准自己没有意义，SIFT 会在裁剪/缩放
    # 文件上反复枚举倍率与相位，实测 2MB 的裁剪图跑了 17 分钟还不返回。
    # 这类情况改走"探针自身坐标系的 TrustMark 直解 + 拟合扫描"，有界返回。
    if reference is None:
        if direct_r is not None or (engine_ok and engine is not None):
            try:
                r = direct_r
                if r is None:
                    r = engine.extract_carriers(path, expected, workers=None)
                detected = bool(r.get("matched") or r.get("recorded"))
                return {"found": detected,
                        "decoded": r.get("recorded") or (f"{expected:08X}" if detected else None),
                        "via": "direct_self",
                        "detail": {"coverage": r.get("coverage"),
                                   "meets_bar": r.get("coverage_meets_bar"),
                                   "carriers": r.get("carriers_checked"),
                                   "matched_carriers": r.get("matched"),
                                   "recorded_payload": r.get("recorded"),
                                   "note": "未提供参考图，按探针自身 256px 载体网格直解；"
                                           "偏移裁剪/缩放需传 reference_path 才能配准还原"},
                        "conclusion": ("检出水印：该栅格为本系统加过水印的成果" if detected else
                                       "未检出登记水印：未提供参考图，本判定只覆盖未偏移的载体网格；"
                                       "若数据经过裁剪或缩放，请用 reference_path 指定登记原图后重试")}
            except Exception:  # noqa: BLE001
                pass
    if engine_ok and engine is not None:
        if not _sandbox_reachable(path):
            return {"found": False, "decoded": None, "via": "trustmark_skipped",
                    "detail": {"error": "文件不在沙箱可读的工作区内（需 V:/E: 下的路径）",
                               "path": str(path)},
                    "conclusion": ("TrustMark 通道在受限账户 gds_sbx 内执行，"
                                   "只能读取 V:/E: 工作区；请把待检文件放到工作区后重试")}
        try:
            rr = engine._call(["geo-restore", str(ref), str(path), f"{expected:08X}"], timeout=1800)
            if rr.get("found"):
                return {"found": True, "decoded": f"{expected:08X}",
                        "via": "trustmark:" + str(rr.get("via")),
                        "detail": {"rate": rr.get("rate"), "hits": rr.get("hits"),
                                   "total": rr.get("total"),
                                   "geometry": rr.get("geometry") or {}},
                        "conclusion": "经几何变换后仍检出水印，判定为本系统流出"}
            return {"found": False, "decoded": rr.get("decoded"), "via": "trustmark",
                    "detail": {"rate": rr.get("rate"), "hits": rr.get("hits"),
                               "total": rr.get("total"), "engine_available": True},
                    "conclusion": "未检出登记水印：无法证明该数据由本系统流出"}
        except Exception as e:  # noqa: BLE001
            return {"found": False, "decoded": None, "via": "trustmark",
                    "detail": {"error": f"{type(e).__name__}: {str(e)[:100]}",
                               "engine_available": True},
                    "conclusion": "还原通道异常，未能判定"}

    return {"found": False, "decoded": None, "via": "trustmark_unavailable",
            "detail": {"error": engine_err or "trustmark engine unavailable"},
            "conclusion": "TrustMark 引擎不可用，未能判定（检查 GDS_TRUSTMARK_PYTHON）"}


def _check_raster_dem(path: Path, expected: int, reference: Path | None, work: Path) -> dict:
    from . import dem_stream as ds_mod
    from . import dem_block_channel as dbc

    # ① LSB 直解（原尺寸 / 平移 / 旋转 / 放大）
    try:
        r = ds_mod.cross_tile_extract(path, expected)
        if r.get("matched"):
            return {"found": True, "decoded": r.get("decoded"), "via": "dem_lsb_direct",
                    "detail": {"votes": r.get("votes_for_best"),
                               "groups": r.get("groups_checked")},
                    "conclusion": "检出水印：该 DEM 为本系统加过水印的成果"}
    except Exception as e:  # noqa: BLE001
        return {"found": False, "decoded": None, "via": "dem_lsb",
                "detail": {"error": f"{type(e).__name__}: {str(e)[:80]}"},
                "conclusion": "DEM 通道异常，未能判定"}

    # ② 块均值通道：抗 0.5×/0.75× 及各种插值的缩小（LSB 会被平均抹掉，块均值不会）
    bm_error = None
    try:
        from . import dem_block_channel as dbc
        bm = dbc.block_mean_extract(path, expected)
        if bm.get("matched"):
            return {"found": True, "decoded": f"{expected:08X}", "via": "dem_block_mean",
                    "detail": {"hit": bm.get("hit"), "iterations": bm.get("iterations")},
                    "conclusion": "检出水印：该 DEM 为本系统加过水印的成果（块均值通道，抗重采样缩放）"}
    except Exception as e:  # noqa: BLE001
        bm_error = f"{type(e).__name__}: {str(e)[:80]}"

    # ③ 需要还原：仅支持旋转/放大（缩小会抹掉 LSB）
    if reference is None:
        return {"found": False, "decoded": None, "via": "dem_lsb",
                "detail": {"reason": "无参考图，且 LSB 直解与块均值通道均未命中",
                           "block_mean_error": bm_error},
                "conclusion": "未检出登记水印（缩小场景需参考图且超过能力边界）"}
    try:
        from . import dem_geo_restore as dgr
        rr = dgr.restore_and_decode(reference, path, expected, work)
        return {**rr, "via": "dem_lsb+sift"}
    except Exception as e:  # noqa: BLE001
        return {"found": False, "decoded": None, "via": "dem_lsb+sift",
                "detail": {"error": f"{type(e).__name__}: {str(e)[:80]}"},
                "conclusion": "DEM 还原通道异常，未能判定"}


def _gpkg_metadata_carrier(path: Path) -> str | None:
    """读 gpkg_metadata 里的副载体。返回可读标记或 None。

    副载体是加密文本，本模块拿不到解密密钥（密钥只在 worker 侧）。因此这里只做
    **存在性**判定：能读到 gds.watermark.v1 这一行就说明文件出自本平台流水线。
    逐要素 _wm 与副载体同为加密值，判定口径一致——都只能证「有/无」。
    """
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception:  # noqa: BLE001
        return None
    try:
        rows = con.execute(
            "SELECT metadata FROM gpkg_metadata WHERE md_standard_uri = 'gds.watermark.v1'"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return None
    finally:
        con.close()
    for (token,) in rows:
        if token:
            return f"gpkg_metadata 副载体存在（{len(token)} 字节密文）"
    return None


def _check_gpkg(path: Path, expected: int) -> dict:
    """GPKG: _wm 是加密文本，不是数字载荷。只能判定「有/无」，不能比对具体值。"""
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except Exception as e:  # noqa: BLE001
        return {"found": False, "decoded": None,
                "detail": {"error": f"{type(e).__name__}: {str(e)[:80]}"},
                "conclusion": "无法打开 GPKG"}

    try:
        tables = [r[0] for r in con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchall()]
        if not tables:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'gpkg_%'").fetchall()]

        total = with_wm = 0
        per_table = {}
        for t in tables:
            try:
                cols = [r[1] for r in con.execute(f"PRAGMA table_info('{t}')").fetchall()]
                if "_wm" not in cols:
                    per_table[t] = {"features": 0, "with_wm": 0, "has_column": False}
                    continue
                n = con.execute(f"SELECT COUNT(*) FROM '{t}'").fetchone()[0]
                w = con.execute(
                    f"SELECT COUNT(*) FROM '{t}' WHERE _wm IS NOT NULL AND _wm<>''").fetchone()[0]
                per_table[t] = {"features": n, "with_wm": w, "has_column": True}
                total += n
                with_wm += w
            except Exception:  # noqa: BLE001
                continue
    finally:
        con.close()

    rate = round(with_wm / total * 100, 2) if total else 0.0
    # 副载体兜底：_wm 列被删时，gpkg_metadata 里的标记仍在（见 geopackage_watermark）。
    carrier = _gpkg_metadata_carrier(path)
    found = bool(with_wm > 0 or carrier)
    if with_wm > 0:
        decoded = f"_wm 覆盖 {with_wm}/{total}"
    elif carrier:
        decoded = "_wm 列已删除，由 gpkg_metadata 副载体检出"
    else:
        decoded = None
    return {
        "found": found,
        "decoded": decoded,
        "detail": {"features": total, "with_wm": with_wm, "rate": rate,
                   "tables": per_table,
                   "metadata_carrier": carrier},
        "conclusion": (
            "检出逐要素水印字段 _wm：数据来自本系统（几何变换不影响）"
            if with_wm > 0 else
            "逐要素 _wm 字段已被删除，但检出 gpkg_metadata 元数据副载体：数据仍来自本系统"
            if carrier else
            "未检出 _wm 字段，也未检出元数据副载体：无法证明该数据由本系统流出"
        ),
    }


def _check_osgb(path: Path, expected: int) -> dict:
    from . import osgb_dct_watermark as O
    from .osgb_texture_watermark import _image_spans

    def find_textures(data: bytes):
        """内嵌纹理的字节块。JPEG 与 PNG 都要扫：纹理转 PNG 后仅认 JPEG 签名会漏掉整棵模型。"""
        return [data[s:e] for s, e in _image_spans(data)]

    files = [path] if path.is_file() and path.suffix.lower() in {".osgb", ".osg"} else (
        [] if path.is_file() else sorted(path.rglob("*.osgb")))
    # 外置纹理与内嵌纹理是同一套水印：嵌入时按成员逐张 JPEG 处理（JPEG_DCT_CPU_PER_MEMBER），
    # 很多成果的 .osgb 只是几十字节的索引，真水印在同目录的 .jpg/.png 里。
    # 只扫 .osgb 内部会把这类成果判成"未检出"（实测 model.osgb 1832 字节、0 纹理块，
    # 而 texture.jpg 直解 detected=True）。
    base_dir = path if path.is_dir() else path.parent
    textures = sorted(
        item for item in base_dir.rglob("*")
        if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files and not textures:
        return {"found": False, "decoded": None, "detail": {"reason": "无 .osgb 文件"},
                "conclusion": "目录内未找到 OSGB 文件"}

    checked = hit = too_small = rescaled = external_hit = 0
    for f in files[:200]:
        try:
            for blob in find_textures(f.read_bytes()):
                checked += 1
                r = O.extract_jpeg(blob, expected)
                if not r.get("detected"):
                    # 纹理被缩放过时块网格错位，直提必失败（不是信息丢了）：
                    # 实测把缩小纹理放大回原尺寸即可解出，按倍率枚举重试。
                    r = O.extract_jpeg(blob, expected, scale_search=True)
                    if r.get("detected"):
                        rescaled += 1
                if r.get("detected"):
                    hit += 1
                elif r.get("reason") == "too_small":
                    too_small += 1
        except Exception:  # noqa: BLE001
            continue

    for f in textures[:200]:
        checked += 1
        try:
            blob = f.read_bytes()
            r = O.extract_jpeg(blob, expected)
            if not r.get("detected"):
                r = O.extract_jpeg(blob, expected, scale_search=True)
                if r.get("detected"):
                    rescaled += 1
            if r.get("detected"):
                hit += 1
                external_hit += 1
            elif r.get("reason") == "too_small":
                too_small += 1
        except Exception:  # noqa: BLE001
            continue

    return {
        "found": hit > 0,
        "decoded": f"{hit}/{checked} 纹理命中" if hit else None,
        "detail": {"files_scanned": min(len(files), 200),
                   "external_textures": min(len(textures), 200),
                   "external_hit": external_hit,
                   "textures_checked": checked,
                   "hit": hit, "too_small": too_small, "rescaled_hit": rescaled},
        "conclusion": (
            "检出纹理水印：数据来自本系统（单纯改模型姿态/重编码不影响）"
            if hit else
            "未检出纹理水印：可能是纹理被转为 PNG 或缩小，超出该类通道能力"
        ),
    }
