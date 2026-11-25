from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .geojson_watermark import extract_geojson_watermark
from .vector_watermark import extract_shapefile_watermark
from . import sandbox_compute


def _row(ok: bool, **details: Any) -> dict[str, Any]:
    return {"verified": bool(ok), **details}


def verify_shp(output: Path, expected_text: str, key: bytes) -> dict[str, Any]:
    output = Path(output)
    if output.is_file() and output.suffix.lower() == ".shp":
        layers = [output]
    else:
        layers = sorted(output.glob("*.shp"))
    results = []
    for layer in layers:
        if sandbox_compute.enabled():
            values = sandbox_compute.run("shp_extract", str(layer))
        else:
            values = extract_shapefile_watermark(layer, key)
        results.append({"file": layer.name, "total": len(values), "matched": sum(v == expected_text for v in values)})
    total = sum(row["total"] for row in results)
    matched = sum(row["matched"] for row in results)
    return _row(bool(layers) and matched == total, method="shp_attribute_extract", samples=total, matched=matched, files=results)


def _dominant(values: list[str]) -> str:
    from collections import Counter
    counter = Counter(value for value in values if value)
    return counter.most_common(1)[0][0] if counter else ""


def verify_gpkg(output: Path, expected_text: str, key: bytes) -> dict[str, Any]:
    from .geopackage_watermark import extract_geopackage_watermark
    if sandbox_compute.enabled():
        result = sandbox_compute.run("gpkg_extract", str(output))
    else:
        result = extract_geopackage_watermark(Path(output), key)
    values = result.get("values", [])
    if expected_text:
        matched = sum(value == expected_text for value in values)
        ok = bool(values) and matched == len(values)
    else:
        # 调用方没登记该带什么水印时只做"检出"：只要解出的值非空且全一致，就说明
        # 这份数据确实带我们的水印，具体是哪一个由 watermark_text 回报。
        nonempty = [value for value in values if value]
        matched = len(nonempty)
        ok = bool(nonempty) and len(set(nonempty)) == 1
    row = _row(ok, method="gpkg_attribute_extract", samples=len(values), matched=matched,
               layers=result.get("layers", []))
    row["watermark_text"] = _dominant(values) if not expected_text else expected_text
    return row


def verify_geojson(output: Path, expected_text: str, key: bytes) -> dict[str, Any]:
    if sandbox_compute.enabled():
        values = sandbox_compute.run("geojson_extract", str(output))
    else:
        values = extract_geojson_watermark(Path(output), key)
    if expected_text:
        matched = sum(value == expected_text for value in values)
        ok = bool(values) and matched == len(values)
    else:
        nonempty = [value for value in values if value]
        matched = len(nonempty)
        ok = bool(nonempty) and len(set(nonempty)) == 1
    row = _row(ok, method="geojson_attribute_extract", samples=len(values), matched=matched)
    row["watermark_text"] = _dominant(values) if not expected_text else expected_text
    return row


def _has_georeference(path: Path) -> tuple[bool, dict]:
    """判断栅格是否有地理参考。无参考时 TrustMark 会白跑全部搜索窗口。

    只读头部，不加载像素，毫秒级返回。rasterio 不可用时按「有参考」处理，
    不让校验本身成为新的失败点。
    """
    try:
        import rasterio  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return True, {"checked": False, "why": "rasterio unavailable"}
    try:
        with rasterio.open(path) as ds:
            transform = ds.transform
            has_gcp = bool(getattr(ds, "gcps", None) and ds.gcps[0])
            # 无地理参考时 rasterio 返回单位矩阵。注意 GDAL 的 Affine
            # 按 (a,b,c,d,e,f) 存储，e 在「无参考」时实测为 +1.0，
            # 不是数学上的 -1（早先按 -1 判断会漏判，导致仍跑 118 s）。
            identity = (transform.a == 1 and transform.b == 0 and
                        transform.c == 0 and transform.d == 0 and
                        transform.e == 1 and transform.f == 0)
            detail = {"transform": list(transform)[:6], "has_gcp": has_gcp,
                      "identity": identity, "size": [ds.width, ds.height]}
            return (not identity) or has_gcp, detail
    except Exception as exc:  # noqa: BLE001
        return True, {"checked": False, "why": f"{type(exc).__name__}: {exc}"}


def verify_geotiff(output: Path, expected: int, engine) -> dict[str, Any]:
    if engine is not None:
        # 前置校验：没有地理参考的影像在 TrustMark 里会走完全部搜索窗口
        # （实测 241 s 后才以 exit 143 失败）。沙箱 detect 是交互式调用，
        # 这种等待没有意义，直接给出明确结论。
        geo_ok, geo_detail = _has_georeference(Path(output))
        if not geo_ok:
            return _row(False, method="geotiff_no_georeference", samples=0, matched=0,
                        reason=("影像缺少地理参考（geotransform/GCP/RPC），"
                                "无法按瓦片网格定位载体；这是数据的真实状态，不是接口故障"),
                        detail=geo_detail)
        # Whole-image per-carrier verification: decode every carrier the product was built with
        # (origins come from its own tags, so no search is needed).
        if hasattr(engine, "extract_carriers"):
            try:
                coverage = engine.extract_carriers(Path(output), expected)
            except Exception as exc:  # noqa: BLE001 - fall back rather than fail the whole check
                coverage = {"error": f"{type(exc).__name__}: {exc}"}
            samples = int(coverage.get("samples", 0) or 0)
            if samples:
                matched = int(coverage.get("matched", 0) or 0)
                if matched:
                    return _row(True, method=coverage.get("method", "per_carrier_verify"),
                                samples=samples, matched=matched,
                                not_watermarkable=int(coverage.get("not_watermarkable", 0) or 0),
                                carrier_coverage=coverage.get("coverage"),
                                coverage_bar=coverage.get("coverage_bar"),
                                coverage_meets_bar=coverage.get("coverage_meets_bar"),
                                extractor=coverage)
                # Per-carrier replay found the lattice but decoded nothing. That is exactly what a
                # crop, a resize or a reprojection produces: the recorded origins no longer point
                # at complete carriers, so replaying them cannot succeed. Falling straight through
                # to a false here made the HTTP detection API disagree with the acceptance matrix -
                # measured on a 25 % crop and on an EPSG:4326 reprojection, both of which the
                # runner's search paths decode (phase_sweep_window / anisotropic_tile_window) while
                # this path returned verified=false. Retry with the search-capable extractor and
                # keep the per-carrier row for traceability.
                searched = engine.extract(Path(output), expected)
                row = _row(bool(searched.get("detected")),
                           method=searched.get("method", "trustmark"), samples=1,
                           matched=int(bool(searched.get("detected"))), extractor=searched)
                row["carrier_replay"] = coverage
                row["fallback"] = "per_carrier_replay_failed_search_used"
                return row
        result = engine.extract(Path(output), expected)
        return _row(bool(result.get("detected")), method=result.get("method", "trustmark"), samples=1, matched=int(bool(result.get("detected"))), extractor=result)
    from .geotiff_watermark import extract_geotiff_watermark
    if sandbox_compute.enabled():
        value = sandbox_compute.run("geotiff_robust_extract", str(output))
    else:
        value = extract_geotiff_watermark(Path(output))
    return _row(value == expected, method="geotiff_extract", samples=1, matched=int(value == expected), extracted=value)


def verify_textures(output: Path, expected: int, engine) -> dict[str, Any]:
    files = sorted(path for path in Path(output).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    for path in files:
        try:
            if engine is not None and hasattr(engine, "extract_image"):
                result = engine.extract_image(path, expected)
                result["method"] = "trustmark_texture_extract"
            elif sandbox_compute.enabled():
                result = sandbox_compute.run("texture_extract", str(path), f"{expected:08X}")
                result["method"] = "texture_adaptive_jpeg_extract"
            else:
                from .texture_watermark import extract_texture_watermark_adaptive_jpeg
                result = extract_texture_watermark_adaptive_jpeg(path, expected)
                result["method"] = "texture_adaptive_jpeg_extract"
        except Exception as exc:
            result = {"detected": False, "error": f"{type(exc).__name__}: {exc}"}
        if not result.get("detected") and path.suffix.lower() in {".jpg", ".jpeg"}:
            # 纹理有两套方案：adaptive（老）与 JPEG-DCT（OSGB 逐成员用）。成果由哪套写入
            # 取决于处理路径，验收时只试一套会把正确成果判成未检出，因此补一次 DCT 复核。
            try:
                from .osgb_dct_watermark import extract_jpeg as _dct_extract
                dct = _dct_extract(path.read_bytes(), expected)
                if dct.get("detected"):
                    dct["method"] = "osgb_dct_extract"
                    result = dct
            except Exception:
                pass
        rows.append({"file": path.name, "verified": bool(result.get("detected")), "extractor": result})
    matched = sum(row["verified"] for row in rows)
    return _row(bool(files) and matched == len(files), method="trustmark_texture_extract", samples=len(files), matched=matched, files=rows)


def _verify_osgb_file(args) -> dict:
    """复校单个 .osgb（模块级函数，供多进程池调用）。"""
    path_str, output_str, expected, verifying_subset = args
    from .osgb_texture_watermark import _jpeg_spans, extract_osgb_jpeg_watermark
    from .texture_watermark import extract_texture_watermark_adaptive_bytes

    path = Path(path_str)
    data = path.read_bytes()
    spans = list(_jpeg_spans(data))
    pixel_verified = 0
    metadata_matched = 0
    not_watermarkable = 0
    for start, end in spans:
        payload = data[start:end]
        if extract_osgb_jpeg_watermark(payload, expected).get("detected"):
            # COM 只算可溯源元数据，按口径不计入像素域成绩。
            metadata_matched += 1
            not_watermarkable += 1
            continue
        try:
            if extract_texture_watermark_adaptive_bytes(payload, expected).get("detected"):
                pixel_verified += 1
        except Exception:  # noqa: BLE001
            # Keep the legacy full-result sampling behavior. For a partial OSGB result,
            # an unexpected extraction failure in a supposedly supported tile must fail.
            if verifying_subset:
                raise
            not_watermarkable += 1
    rel = str(path.relative_to(Path(output_str))) if output_str else path.name
    row = {"file": rel, "embedded_jpegs": len(spans),
           "pixel_verified": pixel_verified, "not_watermarkable": not_watermarkable}
    if verifying_subset:
        row["metadata_matched"] = metadata_matched
    return row


def verify_osgb(output: Path, expected: int, *, excluded_paths: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Verify OSGB carriers, separating pixel watermarking from small-texture fallback."""
    if sandbox_compute.enabled():
        if excluded_paths:
            raise ValueError("sandbox OSGB verification does not support verifying a subset of tiles")
        result = sandbox_compute.run("osgb_verify", str(output), f"{expected:08X}", timeout_s=1800)
        samples = int(result.get("pixel_samples", 0))
        matched = int(result.get("pixel_matched", 0))
        fallback = int(result.get("not_watermarkable", 0))
        return _row((samples > 0 and matched == samples) or (samples == 0 and fallback > 0),
                    method="sandbox_osgb_pixel_extract_with_small_texture_fallback",
                    samples=samples, matched=matched,
                    not_watermarkable=fallback,
                    verification_quality="PIXEL_ONLY" if fallback == 0 else "PIXEL_PLUS_SMALL_TEXTURE_FALLBACK",
                    files_processed=int(result.get("files_processed", 0)), files=result.get("files", []),
                    runner_user="gds_sbx")
    import os
    output = Path(output)
    files = sorted(output.rglob("*.osgb"))
    if excluded_paths:
        paths = {file.relative_to(output).as_posix(): file for file in files}
        if not excluded_paths <= paths.keys():
            raise ValueError("OSGB processing report names a skipped tile that is missing from the result")
        files = [file for name, file in paths.items() if name not in excluded_paths]
    # 抽样上限：全量逐块复校是 O(文件数 × 纹理块)，单块 1024² 纹理解码实测 520 ms，
    # 2.8 万个文件串行要 4 小时，直接把「10 GB 倾斜模型 ≤30 分钟」的指标打爆。
    # 复校目的是确认水印确实写进像素域，等间隔抽样覆盖整棵目录足够。
    # 需要全量口径时设 GDS_OSGB_VERIFY_FILES=0。
    try:
        cap = int(os.environ.get("GDS_OSGB_VERIFY_FILES", "1200") or 1200)
    except ValueError:
        cap = 1200
    total_files = len(files)
    if cap > 0 and total_files > cap:
        step = total_files / cap
        files = [files[int(i * step)] for i in range(cap)]

    try:
        workers = max(1, int(os.environ.get("GDS_OSGB_VERIFY_WORKERS", "8") or 8))
    except ValueError:
        workers = 8

    jobs = [(str(p), str(output), expected, bool(excluded_paths)) for p in files]
    rows: list[dict] = []
    if jobs:
        # 必须用进程池：逐块解码走 scipy.fft.dct，不释放 GIL，线程池实测只有 0.8 核。
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(workers, len(jobs))) as pool:
                rows = list(pool.map(_verify_osgb_file, jobs, chunksize=8))
        except Exception:  # noqa: BLE001
            # 进程池起不来（受限环境/无 spawn 权限）时退回串行，宁可慢也不能让任务失败。
            rows = [_verify_osgb_file(job) for job in jobs]

    samples = sum(row["embedded_jpegs"] - row["not_watermarkable"] for row in rows)
    matched = sum(row["pixel_verified"] for row in rows)
    fallback = sum(row["not_watermarkable"] for row in rows)
    verification = _row(
        bool(rows) and ((samples > 0 and matched == samples) or (samples == 0 and fallback > 0)),
        method="osgb_pixel_carrier_extract_with_small_texture_fallback",
        samples=samples, matched=matched, not_watermarkable=fallback,
        verification_quality="PIXEL_ONLY" if fallback == 0 else "PIXEL_PLUS_SMALL_TEXTURE_FALLBACK",
        sampled_files=len(rows), total_files=total_files,
        sampling="full" if len(rows) == total_files else f"every {total_files/len(rows):.1f} files",
        files=rows[:20], files_processed=len(rows),
    )
    if excluded_paths:
        metadata_matched = sum(row["metadata_matched"] for row in rows)
        verification["metadata_matched"] = metadata_matched
        verification["excluded_files"] = len(excluded_paths)
        verification["verified"] = verification["verified"] and (matched > 0 or metadata_matched > 0)
    return verification


def verify_dem(output: Path, expected: int) -> dict[str, Any]:
    """Re-extract a streamed DEM payload from the produced elevation raster(s).

    Mirrors the task-side DEM verification (`process_dem` -> dem_stream_extract) so the
    read-only watermark query answers with the same evidence shape as the other classes.
    """
    from .dem_stream import stream_extract

    output = Path(output)
    inputs = [output] if output.is_file() else sorted(
        item for item in output.rglob("*") if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}
    )
    if not inputs:
        raise ValueError("no .tif/.tiff DEM files found in output")
    rows = []
    samples = matched = 0
    for item in inputs:
        verification = (sandbox_compute.run("dem_stream_extract", str(item), f"{expected:08X}")
                        if sandbox_compute.enabled() else stream_extract(item, expected))
        rows.append({"path": str(item), "verification": verification})
        samples += int(verification.get("samples", 0) or 0)
        matched += int(verification.get("matched", 0) or 0)
    return _row(bool(samples) and matched == samples, method="dem_stream_extract",
                samples=samples, matched=matched, files=rows)


def verify_artifact(data_type: str, output: Path, expected: Any, key: bytes, engine=None) -> dict[str, Any]:
    normalized = data_type.upper()
    if normalized == "SHP":
        return verify_shp(Path(output), str(expected), key)
    if normalized in {"GPKG", "GEOPACKAGE"}:
        return verify_gpkg(Path(output), str(expected), key)
    if normalized == "GEOJSON":
        return verify_geojson(Path(output), str(expected), key)
    if normalized in {"GEOTIFF", "GTIFF"}:
        return verify_geotiff(Path(output), int(expected), engine)
    if normalized == "DEM":
        return verify_dem(Path(output), int(expected))
    if normalized in {"TEXTURES", "OSGB_TEXTURES"}:
        return verify_textures(Path(output), int(expected), engine)
    if normalized == "OSGB":
        return verify_osgb(Path(output), int(expected))
    raise ValueError(f"no verifier for {data_type}")
