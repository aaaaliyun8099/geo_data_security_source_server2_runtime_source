from __future__ import annotations

import json
import hashlib
import shutil
import uuid
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .crypto import encrypt_text
from .verification import verify_geotiff
from . import sandbox_compute
from .geojson_watermark import embed_geojson_watermark
from .osgb_texture_watermark import embed_osgb_tree
from .texture_watermark import embed_texture_watermark_adaptive_jpeg
from .vector_watermark import sha256_file
from .verification import verify_artifact
from .classification import Classification, parse_classification
from .trace_chain import LocalTraceChain
from .integrity import source_integrity_warning


# The partial-result verdict needs a floor: `detected` on its own only asserts that at least one
# carrier decoded. 0.50 is a sanity floor (a real partial product measured 0.83), overridable so an
# operator can raise it without touching code. It is deliberately unrelated to
# GDS_TM_CARRIER_MIN_COVERAGE (0.90 by default), which is a mark-uniformity signal that the
# full-image path does not enforce either.
def partial_verification_passed(verification: dict) -> bool:
    if not verification or not verification.get("verified"):
        return False
    measured = verification.get("carrier_coverage")
    if measured is None:
        return True
    floor = float(os.environ.get("GDS_PARTIAL_MIN_CARRIER_COVERAGE", "0.50") or 0.50)
    return float(measured) >= floor


_VECTOR_SUFFIX_TYPE = {".gpkg": "GPKG", ".shp": "SHP", ".geojson": "GEOJSON", ".json": "GEOJSON"}


@contextmanager
def _shp_phase(result_id: str, phase: str):
    started_at = perf_counter()
    outcome = "failed"
    try:
        yield
        outcome = "ok"
    finally:
        print(f"[TIMING] result={result_id} shp_{phase} {perf_counter() - started_at:.3f}s outcome={outcome}", flush=True)


def _payload_carrier_summary(source: Path, limit: int = 6) -> str:
    """列出载荷里实际有哪些文件，用于把「没有 .shp」讲成运维能直接动手的话。"""
    base = Path(source)
    try:
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
    except OSError:
        return "无法读取"
    counts: dict[str, int] = {}
    samples: dict[str, str] = {}
    for item in files:
        suffix = item.suffix.lower() or "<无扩展名>"
        counts[suffix] = counts.get(suffix, 0) + 1
        samples.setdefault(suffix, item.name)
    if not counts:
        return "目录为空（0 个文件）"
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    parts = [f"{suffix}×{count}（{samples[suffix][:40]}）" for suffix, count in ordered[:limit]]
    return "、".join(parts) + ("…" if len(ordered) > limit else "")


def _content_vector_type(source: Path) -> str | None:
    """Return the vector format actually present in the payload, ignoring the declared type.

    S1 maps format from the file extension, so a ZIP that carries a SHP set is declared GPKG
    (BE-DLG); the GPKG handler then fails with "payload contains no .gpkg" and the whole
    distribution fails. Only the DLG family is answered here: mixed or absent vector content
    returns None so the declared type still decides.
    """
    try:
        if source.is_file():
            found = {_VECTOR_SUFFIX_TYPE.get(source.suffix.lower())}
        else:
            found = {_VECTOR_SUFFIX_TYPE.get(item.suffix.lower())
                     for item in source.rglob("*") if item.is_file()}
    except OSError:
        return None
    found.discard(None)
    if "GPKG" in found and "SHP" in found:
        return None
    for candidate in ("SHP", "GPKG", "GEOJSON"):
        if candidate in found:
            return candidate
    return None


class UnsupportedDataTypeError(ValueError):
    pass


@dataclass(frozen=True)
class ProcessingArtifact:
    artifact_id: str
    data_type: str
    source: str
    output: str
    files_processed: int
    status: str


class ProcessingService:
    """统一处理 GeoJSON、GeoTIFF 和纹理目录，结果写入项目结果目录。"""

    def __init__(self, workspace: Path, key: bytes, trustmark_engine=None, trace_path: Path | str | None = None):
        self.workspace = Path(workspace)
        self.key = key
        self.trustmark_engine = trustmark_engine
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.trace_chain = LocalTraceChain(trace_path or (self.workspace / "state" / "server2_trace.jsonl"))

    @staticmethod
    def _validate_project_id(project_id: str) -> str:
        if not project_id or project_id in {".", ".."} or any(char in project_id for char in "/\\"):
            raise ValueError("invalid project id")
        return project_id

    def _artifact_dir(self, project_id: str) -> tuple[str, Path]:
        project_id = self._validate_project_id(project_id)
        artifact_id = str(uuid.uuid4())
        project_root = (self.workspace / "projects" / project_id).resolve()
        workspace_root = self.workspace.resolve()
        if workspace_root != project_root and workspace_root not in project_root.parents:
            raise ValueError("project path escapes workspace")
        output = project_root / "results" / artifact_id
        output.mkdir(parents=True, exist_ok=False)
        return artifact_id, output

    def _failed(self, output_dir: Path) -> None:
        shutil.rmtree(output_dir, ignore_errors=True)

    @staticmethod
    def _sha256_path(path: Path) -> str:
        path = Path(path)
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        if path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                digest.update(str(child.relative_to(path)).replace("\\", "/").encode("utf-8"))
                with child.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            return digest.hexdigest()
        raise FileNotFoundError(path)

    @classmethod
    def _write_manifest(cls, output_dir: Path, project_id: str, artifact_id: str, data_type: str, source: Path, output: Path, count: int, classification: Classification | str | None = None, precision_report: dict | None = None, watermark: int | None = None) -> None:
        manifest = {
            "artifact_id": artifact_id,
            "project_id": project_id,
            "data_type": data_type,
            "source": str(source),
            "output": str(output),
            "files_processed": count,
            "source_hash": cls._sha256_path(source),
            "output_hash": cls._sha256_path(output),
            "status": "PENDING",
        }
        if classification is not None:
            manifest["classification"] = str(getattr(classification, "value", classification))
        if watermark is not None:
            # 水印号必须写进清单：提取时要用它做 payload 校验，不是它就会满幅 0 命中
            # （实测 2026-11-12：拿 0x20261111 去测 0x20261112 的成果，128 载体全灭）。
            # 同时写十六进制字符串，前端与运维不用自己做进制换算。
            manifest["expected_watermark"] = int(watermark)
            manifest["expected_watermark_hex"] = f"0x{int(watermark):08X}"
        if precision_report is not None:
            manifest["precision_reduction"] = {"applied": True, **precision_report}
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _trace_manifest(self, output_dir: Path, event_type: str) -> str:
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = self.trace_chain.append(event_type, artifact_id=manifest["artifact_id"], project_id=manifest["project_id"],
                                         classification=manifest.get("classification", "UNCLASSIFIED"),
                                         input_hash=manifest["source_hash"], output_hash=manifest["output_hash"],
                                         sandbox_user="gds_sbx" if sandbox_compute.enabled() else "direct")
        manifest["trace_record_hash"] = record["record_hash"]
        if record.get("besu_anchor"):
            manifest["besu_anchor"] = record["besu_anchor"]
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return record["record_hash"]

    def _verify_manifest(self, output_dir: Path, data_type: str, output: Path, expected) -> dict:
        try:
            result = verify_artifact(data_type, output, expected, self.key, self.trustmark_engine)
        except Exception as exc:
            result = {"verified": False, "method": "verification_error", "samples": 0, "matched": 0,
                      "error": f"{type(exc).__name__}: {exc}"}
        path = output_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["verification"] = result
        manifest["verification_status"] = "VERIFIED" if result.get("verified") else "VERIFICATION_FAILED"
        manifest["expected_watermark"] = expected
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if not result.get("verified"):
            raise ValueError(f"watermark verification failed: {result}")
        return result

    def process_geojson(self, project_id: str, source: Path, user_id: str, timestamp: str, decimals: int | None = None, classification: Classification | str | None = None, precision_report: dict | None = None) -> ProcessingArtifact:
        from .precision import reduce_geojson_precision

        source = Path(source)
        artifact_id, output_dir = self._artifact_dir(project_id)
        try:
            prepared = source
            # 只处理「独立的 DECIMALS 请求」（无 precision_mode 计划时）；有计划说明降精度已在
            # _apply_precision 阶段做过，避免重复处理。
            if decimals is not None and precision_report is None:
                prepared = output_dir / "precision_reduced.geojson"
                reduce_geojson_precision(source, prepared, decimals)
            destination = output_dir / "watermarked.geojson"
            if sandbox_compute.enabled():
                count = int(sandbox_compute.run(
                    "geojson_embed", str(prepared), str(destination), f"{user_id}|{timestamp}"))
            else:
                count = embed_geojson_watermark(prepared, destination, f"{user_id}|{timestamp}", self.key)
            self._write_manifest(output_dir, project_id, artifact_id, "GeoJSON", source, destination, count, classification, precision_report)
            self._verify_manifest(output_dir, "GeoJSON", destination, f"{user_id}|{timestamp}")
            self._trace_manifest(output_dir, "WATERMARK_VERIFIED")
            return ProcessingArtifact(artifact_id, "GeoJSON", str(source), str(destination), count, "PENDING")
        except Exception:
            self._failed(output_dir)
            raise

    def _apply_visible_layer(self, output_dir: Path, destination: Path, *, text: str,
                            anchor: str = "bottom-right", opacity: int = 150,
                            font_scale: float = 0.045, report_holder: dict | None = None) -> dict:
        """在已嵌好不可见水印的栅格上再叠加可见文字（HYBRID/VISIBLE）。

        就地覆写 ``destination``：先把当前文件读出来叠字后写回，保证可见层是**最后一层**，
        不会破坏刚嵌入的不可见水印（不可见水印在频域，叠加可见文字只改像素灰度）。
        """
        from .visible_watermark import overlay_visible_on_raster, psnr

        staged = destination.with_name(destination.stem + ".visible.tif")
        report = overlay_visible_on_raster(destination, staged, text, anchor=anchor,
                                           opacity=opacity, font_scale=font_scale)
        before, after = report.pop("_arrays")
        report["psnr_db"] = psnr(before, after)
        staged.replace(destination)
        if report_holder is not None:
            report_holder.update(report)
        return report

    def process_geotiff(self, project_id: str, source: Path, watermark: int, classification: Classification | str | None = None, preflight: list[dict] | None = None, precision_report: dict | None = None,
                        watermark_mode: str = "FORENSIC", visible_text: str | None = None) -> ProcessingArtifact:
        from .geotiff_watermark import embed_geotiff_watermark

        source = Path(source)
        artifact_id, output_dir = self._artifact_dir(project_id)
        try:
            destination = output_dir / "watermarked.tif"
            detail = (preflight or [{}])[0]
            if detail.get("source_integrity") == "INCOMPLETE":
                # Accepting an incomplete source is the platform default; GDS_PARTIAL_RASTER_MODE=0
                # turns the readable-region path off and makes the task refuse such a source.
                if os.environ.get("GDS_PARTIAL_RASTER_MODE", "1") != "1":
                    raise ValueError("partial raster mode is disabled")
                partial_result = (self.trustmark_engine.embed_partial(source, destination, watermark)
                                  if self.trustmark_engine is not None
                                  else sandbox_compute.run("partial_raster_embed", str(source), str(destination), f"{watermark:08X}"))
                embedded = int(partial_result.get("carriers_embedded", partial_result.get("blocks_embedded", 0)) or 0)
                lattice = int(partial_result.get("carriers_total", 0) or 0)
                skipped = int(partial_result.get("carriers_skipped", 0) or 0)
                readable = max(1, lattice - skipped)
                coverage = embedded / readable
                # Zero embedded carriers is an ENGINE failure, not a partially readable source:
                # publishing it would hand over an all-zero raster as a manual-review "partial
                # result" (measured on the real 3 GB truncation: 3,689/3,834 blocks died with a
                # NumPy/Torch import-order error and the approved file was zero in every window).
                if embedded <= 0:
                    raise ValueError(
                        "partial raster embed produced no watermarked carrier - the watermark engine "
                        "failed; see the server2 log and the task audit record instead of approving this result"
                    )
                bar = float(os.environ.get("GDS_PARTIAL_MIN_EMBED_COVERAGE", "0.90") or 0.90)
                if coverage < bar:
                    raise ValueError(
                        f"partial raster embed covered only {coverage:.1%} of the readable carriers "
                        f"({embedded}/{readable}) against a {bar:.0%} gate - not publishable as a partial result"
                    )
                count = embedded
                partial = True
            elif self.trustmark_engine is not None:
                try:
                    count = int(self.trustmark_engine.embed(source, destination, watermark)); partial = False
                except Exception as exc:  # noqa: BLE001
                    # 兜底：完整性扫描漏判时（例如源尾部坏块只在严格嵌入阶段才暴露），
                    # 读取失败不该让整条任务失败 —— 退回"可读区域嵌入"，按部分成果交付并标记复核。
                    msg = f"{type(exc).__name__}: {exc}"
                    if not any(k in msg for k in ("RasterioIOError", "Read failed", "ReadFailed",
                                                  "IReadBlock failed", "TIFFReadEncodedTile")):
                        raise
                    if os.environ.get("GDS_PARTIAL_RASTER_MODE", "1") != "1":
                        raise
                    partial_result = self.trustmark_engine.embed_partial(source, destination, watermark)
                    embedded = int(partial_result.get("carriers_embedded",
                                                      partial_result.get("blocks_embedded", 0)) or 0)
                    lattice = int(partial_result.get("carriers_total", 0) or 0)
                    skipped = int(partial_result.get("carriers_skipped", 0) or 0)
                    readable = max(1, lattice - skipped)
                    coverage = embedded / readable
                    if embedded <= 0:
                        raise ValueError(
                            "partial raster embed produced no watermarked carrier after the strict "
                            "embed failed on unreadable source blocks") from exc
                    bar = float(os.environ.get("GDS_PARTIAL_MIN_EMBED_COVERAGE", "0.90") or 0.90)
                    if coverage < bar:
                        raise ValueError(
                            f"partial raster embed covered only {coverage:.1%} of the readable "
                            f"carriers ({embedded}/{readable}) against a {bar:.0%} gate") from exc
                    count = embedded
                    partial = True
                    detail = dict(detail or {})
                    detail.setdefault("source_integrity", "INCOMPLETE")
                    detail["detected_on_embed_retry"] = msg[:300]
            elif sandbox_compute.enabled():
                sandbox_compute.run("geotiff_embed", str(source), str(destination), f"{watermark:08X}")
                count = 1; partial = False
            else:
                embed_geotiff_watermark(source, destination, watermark); count = 1; partial = False
            self._write_manifest(output_dir, project_id, artifact_id, "GeoTIFF", source, destination, count, classification, precision_report, watermark)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            if partial:
                manifest["source_integrity"] = "INCOMPLETE"
                manifest["verification_status"] = "VERIFIED_PARTIAL"
                manifest["approval_policy"] = "MANUAL_REVIEW_REQUIRED"
                manifest["partial_processing"] = {**detail, **partial_result}
                # same warning shape the task and the ingest check expose, so the frontend has one
                # record to render in every view
                # 预检 detail 可能只有宽高（扫描缓存命中时字段不全），此时用 partial_embed
                # 自己数出来的块统计，否则警告会显示成 "0/0 个数据块无法读取（占比未知）"。
                warn_detail = dict(detail or {})
                if not warn_detail.get("blocks_total"):
                    warn_detail["blocks_total"] = partial_result.get("copy_blocks_total")
                    warn_detail["copy_blocks_unreadable"] = partial_result.get("copy_blocks_unreadable")
                if not warn_detail.get("blocks_read") and warn_detail.get("blocks_total"):
                    warn_detail["blocks_read"] = (int(warn_detail["blocks_total"])
                                                  - int(partial_result.get("copy_blocks_unreadable") or 0))
                if warn_detail.get("blocks_total"):
                    warn_detail.setdefault("source_integrity", "INCOMPLETE")
                warning = source_integrity_warning(warn_detail)
                if warning:
                    warning.update({
                        "carriers_total": partial_result.get("carriers_total"),
                        "carriers_embedded": partial_result.get("carriers_embedded"),
                        "carriers_skipped": partial_result.get("carriers_skipped"),
                    })
                    manifest["warnings"] = [warning]
                verification = verify_geotiff(destination, watermark, self.trustmark_engine)
                manifest["verification"] = verification
                # A partial product is "verified partial" when the mark re-extracts from the carriers
                # that were actually embedded. `detected` (at least one carrier decoded) is the same
                # product-level criterion the full-image path uses, so it is not enough on its own -
                # a product whose mark survived in a handful of carriers must not read as verified.
                # The extra check is a sanity floor on the measured per-carrier coverage, and it is
                # deliberately NOT the project-wide `coverage_meets_bar` (default 0.90): that number
                # is a mark-uniformity signal that the full-image path never enforces, and real
                # 27904x36352 orthos measure 0.82-0.83 through the untouched production embed() too.
                manifest["verification_status"] = (
                    "VERIFIED_PARTIAL" if partial_verification_passed(verification) else "VERIFICATION_FAILED"
                )
                (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                self._verify_manifest(output_dir, "GeoTIFF", destination, watermark)
            # 可见水印是最后一层：不可见水印（若有）已写盘并校验通过，再叠字不会破坏它。
            # 注意：`_verify_manifest` 会自己落盘 manifest，所以这里必须【重新读取】,
            # 否则会用验证前读到的旧副本覆盖掉 verification 字段。
            mode = (watermark_mode or "FORENSIC").upper()
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            if mode in {"VISIBLE", "HYBRID"}:
                visible_report = {}
                self._apply_visible_layer(
                    output_dir, destination,
                    text=visible_text or f"GDS-{watermark:08X}",
                    report_holder=visible_report,
                )
                manifest["watermark_mode"] = mode
                manifest["visible_watermark"] = visible_report
                # 可见水印改动了像素，输出哈希必须重算，否则下载校验会失败
                manifest["output_hash"] = self._sha256_path(destination)
            else:
                manifest["watermark_mode"] = "FORENSIC"
            (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if mode in {"VISIBLE", "HYBRID"}:
                # 可见层改动像素后重新抽取不可见水印，证明 HYBRID 的两层同时存活
                verify_after = verify_geotiff(destination, watermark, self.trustmark_engine)
                manifest["verification_after_visible"] = verify_after
                manifest["visible_layer_preserves_invisible"] = bool(
                    verify_after.get("detected") or verify_after.get("verified")
                ) if mode == "HYBRID" else None
                (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                if mode == "HYBRID" and not manifest["visible_layer_preserves_invisible"]:
                    raise ValueError(
                        "HYBRID watermark: the invisible mark no longer extracts after the visible "
                        "overlay was applied - refusing to publish a hybrid result"
                    )
            self._trace_manifest(output_dir, "WATERMARK_PARTIAL" if partial else "WATERMARK_VERIFIED")
            return ProcessingArtifact(artifact_id, "GeoTIFF", str(source), str(destination), count, "PENDING")
        except Exception:
            self._failed(output_dir)
            raise

    def process_texture_directory(self, project_id: str, source_dir: Path, watermark: int, classification: Classification | str | None = None, precision_report: dict | None = None) -> ProcessingArtifact:
        source_dir = Path(source_dir)
        textures = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}) if source_dir.is_dir() else []
        if not textures:
            raise ValueError("no JPG/PNG textures found")
        use_trustmark = self.trustmark_engine is not None and hasattr(self.trustmark_engine, "embed_image")
        artifact_id, output_dir = self._artifact_dir(project_id)
        try:
            for source in textures:
                destination = output_dir / source.name
                if use_trustmark:
                    self.trustmark_engine.embed_image(source, destination, watermark)
                elif sandbox_compute.enabled():
                    sandbox_compute.run("texture_embed", str(source), str(destination), f"{watermark:08X}")
                else:
                    embed_texture_watermark_adaptive_jpeg(source, destination, watermark)
            self._write_manifest(output_dir, project_id, artifact_id, "OSGB_TEXTURES", source_dir, output_dir, len(textures), classification, precision_report, watermark)
            self._verify_manifest(output_dir, "OSGB_TEXTURES", output_dir, watermark)
            self._trace_manifest(output_dir, "WATERMARK_VERIFIED")
            return ProcessingArtifact(artifact_id, "OSGB_TEXTURES", str(source_dir), str(output_dir), len(textures), "PENDING")
        except Exception:
            self._failed(output_dir)
            raise

    def process_osgb_tree(self, project_id: str, source_dir: Path, watermark: int, classification: Classification | str | None = None, processing_profile: str = "PERFORMANCE", precision_report: dict | None = None) -> ProcessingArtifact:
        artifact_id, output_dir = self._artifact_dir(project_id)
        try:
            if sandbox_compute.enabled() and processing_profile == "STRICT_PER_MEMBER":
                output_dir.mkdir(parents=True, exist_ok=True)
                for path in source_dir.rglob('*'):
                    if path.is_dir():
                        (output_dir / path.relative_to(source_dir)).mkdir(parents=True, exist_ok=True)
                    elif path.suffix.lower() != '.osgb':
                        target = output_dir / path.relative_to(source_dir); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
                files = sorted(source_dir.rglob('*.osgb'))
                # 固定 4 个 worker 是按当时的小机器定的；本机 18 核 36 线程，
                # 11.6 GB 的 OSGB 包用 4 路会明显吃不满。改为可配，默认 8。
                try:
                    _wanted = int(os.environ.get("GDS_OSGB_WORKERS", "8") or 8)
                except ValueError:
                    _wanted = 8
                workers = min(max(1, _wanted), max(1, len(files)))
                # 单批超时：11.6GB/28581 文件的实测耗时是 26 分钟（4 路并行、机器空闲时），
                # 但并发竞争时会翻倍到 60 分钟以上并撞上原先硬编码的 3600s，
                # 表现为"数据传完了、处理却超时失败"。改成可配，默认 120 分钟留足余量。
                try:
                    _timeout = int(os.environ.get("GDS_OSGB_TIMEOUT_S", "7200") or 7200)
                except ValueError:
                    _timeout = 7200
                jobs=[]
                for i in range(workers):
                    items=[{'source':str(p), 'destination':str(output_dir / p.relative_to(source_dir))} for p in files[i::workers]]
                    job_path=self.workspace / 'sandbox' / 'work' / f'osgb_dct_api_{artifact_id}_{i}.json'; job_path.parent.mkdir(parents=True, exist_ok=True); job_path.write_text(json.dumps({'watermark':f'{watermark:08X}','items':items}, ensure_ascii=False), encoding='utf-8'); jobs.append(job_path)
                parts=[]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures=[pool.submit(sandbox_compute.run, 'osgb_dct_file_batch', str(job), profile='main', timeout_s=_timeout) for job in jobs]
                    for future in as_completed(futures): parts.append(future.result())
                result={'runner_user':'gds_sbx','files':sum(int(p.get('files',0)) for p in parts),'pixel_samples':sum(int(p.get('pixel_samples',0)) for p in parts),'pixel_matched':sum(int(p.get('pixel_matched',0)) for p in parts),'not_watermarkable':sum(int(p.get('not_watermarkable',0)) for p in parts),'failed':sum(int(p.get('failed',0)) for p in parts)}
                count = int(result.get('files', 0))
            elif sandbox_compute.enabled():
                result = sandbox_compute.run('osgb_tree_embed', str(source_dir), str(output_dir), f'{watermark:08X}')
                count = int(result.get("files_processed", 0) if isinstance(result, dict) else (result or 0))
            else:
                count = embed_osgb_tree(source_dir, output_dir, watermark)
            self._write_manifest(output_dir, project_id, artifact_id, "OSGB", source_dir, output_dir, count, classification, precision_report, watermark)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["processing_profile"] = processing_profile
            manifest["watermark_method"] = "JPEG_DCT_CPU_PER_MEMBER" if processing_profile == "STRICT_PER_MEMBER" else "SPATIAL_TILE_CARRIER"
            if processing_profile == "STRICT_PER_MEMBER" and sandbox_compute.enabled():
                verification = {
                    "verified": result.get("failed", 0) == 0 and result.get("pixel_samples", 0) > 0 and result.get("pixel_matched", 0) == result.get("pixel_samples", 0),
                    "method": "jpeg_dct_cpu_per_member_extract",
                    "samples": int(result.get("pixel_samples", 0)),
                    "matched": int(result.get("pixel_matched", 0)),
                    "not_watermarkable": int(result.get("not_watermarkable", 0)),
                    "failed": int(result.get("failed", 0)),
                    "files_processed": count,
                    "runner_user": result.get("runner_user", "gds_sbx"),
                }
                manifest["verification"] = verification
                manifest["verification_status"] = "VERIFIED" if verification["verified"] else "VERIFICATION_FAILED"
                manifest["expected_watermark"] = watermark
            (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if not (processing_profile == "STRICT_PER_MEMBER" and sandbox_compute.enabled()):
                self._verify_manifest(output_dir, "OSGB", output_dir, watermark)
            elif not manifest["verification"]["verified"]:
                verification = manifest["verification"]
                if int(verification.get("samples", 0)) == 0:
                    # 不可嵌入不是「处理出错」，是载体类型不支持：说清楚怎么办，别抛字典。
                    raise ValueError(
                        "该 OSGB 没有可用的 JPEG 纹理载体，无法嵌入水印。"
                        "外置 JPEG 纹理已支持；纹素为 PNG 或内嵌 RGB 时当前 DCT 方案处理不了。"
                        "请改用 JPEG 纹理的 OSGB，或先把纹理转成 JPEG 再上传。"
                        f"（本次文件数 {verification.get('files_processed', 0)}，"
                        f"不可嵌入成员 {verification.get('not_watermarkable', 0)}）")
                raise ValueError(f"DCT per-member verification failed: {verification}")
            self._trace_manifest(output_dir, "WATERMARK_VERIFIED")
            return ProcessingArtifact(artifact_id, "OSGB", str(source_dir), str(output_dir), count, "PENDING")
        except Exception:
            self._failed(output_dir)
            raise

    def process_shapefile_directory(self, project_id: str, source_dir: Path, user_id: str, timestamp: str, watermark_text: str | None = None, classification: Classification | str | None = None, precision_report: dict | None = None) -> ProcessingArtifact:
        """Embed the encrypted attribute watermark into every .shp layer of a DLG/SHP
        directory (real DLG data is a multi-layer SHP folder). Geometry untouched."""
        from .vector_watermark import embed_shapefile_watermark

        source_dir = Path(source_dir)
        # 大写 .SHP 在 Linux（区分大小写）上 rglob("*.shp") 匹配不到，统一按扩展名小写判断；
        # 单文件载荷（S1 只传了一个 .shp）以前直接落到「没有图层」，把现场问题说成了空话。
        if source_dir.is_file():
            raise ValueError(
                f"no .shp layers found：本次上传的是单个文件 {source_dir.name}，"
                "而 Shapefile 水印要写进 .dbf 属性表，必须 .shp/.shx/.dbf/.prj 同名整套一起上传。"
                "请把整个图层目录打成 ZIP 重新上传（或用工具导出为单个 .gpkg 后按 GPKG 上传）。")
        layers = sorted(path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".shp")
        if not layers:
            raise ValueError(
                f"no .shp layers found：解包目录里没有 .shp，实际内容：{_payload_carrier_summary(source_dir)}。"
                "若是 File Geodatabase(.gdb)、CAD(.dwg/.dxf)、压缩包里还套着压缩包或空目录，"
                "请先导出为 Shapefile（.shp/.shx/.dbf/.prj）或转成单个 .gpkg 再上传。")
        text = watermark_text or f"{user_id}|{timestamp}"
        artifact_id, output_dir = self._artifact_dir(project_id)
        try:
            with _shp_phase(artifact_id, "embed"):
                if sandbox_compute.enabled():
                    # 单次沙箱会话批量嵌入全部图层（避免逐层拉起进程；1GB 级图层目录必需）
                    jobs_dir = Path(self.workspace) / "sandbox" / "work" / "jobs"
                    jobs_dir.mkdir(parents=True, exist_ok=True)
                    filelist = jobs_dir / f"shp_batch_{uuid.uuid4().hex}.json"
                    filelist.write_text(json.dumps({
                        "layers": [
                            {"src": str(source), "dst": str(output_dir / source.name), "text": text}
                            for source in layers
                        ],
                    }, ensure_ascii=False), encoding="utf-8")
                    try:
                        sandbox_compute.run("shp_batch_embed", str(filelist))
                    finally:
                        filelist.unlink(missing_ok=True)
                else:
                    for source in layers:
                        embed_shapefile_watermark(source, output_dir / source.name, text, self.key)
            with _shp_phase(artifact_id, "manifest_hash"):
                self._write_manifest(output_dir, project_id, artifact_id, "SHP", source_dir, output_dir, len(layers), classification, precision_report)
            preexisting = []
            for layer in layers:
                try:
                    import shapefile
                    with shapefile.Reader(str(layer), encoding="gbk", encodingErrors="replace") as reader:
                        if any(field[0].upper() == "_WM" for field in reader.fields[1:]):
                            preexisting.append(layer.stem)
                except Exception:
                    pass
            self._mark_rewatermark(output_dir, preexisting)
            with _shp_phase(artifact_id, "verify"):
                self._verify_manifest(output_dir, "SHP", output_dir, text)
            self._trace_manifest(output_dir, "WATERMARK_VERIFIED")
            return ProcessingArtifact(artifact_id, "SHP", str(source_dir), str(output_dir), len(layers), "PENDING")
        except Exception:
            self._failed(output_dir)
            raise

    def process_geopackage(self, project_id: str, source: Path, user_id: str, timestamp: str, watermark_text: str | None = None, classification: Classification | str | None = None, precision_report: dict | None = None) -> ProcessingArtifact:
        """Embed encrypted attribute watermarks in all GeoPackage feature layers."""
        from .geopackage_watermark import embed_geopackage_watermark
        source = Path(source)
        if not source.is_file() or source.suffix.lower() != ".gpkg":
            raise ValueError("GeoPackage source must be a .gpkg file")
        text = watermark_text or f"{user_id}|{timestamp}"
        artifact_id, output_dir = self._artifact_dir(project_id)
        destination = output_dir / source.name
        try:
            result = sandbox_compute.run("gpkg_embed", str(source), str(destination), text) if sandbox_compute.enabled() else embed_geopackage_watermark(source, destination, text, self.key)
            self._write_manifest(output_dir, project_id, artifact_id, "GPKG", source, destination, int(result.get("features", 0)), classification, precision_report)
            self._mark_rewatermark(output_dir, self._preexisting_gpkg_layers(source))
            self._verify_manifest(output_dir, "GPKG", destination, text)
            self._trace_manifest(output_dir, "WATERMARK_VERIFIED")
            return ProcessingArtifact(artifact_id, "GPKG", str(source), str(destination), int(result.get("features", 0)), "PENDING")
        except Exception:
            self._failed(output_dir)
            raise

    def _tree_bytes(self, source: Path) -> int:
        if source.is_file():
            return source.stat().st_size
        return sum(p.stat().st_size for p in source.rglob("*") if p.is_file())

    def _ensure_disk_headroom(self, source: Path) -> None:
        """处理前先看余量：按源大小估算成果体积，再留 2GB 余量。

        不预检的后果是先把大文件跑上几分钟、写满盘才报 No space left on device，
        既白耗一次上传，又在成果目录留下半成品。这里直接给可操作的中文提示。

        必须量「成果要写进去的那块盘」（workspace），不是源文件所在的盘：
        E: 只放上传，成果写 V:。按源盘去量会拿 E: 的余量否掉本来能跑的任务。
        系数也不是固定 2x：只有源与成果同盘时才需要源 + 成果并存，异盘只需放成果本身
        （实测 DEM 源 18.5GB 在 E:、V: 剩 38.2GB，被 39.1GB 的 2x 估算误判为不足）。
        """
        try:
            size = self._tree_bytes(source)
            try:
                src_drive = os.path.splitdrive(str(Path(source).resolve()))[0].upper()
                dst_drive = os.path.splitdrive(str(self.workspace.resolve()))[0].upper()
                same_volume = src_drive == dst_drive
            except OSError:
                same_volume = True
            need = size * (2 if same_volume else 1) + 2 * 1024 ** 3
            free = shutil.disk_usage(self.workspace).free
        except OSError:
            return
        if free >= need:
            return
        raise RuntimeError(
            f"成果写入盘（{self.workspace}）空间不足：本次处理约需 {need / 2**30:.1f}GB，"
            f"当前可用 {free / 2**30:.1f}GB。请清理该盘上的历史成果与导出包后重试。")

    def _mark_rewatermark(self, output_dir: Path, layers: list[str]) -> None:
        """在成果清单里记下「输入本身已带水印、本次为覆盖重加」。

        覆盖重加是允许的（导出的成果再导入属于正常业务），但验收和事后追溯必须能看出来，
        否则无法区分「首次加注」与「替换了上一手标记」。
        """
        if not layers:
            return
        manifest_path = Path(output_dir) / "manifest.json"
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["rewatermarked"] = True
        manifest["rewatermarked_layers"] = layers
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _preexisting_gpkg_layers(source: Path) -> list[str]:
        import sqlite3
        found = []
        try:
            with sqlite3.connect(Path(source)) as connection:
                for (table,) in connection.execute(
                        "SELECT table_name FROM gpkg_contents WHERE data_type = 'features'"):
                    columns = {row[1].lower() for row in connection.execute(f'PRAGMA table_info("{table}")')}
                    if "_wm" in columns:
                        found.append(table)
        except Exception:
            return []
        return found

    def record_artifact_policy(self, artifact: ProcessingArtifact, *, watermark_enabled: bool, anchor_granularity: str) -> None:
        output = Path(artifact.output)
        manifest_path = output / "manifest.json" if (output / "manifest.json").is_file() else output.parent / "manifest.json"
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["watermark_enabled"] = bool(watermark_enabled)
        manifest["anchor_granularity"] = anchor_granularity
        if not watermark_enabled:
            manifest["watermark_method"] = "NONE"
            manifest["verification_status"] = "NOT_REQUESTED"
        if anchor_granularity in {"UNIT", "TILE"}:
            root = output if output.is_dir() else output.parent
            candidates = sorted(item for item in root.rglob("*")
                                if item.is_file() and item.name != "manifest.json")
            if anchor_granularity == "TILE":
                # 最小上链对象：栅格按内部 tile 切分，而不是按整个文件。
                # 一个 GeoTIFF 内部可能有很多 tile，把每个 tile 的像素摘要各自上链，
                # 这样"篡改了哪一块"可以直接落到区块粒度上说清。
                candidates = [item for item in candidates
                              if item.suffix.lower() in {".tif", ".tiff", ".img", ".vrt"}] or candidates
            entries = []
            for item in candidates:
                if anchor_granularity == "UNIT":
                    units = [(str(item.relative_to(root)).replace("\\", "/"),
                              self._sha256_path(item))]
                else:
                    units = self._tile_units(item, root)
                for relative, digest in units:
                    entries.append({"unit": relative, "sha256": digest})
                    self.trace_chain.append("DATA_UNIT_ANCHORED", artifact_id=artifact.artifact_id,
                                            project_id=manifest["project_id"],
                                            classification=manifest.get("classification", "UNCLASSIFIED"),
                                            unit=relative, input_hash=digest, output_hash=digest,
                                            granularity=anchor_granularity,
                                            sandbox_user="gds_sbx" if sandbox_compute.enabled() else "direct")
            manifest["anchor_units"] = entries
            manifest["anchor_granularity"] = anchor_granularity
            # 若请求 TILE 但载体读不出块，会退回按文件；必须让这件事可见，不能静默。
            actually_tiled = any("#tile" in str(entry.get("unit", "")) for entry in entries)
            manifest["anchor_granularity_effective"] = (
                "TILE" if actually_tiled else ("UNIT" if anchor_granularity == "TILE" else anchor_granularity))
            if anchor_granularity == "TILE" and not actually_tiled and entries:
                manifest["anchor_granularity_note"] = (
                    "请求 TILE 粒度，但载体无法按内部块切分，已如实退回按文件上链（未假装切分）")
            manifest["anchor_unit_count"] = len(entries)
        else:
            # UPLOAD（最大粒度）：整个上传算一个区块，只锚定整体输出哈希，
            # 因此不展开逐单元；但仍要如实报出"实际生效粒度"和单元数，保持接口口径一致。
            manifest["anchor_granularity"] = anchor_granularity
            manifest["anchor_granularity_effective"] = anchor_granularity
            manifest["anchor_unit_count"] = 1
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _tile_units(self, item: Path, root: Path) -> list[tuple[str, str]]:
        """把一个栅格按内部块切分成可独立上链的最小单元。

        用 rasterio 的 block_windows 读真实块（GeoTIFF 内部本来就是分块存储的），
        每个块算一个摘要。读不了（非栅格/驱动不支持）时退回整文件一个单元——
        绝不假装切过。
        """
        relative = str(item.relative_to(root)).replace("\\", "/")
        try:
            import rasterio
            with rasterio.open(item) as src:
                units: list[tuple[str, str]] = []
                # rasterio >=1.3 的 block_windows 产出 ((row_block, col_block), Window)；
                # 旧版只产出 Window。两种都接住，否则会静默退回整文件一个单元。
                for block_index, window in src.block_windows(1):
                    rb, cb = block_index if isinstance(block_index, tuple) else (block_index, 0)
                    if hasattr(window, "row_off"):
                        row_off, col_off = window.row_off, window.col_off
                        view = window
                    else:  # 极旧版本只给 Window
                        row_off, col_off = window
                        view = rasterio.windows.Window(col_off, row_off,
                                                       src.block_shapes[0][1], src.block_shapes[0][0])
                    block = src.read(1, window=view)
                    digest = hashlib.sha256(block.tobytes()).hexdigest()
                    units.append((f"{relative}#tile{rb}_{cb}@r{row_off}c{col_off}", digest))
                if units:
                    return units
        except Exception:
            pass
        return [(relative, self._sha256_path(item))]

    def process_unwatermarked(self, project_id: str, source: Path, data_type: str, classification=None,
                              anchor_granularity: str = "UPLOAD", precision_report: dict | None = None) -> ProcessingArtifact:
        """Store the uploaded payload unchanged while still creating a traceable artifact.

        注意：关水印不等于关降精度。调用方应先按需做完降精度，再把结果路径传进来；
        precision_report 只用于把降精度报告写进清单，不在这里执行降精度。
        """
        source = Path(source)
        artifact_id, output_dir = self._artifact_dir(project_id)
        try:
            destination = output_dir / (source.name if source.is_file() else "payload")
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            units = [destination] if destination.is_file() else sorted(
                item for item in destination.rglob("*") if item.is_file() and item.name not in {".gds_extract_complete", "payload_inventory.json", "manifest.json"}
            )
            self._write_manifest(output_dir, project_id, artifact_id, data_type, source, destination, len(units), classification)
            manifest_path = output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({"watermark_enabled": False, "watermark_method": "NONE", "verification_status": "NOT_REQUESTED", "anchor_granularity": anchor_granularity})
            if precision_report is not None:
                manifest["precision"] = precision_report
            if anchor_granularity == "UNIT":
                manifest["anchor_units"] = [{"unit": str(item.relative_to(destination if destination.is_dir() else item.parent)), "sha256": self._sha256_path(item)} for item in units]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            self._trace_manifest(output_dir, "WATERMARK_SKIPPED")
            return ProcessingArtifact(artifact_id, data_type, str(source), str(destination), len(units), "PENDING")
        except Exception:
            self._failed(output_dir)
            raise

    def process(self, project_id: str, data_type: str, source: Path, **kwargs) -> ProcessingArtifact:
        normalized = data_type.upper()
        # S1 按扩展名映射格式，ZIP 承载的 SHP 会被声明成 GPKG（BE-DLG），直接按声明分派
        # 就会在 GPKG 处理器里报「payload 里没有 .gpkg」。这里按载荷实际内容校正一次：
        # 只在本族内互换（GPKG/SHP/GEOJSON 都是 DLG 矢量处理器），不会把栅格当矢量。
        corrected = _content_vector_type(Path(source))
        if corrected and corrected != normalized and {normalized, corrected} <= {"GPKG", "GEOPACKAGE", "SHP", "GEOJSON"}:
            data_type = corrected
            normalized = corrected
        classification = parse_classification(kwargs["classification"]) if kwargs.get("classification") else None
        precision_report = kwargs.get("precision_report")
        self._ensure_disk_headroom(Path(source))
        if normalized in {"GEOJSON"}:
            return self.process_geojson(project_id, source, kwargs["user_id"], kwargs["timestamp"], kwargs.get("decimals"), classification, precision_report)
        if normalized == "SHP":
            return self.process_shapefile_directory(project_id, source, kwargs["user_id"], kwargs["timestamp"], kwargs.get("watermark_text"), classification, precision_report)
        if normalized in {"GPKG", "GEOPACKAGE"}:
            return self.process_geopackage(project_id, source, kwargs["user_id"], kwargs["timestamp"], kwargs.get("watermark_text"), classification, precision_report)
        if normalized in {"GEOTIFF", "GTIFF"}:
            return self.process_geotiff(project_id, source, kwargs["watermark"], classification, kwargs.get("preflight"), precision_report,
                                        kwargs.get("watermark_mode", "FORENSIC"), kwargs.get("visible_watermark_text"))
        if normalized == "DEM":
            return self.process_dem(project_id, source, kwargs["watermark"], classification, precision_report)
        if normalized == "POINT_CLOUD":
            return self.process_point_cloud(project_id, source, kwargs["watermark_text"], classification, precision_report)
        if normalized == "OSGB":
            return self.process_osgb_tree(project_id, source, kwargs["watermark"], classification, kwargs.get("processing_profile", "PERFORMANCE"), precision_report)
        if normalized in {"OSGB_TEXTURES", "TEXTURES"}:
            return self.process_texture_directory(project_id, source, kwargs["watermark"], classification, precision_report)
        raise UnsupportedDataTypeError(f"no verified local handler for {data_type}")

    def process_dem(self, project_id: str, source: Path, watermark: int, classification=None, precision_report: dict | None = None) -> ProcessingArtifact:
        """Stream each DEM raster; uploaded ZIP payloads are directories, not raster files.

        使用**跨瓦片（grid-independent）**方案：水印按行内 64 采样的整组写入，
        因此提取与瓦片划分无关——裁剪、换 stride 都能读回（旧 `stream_embed` 只能同
        瓦片网格提取）。
        """
        from .dem_stream import cross_tile_embed, cross_tile_extract
        from .dem_block_channel import block_mean_embed, block_mean_extract
        source = Path(source)
        inputs = [source] if source.is_file() else sorted(
            item for item in source.rglob("*") if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}
        )
        if not inputs:
            raise ValueError("no .tif/.tiff DEM files found in input")
        artifact_id, output_dir = self._artifact_dir(project_id)
        try:
            rows = []
            for item in inputs:
                relative = Path(item.name) if source.is_file() else item.relative_to(source)
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if sandbox_compute.enabled():
                    result = sandbox_compute.run("dem_cross_tile_embed", str(item), str(destination), f"{watermark:08X}")
                    verification = sandbox_compute.run("dem_cross_tile_extract", str(destination), f"{watermark:08X}")
                else:
                    # 临时目录必须显式指定：TMP/TEMP 环境变量在某些启动路径下没生效，
                    # 而 tempfile 默认落到 C:（实测只剩 4.5GB），GDAL 写 lsb.tif 需要
                    # 5.8GB 就直接失败，报 'Free disk space available is ...'。
                    scratch = os.environ.get("GDS_SCRATCH") or str(Path(self.workspace) / "tmp")
                    Path(scratch).mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix="gds_dem_", dir=scratch) as td:
                        lsb_tmp = Path(td) / "lsb.tif"
                        result = cross_tile_embed(item, lsb_tmp, watermark)
                        # 块均值通道是**增强项**：它失败绝不能连累主嵌入。
                        # 之前这里直接写 destination，一旦抛异常整个 DEM 任务失败，
                        # 而 LSB 明明已经成功——这正是"上传嵌入失败"的来源。
                        try:
                            block = block_mean_embed(lsb_tmp, destination, watermark)
                            result = {**result, "block_mean": block}
                        except Exception as exc:  # noqa: BLE001
                            shutil.copy2(lsb_tmp, destination)
                            result = {**result, "block_mean": {
                                "error": f"{type(exc).__name__}: {str(exc)[:120]}",
                                "degraded": True,
                                "note": "块均值通道不可用，已退回仅 LSB 通道（主水印完好）"}}
                    verification = cross_tile_extract(destination, watermark)
                    if not verification.get("verified"):
                        bm = block_mean_extract(destination, watermark)
                        verification = {**verification, "block_mean": bm,
                                        "verified": bool(verification.get("verified") or bm.get("matched"))}
                if not verification.get("verified"):
                    raise ValueError(f"DEM cross-tile verification failed for {relative}: {verification}")
                rows.append({"source": str(item), "output": str(destination), "embedding": result, "verification": verification})
            total_tiles = sum(int(row["embedding"].get("replicas", 0)) for row in rows)
            self._write_manifest(output_dir, project_id, artifact_id, "DEM", source, output_dir, total_tiles, classification, precision_report, watermark)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["watermark_scheme"] = "cross_tile_v2"
            manifest["grid_independent_extraction"] = True
            layout = rows[0]["embedding"] if rows else {}
            manifest["cross_tile_layout"] = {
                "samples_per_bit": layout.get("samples_per_bit"),
                "replica_length": layout.get("group_length"),
                "replica_stride": layout.get("replica_stride"),
                "min_window_width": layout.get("min_window_width"),
                "participation": layout.get("participation"),
            }
            manifest["verification"] = {"verified": True, "files": rows,
                                        "samples": sum(int(row["verification"].get("groups_checked", 0)) for row in rows),
                                        "matched": sum(int(row["verification"].get("matched", 0)) for row in rows)}
            manifest["verification_status"] = "VERIFIED"
            (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            self._trace_manifest(output_dir, "WATERMARK_VERIFIED")
            return ProcessingArtifact(artifact_id, "DEM", str(source), str(output_dir), total_tiles, "PENDING")
        except Exception:
            self._failed(output_dir); raise

    def process_point_cloud(self, project_id: str, source: Path, watermark_text: str, classification=None, precision_report: dict | None = None) -> ProcessingArtifact:
        """点云水印：文本格式走**坐标域 LSB**，二进制格式显式失败（不再静默写 sidecar）。"""
        from .point_cloud import COORDINATE_DOMAIN_EXTENSIONS, discover, embed_file, extract_file
        files = discover(source)
        if not files:
            raise ValueError("no supported point-cloud files found")
        artifact_id, output_dir = self._artifact_dir(project_id)
        try:
            rows = []
            for item in files:
                destination = output_dir / item.name
                if sandbox_compute.enabled():
                    row = sandbox_compute.run("point_cloud_coordinate_embed", str(item), str(destination), watermark_text)
                    check = sandbox_compute.run("point_cloud_coordinate_extract", str(destination), watermark_text)
                else:
                    row = embed_file(item, destination, watermark_text)
                    check = extract_file(destination, watermark_text)
                rows.append({"file": item.name, "embedding": row, "verification": check})
            failed = [row["file"] for row in rows if not row["verification"].get("verified")]
            if failed:
                raise ValueError(f"point-cloud coordinate-domain verification failed: {failed}")
            coordinate_domain = all(row["embedding"].get("coordinate_domain_verified") for row in rows)
            method = "coordinate_lsb" if coordinate_domain else "streamed_hash_sidecar"
            self._write_manifest(output_dir, project_id, artifact_id, "POINT_CLOUD", Path(source), output_dir, len(rows), classification, precision_report)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            manifest["watermark_domain"] = "coordinate_lsb" if coordinate_domain else "hash_sidecar"
            manifest["verification"] = {
                "verified": True,
                "method": method,
                "coordinate_domain_verified": coordinate_domain,
                "samples": len(rows),
                "matched": len(rows),
                "files": rows,
            }
            manifest["verification_status"] = "VERIFIED"
            (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            self._trace_manifest(output_dir, "WATERMARK_VERIFIED")
            return ProcessingArtifact(artifact_id, "POINT_CLOUD", str(source), str(output_dir), len(rows), "PENDING")
        except Exception:
            self._failed(output_dir); raise
