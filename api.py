from __future__ import annotations

import os
import json
import math
import secrets
import sys
import hashlib
import ipaddress
import re
import shutil
import sqlite3
import threading
import time
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

# A partial result is a legitimate deliverable and stays readable, but it must never be
# indistinguishable from a full approval, so the artifact status carries the distinction and every
# download/export gate accepts exactly these two states.
APPROVAL_APPROVED = "APPROVED"
APPROVAL_PARTIAL = "PARTIALLY_APPROVED"
APPROVAL_REJECTED = "REJECTED"
APPROVED_STATES = frozenset({APPROVAL_APPROVED, APPROVAL_PARTIAL})
# 部分处理的成果只要已嵌入载体的覆盖率不低于验收口径（水印嵌入成功率 ≥95%），
# 就按"部分通过"交付，而不是卡在人工复核队列里等一个不会改变结果的签字。
# 环境变量可覆盖，便于现场按数据质量调整。
PARTIAL_ACCEPT_MIN_COVERAGE = float(os.environ.get("GDS_PARTIAL_ACCEPT_COVERAGE", "0.95") or 0.95)
# States a result carries a recorded human decision and therefore never leaves: a reviewer's verdict
# is not flipped by a later call, and delivering rejected data needs a new task.
DECIDED_STATES = APPROVED_STATES | {APPROVAL_REJECTED}

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Form, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.requests import ClientDisconnect
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, BeforeValidator, model_validator
from pydantic_core import PydanticCustomError

from .pipeline import LocalProcessingPipeline, OutputNotReadyError, UnknownResultError
from .processing import ProcessingService, UnsupportedDataTypeError
from .task_store import TaskStore
from .loan_store import (LoanStore, STATUS_LENT, STATUS_OVERDUE, STATUS_RETURNED,
                         VERDICT_OK, VERDICT_OVERDUE, VERDICT_MISMATCH,
                         VERDICT_NOT_OURS, VERDICT_UNKNOWN, summarize as summarize_loans,
                         default_due)
from .circulation import CirculationStore
from .circulation_api import build_circulation_router
from .chain_board import render_html as render_chain_board
from .chain_graph import render_html as render_chain_graph
from .sandbox_console import render_html as render_sandbox_console
from .classification import Classification, CAPABILITIES, parse_classification, allowed_for_data_type
from .verification import verify_artifact
from .admin_auth import AdminAuthStore
from .storage_policy import StoragePolicy, PolicyError
from .sandbox_policy import SandboxPolicyStore
from .sandbox_pool import (SandboxPoolStore, NoCapacityError, create_container,
                           mount_container, unmount_letter, free_drive_letters,
                           PLATFORM_DIRNAME)
from .timefmt import now_local
from .integrity import source_integrity_warning, unreadable_source_warning
from . import sandbox_compute

try:  # prefer server2/audit when deployed as a sibling module
    import audit_log  # type: ignore
except Exception:  # package-local fallback used by PYTHONPATH=server2/pipeline
    from . import audit_log  # type: ignore

_HAS_AUDIT = True


def _as_int(value, default=None):
    """安全取整：manifest 里的计数可能是 None 或字符串，缺值时返回 default。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def rejected_result_guard(status: str) -> None:
    """A rejected result is a decision, not a missing approval - report it as such."""
    if status == APPROVAL_REJECTED:
        raise HTTPException(status_code=409, detail="result was rejected by manual review and is not deliverable")


def _asset_source_resolver(task_store):
    """把流转资产上的 result_id 解析成真实产出路径（文件或目录）。"""
    def resolve(asset: dict) -> Path | None:
        result_id = str((asset or {}).get("result_id") or "")
        if not result_id:
            return None
        record = task_store.get_artifact_record(result_id)
        output = (record or {}).get("output") or ""
        candidate = Path(str(output)) if output else None
        return candidate if candidate and candidate.exists() else None
    return resolve


def _result_status_resolver(task_store):
    """查成果当前审批状态；库里没有这条成果时返回 None（由调用方要求显式水印号）。"""
    def status(result_id: str) -> str | None:
        record = task_store.get_artifact_record(result_id)
        return (record or {}).get("status")
    return status


def partial_watermark_acceptable(manifest: dict) -> bool:
    """部分处理的成果，水印是否已经达到可交付标准。

    不完整上传（源文件截断）不该阻断业务闭环：缺的那几块像素本来就没有内容，
    也嵌不进任何水印。真正决定这份数据能不能被溯源、能不能交付的，是**已嵌入
    载体上的水印能不能重新抽出来**，以及覆盖率是否达到验收口径的 95%。

    这里只看水印证据，不看源文件完整性 —— 完整性由 `SOURCE_TRUNCATED` 警告和
    `verification_status=VERIFIED_PARTIAL` 如实保留，前端照常显示"部分处理"。
    """
    if manifest.get("verification_status") != "VERIFIED_PARTIAL":
        return False
    partial = manifest.get("partial_processing") or {}
    total = _as_int(partial.get("carriers_total"))
    embedded = _as_int(partial.get("carriers_embedded"))
    if not total or embedded is None:
        return False
    if embedded / total < PARTIAL_ACCEPT_MIN_COVERAGE:
        return False
    verification = manifest.get("verification") or {}
    # detected 表示至少一个载体解出了标记；verified 表示达到项目整体的载体覆盖率门槛。
    return bool(verification.get("verified") or verification.get("detected"))


def needs_manual_review(manifest: dict) -> bool:
    """Whether a manifest belongs in the human review queue.

    Keyed on the data-quality signals the processing stage writes, not on the artifact status, so a
    result that was never approved but also never flagged stays out of the queue.
    """
    return (
        manifest.get("source_integrity") == "INCOMPLETE"
        or manifest.get("verification_status") in {"VERIFIED_PARTIAL", "VERIFICATION_FAILED"}
        or manifest.get("approval_policy") == "MANUAL_REVIEW_REQUIRED"
    )


class ApproveRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128)
    manual_review_confirmed: bool = False
    review_conclusion: str | None = Field(default=None, min_length=1, max_length=2000)


class RejectRequest(BaseModel):
    """A rejection must say who decided and why; there is no body-less reject."""
    reviewer_id: str = Field(min_length=1, max_length=128)
    review_conclusion: str = Field(min_length=1, max_length=2000)


class ResultLookupRequest(BaseModel):
    """Batch reconciliation: a caller asks what this node still holds for a set of result ids."""
    result_ids: list[str] = Field(default_factory=list, max_length=200)


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


class AdminPasswordChangeRequest(BaseModel):
    old_password: str = Field(min_length=1, max_length=512)
    new_password: str = Field(min_length=12, max_length=512)


class AdminUserCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    role: str = Field(pattern=r"^(ADMIN|USER)$")
    password: str = Field(min_length=12, max_length=512)


class RetentionUpdateRequest(BaseModel):
    """日志存储配置的局部更新：只改传上来的字段。"""
    retentionDays: int | None = None
    batchSize: int | None = None
    scheduleEnabled: bool | None = None
    cron: str | None = Field(default=None, max_length=64)
    maxTotalMb: int | None = None
    targets: dict[str, bool] | None = None
    operator: str | None = Field(default=None, max_length=128)


class RetentionCleanupRequest(BaseModel):
    """立即清理。默认 dryRun=true，前端先预演给用户看。"""
    dryRun: bool = True
    targets: dict[str, bool] | None = None
    confirmCapacity: bool = False
    operator: str | None = Field(default=None, max_length=128)


# 清理目标白名单：只有数据区可清。state/（链 + 管理员凭据）、audit/（合规审计）、
# sandbox/worker/（沙箱执行体）永久排除，避免一次清理把平台变成不可恢复状态。
SANDBOX_CLEANABLE_AREAS = ("incoming", "projects", "exports", "sandbox_inbox", "provenance_geom")

# 分片上传的吞吐取决于「每片固定往返开销」，不是带宽。
# 实测（本机 127.0.0.1，串行）：64KB 分片 → 2.9 MB/s；512KB → 22.6；4MB → 88；
# 16MB → 168；32MB → 177；64MB → 164（已经开始受单次写入影响）。
# 即：每多一个分片就多付约 30ms，分片越小吞吐成比例掉。
# 前端如果按 64KB 切，1GB 文件要 16384 次往返，纯服务端开销就是 8 分钟。
UPLOAD_CHUNK_RECOMMENDED = 16 * 1024 * 1024
UPLOAD_CHUNK_MIN_SANE = 4 * 1024 * 1024
UPLOAD_CHUNK_MAX = 64 * 1024 * 1024

# 剩余空间检查按盘符做 5 秒缓存：原来每片都 stat 一次加密卷，
# 分片上千时这一项就能把上传拖慢一截；而盘空间在 5 秒内不会突变。
_FREE_SPACE_CACHE: dict[str, tuple[float, int]] = {}


def _free_bytes_cached(anchor: str, ttl: float = 5.0) -> int | None:
    now = time.monotonic()
    hit = _FREE_SPACE_CACHE.get(anchor)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        free = shutil.disk_usage(anchor).free
    except OSError:
        return None
    _FREE_SPACE_CACHE[anchor] = (now, free)
    return free


def _chunk_advice(requested: int, size_bytes: int) -> dict:
    """给调用方的分片建议。**不改变实际分片大小**——偏移量必须用调用方真实用的分片算，
    否则 64KB 分片按 4MB 落盘会把文件写错位，最后 SHA-256 对不上、整次上传白做。

    所以这里只做「告知 + 审计」，把吞吐差距摊在明面上；
    真正提速要靠调用方按返回的 recommended_chunk_size 切。
    """
    if size_bytes <= UPLOAD_CHUNK_RECOMMENDED:
        return {"recommended_chunk_size": max(1024, int(size_bytes)),
                "too_small": False, "estimated_throughput_mbps": None,
                "note": "文件小于推荐分片，单次上传即可"}
    too_small = int(requested) < UPLOAD_CHUNK_MIN_SANE
    # 用小分片时，理论吞吐 ≈ 分片大小 / 每次固定开销(约 30ms)
    est = round((int(requested) / (30 / 1000)) / 1024**2, 1) if too_small else None
    return {"recommended_chunk_size": UPLOAD_CHUNK_RECOMMENDED,
            "too_small": too_small,
            "estimated_throughput_mbps": est,
            "note": ("分片偏小，每次往返固定开销约 30ms，吞吐会被分片数拖死；"
                     "改成 16MB 后本机实测从 2.9 MB/s 提升到 168 MB/s" if too_small
                     else "分片大小合适")}


class AdminSandboxQuotaRequest(BaseModel):
    """设置沙箱容量上限。只暴露真正生效的字段，避免出现"配置了但不执行"的假开关。"""
    quota_bytes: int | None = Field(default=None, ge=0)
    reason: str | None = Field(default=None, max_length=500)


class AdminSandboxCleanupRequest(BaseModel):
    """清理沙箱数据。``dry_run`` 默认 True，必须显式给 False 才会真的删。"""
    dry_run: bool = True
    force: bool = False
    areas: list[str] | None = Field(default=None, max_length=5)


class AdminVolumeCreateRequest(BaseModel):
    """新建一个沙箱卷（VeraCrypt 容器）。全 CLI，不需要 GUI。"""
    size_gb: int = Field(ge=1, le=65536)
    letter: str | None = Field(default=None, max_length=1)
    label: str | None = Field(default=None, max_length=128)
    priority: int = Field(default=50, ge=0, le=1000)
    # 口令不落库、不回显：省略时由服务端用主卷的卷口令（同机同一机密），
    # 或由部署方在 server2.env 里通过 GDS_SANDBOX_VOLUME_PASSWORD 注入。
    password: str | None = Field(default=None, min_length=20, max_length=512)


class AdminVolumeActionRequest(BaseModel):
    reason: str | None = Field(default=None, max_length=500)


def _whole_int(value, info):
    """整型字段的前置归一化。

    接受 2 / 2.0 / "2" / 2.0000001（浮点噪声）；拒绝真正带小数的 2.5，并给中文提示，
    避免把 pydantic 的英文 int_from_float 报错直接抛给前端。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise PydanticCustomError("whole_int_bool", "必须是整数，不能是布尔值")
    if isinstance(value, int):
        return value
    raw = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise PydanticCustomError("whole_int_empty", "必须是整数，收到空字符串")
        try:
            raw = float(text)
        except ValueError:
            raise PydanticCustomError(
                "whole_int_text", "必须是整数，收到 {text}",
                {"text": text[:32]}) from None
    if isinstance(raw, float):
        if math.isfinite(raw) and abs(raw - round(raw)) <= 1e-9:
            return int(round(raw))
        raise PydanticCustomError(
            "whole_int_fraction",
            "必须是整数，收到 {value}；请填写不带小数的整数（小数位填 0 到 15，例如 3）",
            {"value": raw})
    raise PydanticCustomError(
        "whole_int_type", "必须是整数，收到 {kind}", {"kind": type(value).__name__})


WholeInt = Annotated[int, BeforeValidator(_whole_int)]


# 参数校验的默认报错是英文（如 "Input should be a valid integer"），前端会原样展示给用户。
# 这里按错误类型给出中文说明，并保留 loc 字段名，方便前端定位是哪个参数填错了。
_VALIDATION_ZH = {
    "missing": "缺少必填参数",
    "int_from_float": "必须是整数，不能带小数",
    "int_parsing": "必须是整数",
    "float_parsing": "必须是数字",
    "bool_parsing": "必须是布尔值 true 或 false",
    "string_too_short": "长度不足，至少 {min_length} 个字符",
    "string_too_long": "长度超限，最多 {max_length} 个字符",
    "greater_than": "必须大于 {gt}",
    "greater_than_equal": "必须大于等于 {ge}",
    "less_than": "必须小于 {lt}",
    "less_than_equal": "必须小于等于 {le}",
    "string_pattern_mismatch": "格式不符合要求，允许的写法：{pattern}",
    "json_invalid": "请求体不是合法的 JSON",
    "literal_error": "取值不在允许范围内，应为 {expected}",
    "enum": "取值不在允许范围内",
    "string_type": "必须是字符串",
    "dict_type": "必须是对象",
    "list_type": "必须是数组",
    "extra_forbidden": "不支持该参数",
}


def _zh_validation_errors(exc) -> list[dict]:
    out = []
    for err in exc.errors():
        loc = [str(x) for x in err.get("loc", [])]
        field = ".".join(loc[1:]) if loc and loc[0] == "body" else ".".join(loc)
        tpl = _VALIDATION_ZH.get(err.get("type", ""), "")
        ctx = err.get("ctx") or {}
        if tpl:
            try:
                msg = tpl.format(**ctx)
            except (KeyError, IndexError, ValueError):
                msg = tpl
        else:
            msg = str(err.get("msg", "参数不合法"))
            for prefix in ("Value error, ", "Assertion failed, ", "Value error, "):
                if msg.startswith(prefix):
                    msg = msg[len(prefix):]
                    break
        out.append({
            "loc": loc,
            "field": field or "(请求体)",
            "type": err.get("type", ""),
            # 不把字段名拼进 msg：Server1 转发时会自己加 “字段名: ”，否则前端会看到 “x: x：...”。
            "msg": msg,
        })
    return out


def _apply_watermark_switch_rule(model) -> None:
    """按「是否加水印」开关决定 watermark_text 的必填性。

    规则（Server1 与前端约定的契约）：
      * watermark_enabled=true  → watermark_text 必须给，且非空；
      * watermark_enabled=false → watermark_text 允许缺失、允许空串；给了也不使用。

    为什么不在字段上直接写 required：pydantic 的必填校验发生在开关之前，
    关闭水印的请求永远等不到分支判断就被 422 挡在门外（实测 90x MB 的文件
    传完 32 个分块后卡在这里，两次分发都失败）。开关语义必须用模型级校验表达。
    """
    enabled = bool(getattr(model, "watermark_enabled", True))
    text = (getattr(model, "watermark_text", None) or "").strip()
    if enabled and not text:
        raise PydanticCustomError(
            "watermark_text_missing",
            "开启水印时必须提供水印文本（watermark_text）；若这次不需要加水印，请把 watermark_enabled 设为 false",
        )
    if not enabled and text:
        # 显式关闭却仍带着编号，说明上游开关与参数不一致。不报错（避免打断历史调用），
        # 但要清空，防止后续把编号写进清单造成"说没加、其实加了"的假记录。
        model.watermark_text = None


class TaskRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    data_type: str = Field(min_length=1, max_length=32)
    source_path: str = Field(min_length=1, max_length=2048)
    # 关闭水印时不需要水印号：前端可以选「不加防伪水印」，此时 Server1 只会送
    # watermark_enabled=false，不会再造一个假编号。必填性改由下面的模型级校验按
    # watermark_enabled 分支决定，而不是让 pydantic 无条件当必填（那样分发会在
    # 创建任务这一步就 422，连沙箱都进不去）。
    watermark_text: str | None = Field(default=None, max_length=512)
    classification: str = Field(default="", max_length=64)
    precision_decimals: WholeInt | None = Field(default=None, ge=0, le=15)
    precision_mode: str = Field(default="NONE", pattern=r"^(NONE|DECIMALS|DOUGLAS_PEUCKER|RESAMPLE)$")
    douglas_peucker_tolerance: float | None = Field(default=None, gt=0)
    resample_scale: float | None = Field(default=None, gt=0, le=1)
    resample_width: WholeInt | None = Field(default=None, ge=1)
    resample_height: WholeInt | None = Field(default=None, ge=1)
    resample_method: str = Field(default="bilinear", pattern=r"^(nearest|bilinear|cubic|average|lanczos)$")
    processing_profile: str = Field(default="STRICT_PER_MEMBER", pattern=r"^(PERFORMANCE|STRICT_PER_MEMBER)$")
    watermark_mode: str = Field(default="FORENSIC", pattern=r"^(FORENSIC|VISIBLE|HYBRID)$")
    visible_watermark_text: str | None = Field(default=None, max_length=256)
    watermark_enabled: bool = True
    anchor_granularity: str = Field(default="UPLOAD", pattern=r"^(UPLOAD|UNIT|TILE)$")

    @model_validator(mode="after")
    def _watermark_text_matches_switch(self):
        _apply_watermark_switch_rule(self)
        return self


class WatermarkQueryRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    data_type: str = Field(min_length=1, max_length=32)
    classification: str = Field(min_length=1, max_length=64)
    source_path: str = Field(min_length=1, max_length=2048)
    watermark_text: str = Field(min_length=1, max_length=512)


class WatermarkVerifyRequest(WatermarkQueryRequest):
    expected_watermark: str | None = Field(default=None, max_length=512)


# ── 降精度安全范围 ──────────────────────────────────────────────────────────
class PrecisionCheckRequest(BaseModel):
    """干跑校验：客户选的降精度强度是否还会给水印留下载体。"""
    upload_id: str | None = Field(default=None, max_length=128)
    data_type: str | None = Field(default=None, max_length=32)
    # 与水印策略同一口径：校验接口要能分辨「水印开着」和「水印关着」，
    # 否则前端取消水印后仍然收到 blocked=true。
    watermark_enabled: bool = Field(default=True)
    watermark_mode: str = Field(default="FORENSIC", pattern=r"^(FORENSIC|VISIBLE|HYBRID)$")
    precision_mode: str = Field(default="NONE", max_length=32)
    precision_decimals: WholeInt | None = Field(default=None, ge=0, le=15)
    douglas_peucker_tolerance: float | None = Field(default=None, gt=0)
    resample_scale: float | None = Field(default=None, gt=0, le=1)
    resample_width: WholeInt | None = Field(default=None, ge=1)
    resample_height: WholeInt | None = Field(default=None, ge=1)
    width: WholeInt | None = Field(default=None, ge=1)
    height: WholeInt | None = Field(default=None, ge=1)
    features: WholeInt | None = Field(default=None, ge=0)


# ── 数据外借 / 归还 ─────────────────────────────────────────────────────────
class LoanCreateRequest(BaseModel):
    result_id: str = Field(min_length=1, max_length=128)
    project_id: str = Field(min_length=1, max_length=128)
    borrower: str = Field(min_length=1, max_length=128)
    user_id: str = Field(default="", max_length=128)
    due_at: str = Field(min_length=10, max_length=64, description="归还期限，ISO 8601")
    classification: str = Field(default="", max_length=64)
    data_type: str = Field(default="", max_length=32)
    purpose: str = Field(default="", max_length=512)
    operator: str = Field(default="", max_length=128)
    expected_watermark: str = Field(default="", max_length=512)
    carrier_path: str = Field(default="", max_length=2048,
                             description="选填；给了就登记载体哈希，归还时可比对")
    due_days: WholeInt | None = Field(default=None, ge=1, le=3650,
                                 description="选填；给了就按 now+due_days 覆盖 due_at")


class LoanCheckInRequest(BaseModel):
    """归还核验：把载体放进来，系统现场检水印 + 比对借出记录 + 判期限。"""
    source_path: str = Field(min_length=1, max_length=2048)
    data_type: str = Field(min_length=1, max_length=32)
    classification: str = Field(min_length=1, max_length=64)
    user_id: str = Field(default="", max_length=128)
    watermark_text: str | None = Field(default=None, max_length=512,
                                       description="选填；不给就用借出单登记的水印号")
    operator: str = Field(default="", max_length=128)
    note: str = Field(default="", max_length=1024)


class UploadCreateRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    data_type: str = Field(min_length=1, max_length=32)
    classification: str = Field(min_length=1, max_length=64)
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: WholeInt = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    chunk_size: WholeInt = Field(default=UPLOAD_CHUNK_RECOMMENDED, ge=1024, le=UPLOAD_CHUNK_MAX)
    # 可选：显式指定落在哪个沙箱卷。省略则由后端按容量+优先级自动选。
    # 指定的卷放不下时返回 507 而不是退回其他卷 —— 调用方点名要的卷不能静默换掉。
    volume_id: str | None = Field(default=None, max_length=64)


class V1TaskRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    upload_id: str = Field(min_length=1, max_length=128)
    data_type: str = Field(min_length=1, max_length=32)
    classification: str = Field(min_length=1, max_length=64)
    # 与 TaskRequest 同一契约：watermark_enabled=false 时无需 watermark_text。
    watermark_text: str | None = Field(default=None, max_length=512)
    precision_decimals: WholeInt | None = Field(default=None, ge=0, le=15)
    precision_mode: str = Field(default="NONE", pattern=r"^(NONE|DECIMALS|DOUGLAS_PEUCKER|RESAMPLE)$")
    douglas_peucker_tolerance: float | None = Field(default=None, gt=0)
    resample_scale: float | None = Field(default=None, gt=0, le=1)
    resample_width: WholeInt | None = Field(default=None, ge=1)
    resample_height: WholeInt | None = Field(default=None, ge=1)
    resample_method: str = Field(default="bilinear", pattern=r"^(nearest|bilinear|cubic|average|lanczos)$")
    processing_profile: str = Field(default="STRICT_PER_MEMBER", pattern=r"^(PERFORMANCE|STRICT_PER_MEMBER)$")
    watermark_mode: str = Field(default="FORENSIC", pattern=r"^(FORENSIC|VISIBLE|HYBRID)$")
    visible_watermark_text: str | None = Field(default=None, max_length=256)
    watermark_enabled: bool = True
    anchor_granularity: str = Field(default="UPLOAD", pattern=r"^(UPLOAD|UNIT|TILE)$")

    @model_validator(mode="after")
    def _watermark_text_matches_switch(self):
        _apply_watermark_switch_rule(self)
        return self


class ExportCreateRequest(BaseModel):
    anchor_granularity: str | None = Field(default=None, pattern=r"^(UPLOAD|UNIT|TILE)$")


def _service_token() -> str:
    return os.environ.get("GEO_SECURITY_SERVER1_TOKEN", "server1-local")


def _require_service_token(value: str | None) -> None:
    if not value or not secrets.compare_digest(value, _service_token()):
        raise HTTPException(status_code=401, detail="service authentication required")


def _admin_token() -> str:
    return os.environ.get("GDS_ADMIN_TOKEN", "")


def _resolve_sandbox_target(file_id: str, store=None) -> tuple[Path, dict | None]:
    """把沙箱页传来的 file_id 解析成 (可直接读的数据文件, 登记记录)。

    接受三种形式：
      · `file_<artifact_id>` —— 系统内已登记成果，取 **output**（成品 .gpkg）
      · 绝对路径到成品（`V:\\...\\results\\...\\hubei.gpkg`）—— 直接使用
      · 绝对路径到归档（`...\\hubei-260801-free.gpkg.zip`）—— **先解包再定位 .gpkg**

    第三点是 2026-10-22 修的缺陷：服务器1 传的是**上传原件 zip 的路径**，
    此前绝对路径直接返回、不做解包，导致 gpkg_extract 拿到 zip
    报 `OperationalError: unable to open database file`（GW-422）。
    """
    cleaned = (file_id or "").strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="fileId must not be empty")

    # 形如盘符开头视为绝对路径
    if len(cleaned) > 2 and cleaned[1] == ":":
        raw = Path(cleaned)
        return _locate_readable_payload(raw, None), None

    artifact_id = cleaned[5:] if cleaned.startswith("file_") else cleaned
    if store is None:
        raise HTTPException(status_code=500, detail="artifact store unavailable")

    record = store.get_artifact_record(artifact_id)
    if not record:
        # 本系统不认识这个编号 —— 可能是服务器1 自己的 fileId。
        # 查外部编号映射表（由 POST /api/v1/external-file-ids 登记）。
        mapped = store.resolve_external_file_id(cleaned)
        if mapped:
            record = store.get_artifact_record(mapped)
        if record:
            artifact_id = record["artifact_id"]
    if not record:
        raise HTTPException(
            status_code=404,
            detail=(f"artifact not found: {artifact_id}. "
                    "可接受的 fileId：① candidates 返回的 file_<artifact_id>；"
                    "② 已登记的外部编号（POST /api/v1/external-file-ids）；"
                    "③ 服务器2 可见的绝对路径（成品 .gpkg 或上传原件 .zip）"))
    # 登记成果一律取 output（成品），绝不取 source（可能是 zip 或解包前的原件）
    carrier = Path(str(record.get("output") or record.get("path") or ""))
    return _locate_readable_payload(carrier, record), record


_SANDBOX_INBOX = Path(os.environ.get("GDS_SANDBOX_INBOX",
                                     r"V:\geo_data_security\sandbox_inbox"))

_SUFFIX_DATA_TYPE = {
    ".gpkg": ("GPKG", "DLG"),
    ".shp": ("SHP", "DLG_LINEWORK"),
    ".geojson": ("GEOJSON", "THEMATIC_VECTOR"),
    ".json": ("GEOJSON", "THEMATIC_VECTOR"),
    ".tif": ("GEOTIFF", "ORTHOIMAGE"),
    ".tiff": ("GEOTIFF", "ORTHOIMAGE"),
    ".img": ("GEOTIFF", "ORTHOIMAGE"),
    ".dem": ("DEM", "THEMATIC_RASTER"),
}


def _guess_type(path: Path, record: dict | None) -> tuple[str, str]:
    """推断 (data_type, classification)。

    优先用登记记录；无登记（外部攻击样/上传原件）时按后缀判断。
    之前 detect/verify 对无登记文件一律硬编码 GPKG，把 .tif 当 GPKG 送进
    gpkg_extract，报 `DatabaseError: file is not a database`（真实缺陷）。
    """
    if record:
        dt = record.get("data_type")
        cl = record.get("classification")
        if dt:
            return str(dt), str(cl or "DLG")
    return _SUFFIX_DATA_TYPE.get(path.suffix.lower(), ("GPKG", "DLG"))


_PAYLOAD_IGNORED = {".gds_extract_complete", "payload_inventory.json",
                    "manifest.json", "README", ".DS_Store"}


def _hash_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    """分块算 SHA-256，避免把大文件整个读进内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _locate_readable_payload(path: Path, record: dict | None) -> Path:
    """把一个用户给的路径收敛成「真正能被 SQLite/rasterio 打开的文件」。

    规则（按优先级）：
      1. 本来就是可读的数据文件（.gpkg/.shp/.tif/...）→ 原样返回
      2. 是 zip 归档 → 解包到同级 payload/ 目录，再在里面找唯一的数据文件
      3. 是目录（已解包的 payload/）→ 在里面找唯一数据文件
    找不到唯一候选时抛 422，并把看到的内容列出来，便于对方自行纠正。
    """
    path = Path(path)
    archive_suffixes = {".zip"}
    data_suffixes = {".gpkg", ".shp", ".tif", ".tiff", ".img", ".dem", ".osgb"}

    def _pick(root: Path) -> Path | None:
        cands = [p for p in root.rglob("*")
                 if p.is_file() and p.name not in _PAYLOAD_IGNORED
                 and p.suffix.lower() in data_suffixes]
        if not cands:
            return None
        # 同一目录下同名 .shp 优先（矢量）；否则取最大的那个
        shp = [p for p in cands if p.suffix.lower() == ".shp"]
        if shp:
            return sorted(shp, key=lambda p: -p.stat().st_size)[0]
        return sorted(cands, key=lambda p: -p.stat().st_size)[0]

    if path.suffix.lower() in archive_suffixes:
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"archive not found: {path}")
        # 复用上传链路的解包逻辑：带路径穿越防护 + 完成标记 + 清单
        root = _prepare_task_source(path)
        picked = _pick(Path(root))
        if picked is None:
            raise HTTPException(
                status_code=422,
                detail=(f"ZIP 已解包到 {root}，但其中没有可读的矢量/栅格文件"
                        "（支持 .gpkg/.shp/.tif/.img）。请确认归档内容。"))
        return picked

    if path.is_dir():
        picked = _pick(path)
        if picked is None:
            raise HTTPException(status_code=422,
                                detail=f"目录中没有可读数据文件: {path}")
        return picked

    return path


def _parse_watermark_int(text: str) -> int:
    cleaned = text.strip()
    try:
        if cleaned.lower().startswith("0x"):
            value = int(cleaned, 16)
        elif re.fullmatch(r"[0-9A-Fa-f]{1,8}", cleaned) and any(ch in "ABCDEFabcdef" for ch in cleaned):
            # Backward-compatible acceptance of existing bare 8-digit hex IDs.
            value = int(cleaned, 16)
        else:
            value = int(cleaned)
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("out of 32-bit range")
        return value
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"watermark_text must be a 32-bit integer, 0x hex, or bare hex ID, got: {text!r}") from exc


def _watermark_id_from_text(text: str) -> str:
    """把文本水印号稳定地映射成 32 位整数编号。

    栅格与模型类水印只能承载 32 位整数（TrustMark / DCT 位串），而 S1 传下来的是
    "DLG-prj_xxx" 这类文本编号。声明与实际内容不一致、被判成栅格后，整条分发会卡在
    "watermark_text must be a 32-bit integer"，所以这里按哈希派生一个固定值：
    同一份文本永远得到同一个号码，成果清单里记录的也是这个号码，核验时对得上。
    """
    digest = hashlib.sha256(text.strip().encode("utf-8")).digest()
    value = int.from_bytes(digest[:4], "big") or 1
    return f"{value:08X}"


def _is_numeric_watermark(text: str) -> bool:
    try:
        _parse_watermark_int(text)
        return True
    except HTTPException:
        return False


def _validate_windows_filename(filename: str) -> str:
    """Reject names Windows cannot create; preserve valid Unicode names unchanged."""
    invalid = set('<>:"/\\\\|?*')
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=422, detail="invalid filename")
    if any(ord(ch) < 32 for ch in filename) or any(ch in invalid for ch in filename):
        raise HTTPException(status_code=422, detail="invalid Windows filename; send UTF-8 filename without reserved characters")
    if filename.endswith((" ", ".")):
        raise HTTPException(status_code=422, detail="invalid Windows filename; trailing spaces or dots are not allowed")
    stem = filename.split(".", 1)[0].upper()
    if stem in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise HTTPException(status_code=422, detail="invalid Windows reserved filename")
    return filename


# --------------------------------------------------------------- payload helpers
# The front end always uploads one ZIP archive; the archive name says nothing about
# the payload, so every entry point resolves the extracted payload and validates the
# real carrier files before dispatching a handler.
TYPE_ALIASES = {"DLG": "SHP", "DOM": "GEOTIFF", "MESH": "OSGB", "THEMATIC_VECTOR": "SHP"}

_PAYLOAD_KIND_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "OSGB": (".osgb",),
    "SHP": (".shp",),
    "GPKG": (".gpkg",),
    "GEOJSON": (".geojson", ".json"),
    "RASTER": (".tif", ".tiff"),
    "TEXTURE": (".jpg", ".jpeg", ".png"),
    "POINT_CLOUD": (".las", ".laz", ".e57", ".ply", ".pcd", ".xyz", ".pts"),
}

_ALLOWED_PAYLOAD_KINDS: dict[str, set[str]] = {
    "OSGB": {"OSGB", "TEXTURE"},
    "TEXTURES": {"TEXTURE"},
    "OSGB_TEXTURES": {"TEXTURE"},
    "SHP": {"SHP", "GPKG"},
    "GEOJSON": {"GEOJSON"},
    "GEOTIFF": {"RASTER"},
    "GTIFF": {"RASTER"},
    "DEM": {"RASTER"},
    "POINT_CLOUD": {"POINT_CLOUD"},
}

_RASTER_DATA_TYPES = {"GEOTIFF", "GTIFF", "DEM"}


def normalize_data_type(data_type: str) -> str:
    normalized = data_type.upper()
    return TYPE_ALIASES.get(normalized, normalized)


_BOOKKEEPING_FILES = {".gds_extract_complete", "payload_inventory.json"}


def _payload_kinds(root: Path) -> dict[str, list[str]]:
    """Classify the real carrier files under ``root`` (a directory) or at ``root`` (a single file)."""
    base = Path(root)
    candidates = [base] if base.is_file() else sorted(base.rglob("*"))
    found: dict[str, list[str]] = {}
    for path in candidates:
        if not path.is_file() or path.name in _BOOKKEEPING_FILES:
            continue
        suffix = path.suffix.lower()
        for kind, extensions in _PAYLOAD_KIND_EXTENSIONS.items():
            if suffix in extensions:
                name = path.name if base.is_file() else str(path.relative_to(base))
                found.setdefault(kind, []).append(name)
                break
    return found


def _payload_report(root: Path) -> str:
    kinds = _payload_kinds(root)
    if not kinds:
        return "no recognised data file"
    return "; ".join(f"{kind}:{len(names)}[{', '.join(names[:3])}]" for kind, names in sorted(kinds.items()))


def _single_band_raster(root: Path) -> bool:
    """载荷里的栅格是否单波段（高程）而不是多波段正射影像。"""
    rasters = payload_rasters(root)
    if not rasters:
        return False
    try:
        import rasterio
        with rasterio.open(rasters[0]) as dataset:
            return int(dataset.count) == 1
    except Exception:
        return False


def _load_manifest(output: Path) -> dict:
    """读取成果清单。输出可能是 payload 子目录或具体文件，逐级向上找一层。"""
    for candidate in (output / "manifest.json", output.parent / "manifest.json",
                      output.parent.parent / "manifest.json"):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except ValueError:
                return {}
    return {}


_DEFAULT_CLASSIFICATION = {
    "SHP": "THEMATIC_VECTOR",
    "GPKG": "THEMATIC_VECTOR",
    "GEOPACKAGE": "THEMATIC_VECTOR",
    "GEOJSON": "THEMATIC_VECTOR",
    "DEM": "ELEVATION_DEM",
    "GEOTIFF": "REMOTE_SENSING_RASTER",
    "GTIFF": "REMOTE_SENSING_RASTER",
    "OSGB": "OBLIQUE_MESH",
    "TEXTURES": "OBLIQUE_MESH",
    "OSGB_TEXTURES": "OBLIQUE_MESH",
}


def _reconcile_data_type(declared: str, root: Path, classification: str = "") -> str:
    """按载荷实际内容校正声明的数据类型，跨大类也校正。

    S1 按扩展名映射格式：ZIP 承载的 DEM 可能被声明成 SHP，栅格被当成矢量，
    整条分发就在「导入沙箱」这一步失败——同一份数据声明对了就成功、声明错了就失败，
    表现为时好时坏。这里按实际内容把大类纠正过来。
    """
    normalized = normalize_data_type(declared)
    if normalized in {"OSGB", "OSGB_TEXTURES", "TEXTURES", "POINT_CLOUD"}:
        return normalized
    present = set(_payload_kinds(Path(root)))
    if not present:
        return normalized
    raster = "RASTER" in present
    vector = bool(present & {"SHP", "GPKG", "GEOJSON"})
    if "OSGB" in present and normalized != "OSGB":
        return "OSGB"
    if raster and not vector:
        if normalized in _RASTER_DATA_TYPES:
            return normalized
        label = (classification or "").upper()
        if "ELEVATION" in label or "DEM" in label:
            return "DEM"
        if "ORTHO" in label or "DOM" in label:
            return "GEOTIFF"
        return "DEM" if _single_band_raster(Path(root)) else "GEOTIFF"
    if vector and not raster and normalized not in {"SHP", "GPKG", "GEOPACKAGE", "GEOJSON", "DLG"}:
        for candidate in ("GPKG", "SHP", "GEOJSON"):
            if candidate in present:
                return candidate
    return normalized


def validate_payload_kind(data_type: str, root: Path, label: str) -> None:
    """Reject a payload whose real content does not match the declared data type.

    A staged ZIP payload must contain a recognised carrier file. A single uploaded file is
    only rejected when its extension is recognised *and* belongs to another data class, so
    exotic raw carrier formats still reach their own handler (which reports its own error).
    """
    normalized = normalize_data_type(data_type)
    allowed = _ALLOWED_PAYLOAD_KINDS.get(normalized)
    if allowed is None:
        return
    kinds = _payload_kinds(root)
    expected = "/".join(sorted(allowed))
    if not kinds:
        if Path(root).is_file():
            suffix = Path(root).suffix.lower()
            if suffix in {".tif", ".tiff"} and normalized in _RASTER_DATA_TYPES:
                return
            return
        raise ValueError(f"{label}: no recognised data file for data_type={normalized}; expected {expected}")
    if not set(kinds) & allowed:
        raise ValueError(
            f"{label}: payload does not match data_type={normalized}; detected {', '.join(sorted(kinds))}, expected {expected}"
        )


def payload_rasters(root: Path) -> list[Path]:
    """GeoTIFFs to integrity-scan: every raster inside a payload directory, or the file itself."""
    base = Path(root)
    candidates = [base] if base.is_file() else sorted(base.rglob("*"))
    return [p for p in candidates if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]


def raster_carrier_shortfall(detail: dict, data_type: str | None = None) -> str | None:
    """Return why a raster cannot carry the tiled watermark, or ``None`` when it can.

    Historical failure this gate used to encode: ``embed`` wrote a flush edge tile that overlapped
    the origin tile below 1024 px on either side, so both code words were destroyed while the embed
    still "succeeded", and the task failed minutes later with an opaque verification error.
    ``embed`` now skips overlapping carriers and encodes sub-carrier rasters on a zero-padded canvas
    (writing back only the original extent), and ``extract`` replays the recorded carrier origins,
    so sub-1024 rasters are genuinely supported. Only sizes below the engine floor are rejected,
    which keeps the cheap up-front rejection without pretending the task would work.
    DEM uses its own per-tile carrier and is therefore exempt.
    """
    if normalize_data_type(data_type or "") == "DEM":
        return None
    minimum = int(os.environ.get("GDS_MIN_RASTER_SIDE", "128"))
    width, height = detail.get("width"), detail.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    if min(width, height) >= minimum:
        return None
    return (f"{width}x{height} is below the watermark carrier minimum "
            f"{minimum}x{minimum} on both sides")


def _partial_scan_result(detail: dict) -> dict:
    """Normalize a bounded raster scan into an explicit incomplete-input record."""
    total = int(detail.get("blocks_total", detail.get("blocks_read", 0)))
    read = int(detail.get("blocks_read", 0))
    failed = detail.get("failed_blocks") or ([] if not detail.get("failed_block") else [detail["failed_block"]])
    return {
        "source_integrity": "INCOMPLETE" if read < total or failed else "COMPLETE",
        "blocks_total": total,
        "blocks_read": read,
        "blocks_missing": max(0, total - read),
        "failed_blocks": failed,
    }


def _partial_mode_enabled() -> bool:
    # Accepting an incomplete source is the platform's default behaviour: the readable region is
    # processed and the artifact is delivered as a partial result under manual review. Set
    # GDS_PARTIAL_RASTER_MODE=0 to make the platform refuse such a source instead.
    return os.environ.get("GDS_PARTIAL_RASTER_MODE", "1") == "1"


_SCAN_CACHE: dict[tuple, dict] = {}
_SCAN_CACHE_LOCK = threading.Lock()
_SCAN_CACHE_MAX = 32


def _scan_raster_inprocess(path: Path) -> dict:
    """进程内逐块读栅格，返回与沙箱 `raster_read_scan` 同构的完整性报告。

    为什么必须有：沙箱未启用时 `sandbox_compute.enabled()` 为 False，调用方会直接拿 `{}`，
    于是 `source_integrity` 缺失 → 处理时走严格嵌入路径 → 源文件尾部坏块直接
    RasterioIOError 把整条任务打挂（实测 2.73 GB 正射图底部 6+ 个块不可读）。
    这个扫描必须与沙箱版本同口径：逐块读、坏块记录、给 INCOMPLETE。
    """
    import rasterio
    with rasterio.open(path) as dataset:
        windows = list(dataset.block_windows(1))
        failed = []
        for ordinal, (_, window) in enumerate(windows):
            try:
                dataset.read(window=window)
            except Exception as exc:  # noqa: BLE001
                failed.append({"ordinal": ordinal, "row": int(window.row_off),
                               "col": int(window.col_off),
                               "width": int(window.width), "height": int(window.height),
                               "error": f"{type(exc).__name__}: {exc}"})
        return {"driver": dataset.driver, "width": dataset.width, "height": dataset.height,
                "count": dataset.count, "dtype": dataset.dtypes[0],
                "blocks_total": len(windows), "blocks_read": len(windows) - len(failed),
                "failed_blocks": failed,
                "source_integrity": "INCOMPLETE" if failed else "COMPLETE"}


def _cached_raster_scan(path: Path, timeout_s: int) -> dict:
    """One block scan per (file, size, mtime): the ingest check and the task pre-flight share it.

    Scanning a multi-gigabyte raster costs seconds to minutes, and the frontend may poll the ingest
    integrity route, so the verdict is cached and invalidated by size/mtime.
    """
    try:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        key = (str(path), 0, 0)
    with _SCAN_CACHE_LOCK:
        cached = _SCAN_CACHE.get(key)
    if cached is not None:
        return cached
    detail = (sandbox_compute.run("raster_read_scan", str(path), timeout_s=timeout_s)
              if sandbox_compute.enabled() else _scan_raster_inprocess(path))
    if isinstance(detail, dict):
        detail = {**detail, "size_bytes": key[1]}
        with _SCAN_CACHE_LOCK:
            if len(_SCAN_CACHE) >= _SCAN_CACHE_MAX:
                _SCAN_CACHE.pop(next(iter(_SCAN_CACHE)))
            _SCAN_CACHE[key] = detail
    return detail


def app_tile_units(item: Path, root: Path) -> list[dict]:
    """把一个栅格按内部块切分成最小上链单元（导出路径用）。

    与 `ProcessingService._tile_units` 同一口径：读真实 block_windows；
    读不了就退回整文件一个单元，绝不假装切过。
    """
    relative = str(item.relative_to(root)).replace("\\", "/")
    try:
        import rasterio
        with rasterio.open(item) as src:
            units = []
            # rasterio >=1.3 产出 ((row_block, col_block), Window)；旧版只产出 Window。
            for block_index, window in src.block_windows(1):
                rb, cb = block_index if isinstance(block_index, tuple) else (block_index, 0)
                if hasattr(window, "row_off"):
                    row_off, col_off = window.row_off, window.col_off
                    view = window
                else:
                    row_off, col_off = window
                    view = rasterio.windows.Window(col_off, row_off,
                                                   src.block_shapes[0][1], src.block_shapes[0][0])
                block = src.read(1, window=view)
                units.append({
                    "unit": f"{relative}#tile{rb}_{cb}@r{row_off}c{col_off}",
                    "sha256": hashlib.sha256(block.tobytes()).hexdigest(),
                })
            if units:
                return units
    except Exception:
        pass
    return [{"unit": relative, "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}]


def _carrier_stats(upload: dict) -> dict[str, Any]:
    """读一个已上传载荷的真实载体统计，供降精度安全范围计算用。

    返回 ``width/height``（栅格）、``features``（矢量）、``points``（点云）；
    取不到的项为 ``None``——绝不猜，猜出来的范围会把客户带到水印失败那条路上。
    """
    stats: dict[str, Any] = {"width": None, "height": None, "features": None, "points": None}
    raw = upload.get("source_path") or ""
    if not raw:
        return stats
    payload = Path(raw)
    archive = Path(raw)
    try:
        if archive.is_file() and archive.suffix.lower() == ".zip":
            extracted = zip_payload_root(archive)
            if extracted.is_dir():
                payload = extracted
    except Exception:
        return stats
    kind = (upload.get("data_type") or "").upper()

    if kind in {"GEOTIFF", "GTIFF", "TIF", "TIFF", "DOM", "DEM"}:
        rasters = sorted(p for p in payload.rglob("*") if p.is_file()
                         and p.suffix.lower() in {".tif", ".tiff"}) if payload.is_dir() else \
                  ([payload] if payload.suffix.lower() in {".tif", ".tiff"} else [])
        for item in rasters:
            try:
                detail = _cached_raster_scan(item, 120)
            except Exception:
                detail = {}
            w = detail.get("width") or detail.get("src_width")
            h = detail.get("height") or detail.get("src_height")
            if not (w and h):
                try:
                    import rasterio
                    with rasterio.open(item) as src:
                        w, h = src.width, src.height
                except Exception:
                    continue
            stats["width"], stats["height"] = int(w), int(h)
            break
    elif kind in {"SHP", "DLG"}:
        try:
            import shapefile
            shapes = sorted(p for p in payload.rglob("*.shp") if p.is_file())
            if shapes:
                with shapefile.Reader(str(shapes[0])) as reader:
                    stats["features"] = len(reader)
        except Exception:
            pass
    elif kind in {"GEOJSON"}:
        try:
            text = payload.read_text(encoding="utf-8") if payload.is_file() else \
                   next(p for p in payload.rglob("*.geojson") if p.is_file()).read_text(encoding="utf-8")
            stats["features"] = len(json.loads(text).get("features") or [])
        except Exception:
            pass
    elif kind == "POINT_CLOUD":
        try:
            from .point_cloud import discover as _pc_discover
            files = _pc_discover(payload)
            if files:
                with files[0].open(encoding="utf-8", errors="replace") as stream:
                    stats["points"] = sum(1 for line in stream if line.strip())
        except Exception:
            pass
    return stats


def _task_resume_payload(request) -> str:
    """把「重放这个任务所需的全部入参」序列化进 tasks.resume_payload。

    只在重启中断后用于一键重提。刻意只收重放必需的白名单字段，
    绝不把 token、绝对路径以外的凭据类信息落库。
    watermark_text 是业务编号（8 位十六进制），不是凭据，必须保留，
    否则重提后水印号会变，历史成果与重提成果对不上。
    """
    fields = (
        "project_id", "user_id", "upload_id", "data_type", "classification",
        "watermark_text", "watermark_enabled", "watermark_mode", "visible_watermark_text",
        "anchor_granularity", "processing_profile",
        "precision_decimals", "precision_mode", "douglas_peucker_tolerance",
        "resample_scale", "resample_width", "resample_height", "resample_method",
    )
    payload: dict = {}
    for name in fields:
        if not hasattr(request, name):
            continue
        value = getattr(request, name)
        if value is None:
            continue
        payload[name] = value
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return ""


def _task_view(task: dict | None) -> dict | None:
    """Task rows keep the source-integrity warning as a JSON column; hand the frontend a list."""
    if task is None:
        return None
    view = dict(task)
    raw = view.get("warnings")
    if isinstance(raw, str):
        try:
            view["warnings"] = json.loads(raw) if raw.strip() else []
        except ValueError:
            view["warnings"] = []
    elif raw is None:
        view["warnings"] = []
    view["source_integrity"] = view.get("source_integrity") or None
    return view


def probe_raster_readability(paths: list[Path], report, data_type: str | None = None) -> list[dict]:
    """Read every block of each raster inside the sandbox before a long watermark run.

    Opening a GeoTIFF, or reading one window, does not prove the whole raster is
    readable: a truncated tile only fails when the watermark runner walks the tile
    grid, which used to surface as an opaque ``RasterioIOError`` after minutes.
    """
    if os.environ.get("GDS_PREFLIGHT_RASTER_SCAN", "1") != "1" or not sandbox_compute.enabled():
        return []
    limit = int(os.environ.get("GDS_PREFLIGHT_RASTER_MAX", "8"))
    timeout_s = int(os.environ.get("GDS_PREFLIGHT_SCAN_TIMEOUT", "3600"))
    ordered = sorted(paths, key=lambda p: p.stat().st_size, reverse=True)
    results: list[dict] = []
    for path in ordered[:limit]:
        try:
            detail = _cached_raster_scan(path, timeout_s)
        except RuntimeError as exc:
            report("PAYLOAD_INTEGRITY_FAILED", path=str(path), reason=str(exc)[:400])
            raise ValueError(
                f"source raster {path.name} is truncated or unreadable ({str(exc)[:300]}); "
                "re-export the source data and upload a new archive - re-running the same archive will fail again"
            ) from exc
        shortfall = raster_carrier_shortfall(detail if isinstance(detail, dict) else {}, data_type)
        if shortfall:
            report("PAYLOAD_CARRIER_UNSUPPORTED", path=str(path), reason=shortfall)
            raise ValueError(
                f"source raster {path.name} cannot carry a watermark: {shortfall}; "
                "a raster below 128 px on either side has too few pixels for a verifiable watermark - "
                "export a larger scene or a higher resolution and upload it again"
            )
        integrity = _partial_scan_result(detail if isinstance(detail, dict) else {})
        if integrity["source_integrity"] == "INCOMPLETE" and not _partial_mode_enabled():
            report("PAYLOAD_INTEGRITY_FAILED", path=str(path), reason=json.dumps(integrity, ensure_ascii=True))
            raise ValueError(
                f"source raster {path.name} is truncated or unreadable; "
                "re-export the source data and upload a new archive - re-running the same archive will fail again"
            )
        results.append({"file": str(path.name), **(detail if isinstance(detail, dict) else {}), **integrity})
    if len(ordered) > limit:
        results.append({"skipped": len(ordered) - limit})
    return results


def _validate_archive(path: Path) -> None:
    """Validate an uploaded ZIP as an archive before any handler touches its payload."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if not members:
                raise HTTPException(status_code=422, detail="ZIP payload is empty")
            names = [member.filename for member in members]
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail="invalid ZIP payload") from exc
    for name in names:
        if name.startswith("/") or ".." in Path(name).parts or re.match(r"^[A-Za-z]:", name):
            raise HTTPException(status_code=422, detail=f"ZIP member path is not allowed: {name}")


def zip_payload_root(archive: Path) -> Path:
    """The controlled extraction directory that belongs to an uploaded archive."""
    return Path(archive).parent / "payload"


# 核验器（sqlite / rasterio / json）要的是数据文件本身，不是解包目录。
_FILE_BASED_TYPES = {"GPKG", "GEOPACKAGE", "GEOJSON", "GEOTIFF", "GTIFF"}
_FILE_SUFFIXES = {
    "GPKG": {".gpkg"},
    "GEOPACKAGE": {".gpkg"},
    "GEOJSON": {".geojson", ".json"},
    "GEOTIFF": {".tif", ".tiff"},
    "GTIFF": {".tif", ".tiff"},
}


def _payload_data_file(normalized: str, directory: Path) -> Path:
    """从解包目录里挑出真正的数据文件。

    上传一律是 ZIP，解包后是「一个大目录里放数据文件」。把目录直接交给
    sqlite 或 rasterio 会报 unable to open database file，归还核验里这个异常
    会被误读成「载体内没有我们的水印」，最终落成「非本系统成果」。
    """
    suffixes = _FILE_SUFFIXES[normalized]
    files = sorted(p for p in Path(directory).rglob("*")
                   if p.is_file() and p.suffix.lower() in suffixes)
    if len(files) == 1:
        return files[0]
    if not files:
        raise HTTPException(status_code=422,
                            detail=f"解包目录里没有可核验的 {normalized} 数据文件：{directory}")
    names = "、".join(p.name for p in files[:5])
    raise HTTPException(status_code=409,
                        detail=f"解包目录里有 {len(files)} 个 {normalized} 数据文件，无法确定核验对象：{names}")


_VECTOR_DECLARED = {"SHP", "GPKG", "GEOPACKAGE", "GEOJSON"}
_VECTOR_SKIP_NAMES = {"manifest.json", "payload_inventory.json"}


def _vector_type_of_file(path: Path) -> str | None:
    """按文件名判断矢量容器。清单和盘点文件不是成果。"""
    if path.name.lower() in _VECTOR_SKIP_NAMES:
        return None
    suffix = path.suffix.lower()
    if suffix == ".gpkg":
        return "GPKG"
    if suffix in {".geojson", ".json"}:
        return "GEOJSON"
    if suffix == ".shp":
        return "SHP"
    return None


def carrier_type_for_file(declared: str, path: Path) -> str:
    """借出单登记的是借出时的容器。格式转换后按归还文件本身选择提取器。"""
    normalized = normalize_data_type(declared)
    if normalized not in _VECTOR_DECLARED or not Path(path).is_file():
        return normalized
    actual = _vector_type_of_file(Path(path))
    if actual is None:
        return normalized
    if normalized == "GEOPACKAGE" and actual == "GPKG":
        return "GEOPACKAGE"
    return actual


def _vector_files(directory: Path) -> list[Path]:
    return sorted(
        item for item in Path(directory).rglob("*")
        if item.is_file() and _vector_type_of_file(item)
    )


def resolve_declared_carrier(declared: str, directory: Path) -> tuple[Path, str]:
    """目录里没有声明格式时，改用同目录里唯一的矢量成果。"""
    normalized = normalize_data_type(declared)
    if normalized in _FILE_BASED_TYPES:
        try:
            found = _payload_data_file(normalized, directory)
            return found, carrier_type_for_file(normalized, found)
        except HTTPException as exc:
            if exc.status_code != 422 or normalized not in _VECTOR_DECLARED:
                raise
    elif normalized == "SHP":
        if any(item.is_file() and item.suffix.lower() == ".shp" for item in Path(directory).rglob("*.shp")):
            return directory, "SHP"
    else:
        return directory, normalized
    converted = _vector_files(directory)
    if len(converted) == 1:
        kind = _vector_type_of_file(converted[0]) or normalized
        if kind == "SHP":
            return converted[0].parent, "SHP"
        return converted[0], kind
    if not converted:
        raise HTTPException(status_code=422,
                            detail=f"解包目录里没有可核验的矢量数据文件：{directory}")
    names = "、".join(item.name for item in converted[:5])
    raise HTTPException(status_code=409,
                        detail=f"解包目录里有 {len(converted)} 个矢量数据文件，无法确定核验对象：{names}")


def _prepare_task_source(source: Path, *, task_id: str | None = None) -> Path:
    """Prepare an uploaded archive in the worker thread, never in the HTTP request."""
    source = Path(source)
    if source.suffix.lower() != ".zip":
        return source
    extracted = source.parent / "payload"
    marker = extracted / ".gds_extract_complete"
    task_context = f" task={task_id[:12]}" if task_id else ""
    if marker.is_file():
        print(f"[TIMING] extract cache-hit {source.name}{task_context}", flush=True)
        return extracted
    lock = source.parent / ".extract.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError("upload archive extraction already in progress") from exc
    staging = source.parent / f".payload_{secrets.token_hex(6)}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        if extracted.exists():
            shutil.rmtree(extracted)
        _t_extract = time.perf_counter()
        with zipfile.ZipFile(source) as archive:
            inventory = _extract_archive(archive, staging)
        print(f"[TIMING] extract {time.perf_counter() - _t_extract:.1f}s "
              f"members={len(inventory)} zip={source.stat().st_size / 2**30:.2f}GB{task_context}", flush=True)
        if extracted.exists():
            shutil.rmtree(extracted)
        staging.replace(extracted)
        marker.write_text("complete\n", encoding="ascii")
        # Record what the archive actually contained so later stages - and the audit
        # trail - can tell a complete package from a truncated one.
        (extracted / "payload_inventory.json").write_text(
            json.dumps({"archive": source.name, "archive_bytes": source.stat().st_size, "members": inventory},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        return extracted
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def _repair_mojibake(segment: str) -> str:
    """把 "╚½╟≥" 这类 cp437 乱码还原成中文。

    上游打包时把 GBK 字节写进了文件名，有的还错误地置了 UTF-8 标记，于是名字本身就是
    乱码，光看标记判断不出来。能按 cp437 编回字节、再用 GBK 解出汉字的，就按乱码处理。
    """
    if not any(ord(char) > 127 for char in segment):
        return segment
    try:
        raw = segment.encode("cp437")
    except UnicodeEncodeError:
        return segment
    for encoding in ("gbk", "utf-8"):
        try:
            fixed = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if fixed != segment and any("\u4e00" <= char <= "\u9fff" for char in fixed):
            return fixed
    return segment


def _archive_member_name(member: zipfile.ZipInfo) -> str:
    """还原 ZIP 成员的真实文件名。

    Windows 上打的包常用 GBK 写中文名且不置 UTF-8 标记，zipfile 一律按 cp437 解码，
    中文就变成 "╚½╟≥250├╫╕▀│╠DEM" 并一路带进解包目录、成果目录，最后出现在导出包。
    名字被整体搞坏时还要逐个路径段还原。
    """
    name = member.filename
    if not (member.flag_bits & 0x800) and not any(ord(char) > 127 for char in name):
        return name
    parts = name.replace("\\", "/").split("/")
    return "/".join(_repair_mojibake(part) for part in parts)


def _extract_archive(archive: zipfile.ZipFile, staging: Path) -> list[dict]:
    """按还原后的名字解包，并挡住路径穿越。

    不用 extractall：它按成员原名落盘，中文名在 GBK 包里必然是乱码。
    """
    root = staging.resolve()
    plan: list[tuple[zipfile.ZipInfo, str, Path]] = []
    for member in archive.infolist():
        name = _archive_member_name(member)
        target = (staging / name).resolve()
        if root != target and root not in target.parents:
            raise ValueError("ZIP contains path traversal")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        plan.append((member, name, target))

    # 24 GB / 2.8 万成员的包实测单线程解包要 5 分钟以上，而 E: 是 NVMe（1.7 GB/s 写）
    # 且单核 inflate 有 292 MB/s —— 瓶颈是逐文件的 Python 开销，不是压缩算法。
    # 并行化后同样的包降到 1 分钟量级。
    try:
        workers = max(1, int(os.environ.get("GDS_EXTRACT_WORKERS", "12") or 12))
    except ValueError:
        workers = 12
    workers = min(workers, 32, max(1, len(plan)))
    if workers <= 1 or len(plan) < 200:
        inventory: list[dict] = []
        for member, name, target in plan:
            with archive.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            inventory.append({"name": name, "size": member.file_size, "crc": f"{member.CRC:08x}"})
        return inventory

    # ZipFile 实例内部共享同一个文件句柄，多线程读同一个实例会互相打断 seek，
    # 所以每个工作线程各开一个句柄，用完统一关闭。
    archive_path = getattr(archive, "filename", None)
    local = threading.local()
    handles: list[zipfile.ZipFile] = []
    handles_lock = threading.Lock()

    def _one(item: tuple[zipfile.ZipInfo, str, Path]) -> dict:
        member, name, target = item
        zf = getattr(local, "zf", None)
        if zf is None:
            zf = zipfile.ZipFile(archive_path)
            local.zf = zf
            with handles_lock:
                handles.append(zf)
        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        return {"name": name, "size": member.file_size, "crc": f"{member.CRC:08x}"}

    try:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=workers) as pool:
            inventory = list(pool.map(_one, plan))
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
    return inventory


_HEAVY_TYPES = {"GEOTIFF", "GTIFF", "DOM", "DEM", "OSGB", "MESH", "OSGB_TEXTURES", "TEXTURES", "POINT_CLOUD"}


def _heavy_concurrency() -> int:
    try:
        return max(1, int(os.environ.get("GDS_HEAVY_TASK_CONCURRENCY", "1") or 1))
    except ValueError:
        return 1


# 重任务闸门：影像与倾斜模型的嵌入/校验都是 CPU 密集型，且每一步内部已经开了
# 多线程或多进程。历史行为是执行器同时放 4 条任务进来，每条再各开 4-8 个 worker，
# 实测把 18 核 36 线程的机器打到 100%，单任务耗时反而成倍增加（见 DEPLOY 记录）。
# 这里把「重任务」串行化，小数据（矢量）不受影响。
_HEAVY_GATE = threading.BoundedSemaphore(_heavy_concurrency())


def _run_task(app, request: TaskRequest, source: Path, task_id: str, audit_record) -> None:
    store = app.state.task_store
    store.update_task_status(task_id, "RUNNING")
    store.set_task_progress(task_id, stage="PREPARING", percent=5, detail="准备与校验源数据")
    # 重任务串行：前序重任务没结束时先排队，而不是一起挤进 CPU。
    # 排队状态要显式告诉前端，否则用户看到的是「一直 5%」。
    heavy = normalize_data_type(getattr(request, "data_type", "")) in _HEAVY_TYPES
    acquired = False
    if heavy:
        if not _HEAVY_GATE.acquire(blocking=False):
            store.set_task_progress(task_id, stage="QUEUED_FOR_RESOURCES", percent=5,
                                    detail="排队等待计算资源，前序重任务完成后自动开始")
            _HEAVY_GATE.acquire()
        acquired = True
    try:
        _t_prepare = time.perf_counter()
        prepared_source = _prepare_task_source(source, task_id=task_id)
        print(f"[TIMING] task={task_id[:12]} prepare {time.perf_counter() - _t_prepare:.1f}s", flush=True)
        if prepared_source.is_dir() and prepared_source.name == "payload":
            label = f"extracted payload of {Path(source).name}"
        else:
            label = f"uploaded file {Path(source).name}"
        reconciled = _reconcile_data_type(request.data_type, prepared_source, request.classification)
        if reconciled != normalize_data_type(request.data_type):
            # 声明与实际内容不符时按内容处理，并把这次校正写进审计，避免"同一份数据时好时坏"
            update = {"data_type": reconciled}
            audit_record("PAYLOAD_TYPE_RECONCILED", actor=request.user_id,
                         project_id=request.project_id, declared=request.data_type,
                         used=reconciled, upload=str(Path(source).name))
            try:
                category = parse_classification(request.classification)
            except ValueError:
                category = None
            if category is None or not allowed_for_data_type(category, reconciled):
                # 类型换了，密级也要跟着换成该类型允许的取值，否则成果清单里的元数据自相矛盾
                fallback = _DEFAULT_CLASSIFICATION.get(reconciled.upper(), "")
                if fallback:
                    update["classification"] = fallback
                    audit_record("PAYLOAD_CLASSIFICATION_RECONCILED", actor=request.user_id,
                                 project_id=request.project_id,
                                 declared=request.classification, used=fallback,
                                 data_type=reconciled)
            request = request.model_copy(update=update)
        # 关闭水印时 watermark_text 就是 None（契约如此），这里若还去派生整数水印号，
        # 会在 text.strip() 上抛 AttributeError，任务在 PREPARING 5% 直接失败——
        # 表现为「校验通过、任务却立刻挂」，比 422 更难查。派生只在水印开启时才需要。
        if (getattr(request, "watermark_enabled", True)
                and normalize_data_type(request.data_type) in _RASTER_DATA_TYPES | {"OSGB", "TEXTURES", "OSGB_TEXTURES"}):
            if not _is_numeric_watermark(request.watermark_text):
                # 栅格与模型只能承载 32 位整数水印；文本编号按哈希派生固定号码，保证可复核
                derived = _watermark_id_from_text(request.watermark_text)
                audit_record("WATERMARK_ID_DERIVED", actor=request.user_id,
                             project_id=request.project_id, text=request.watermark_text,
                             watermark_id=derived, data_type=request.data_type)
                request = request.model_copy(update={"watermark_text": derived})
        # Same gates for a ZIP payload and for a directly uploaded carrier file: the error
        # the sender sees must not depend on how the raster was packaged.
        audit_record("PAYLOAD_INSPECTED", actor=request.user_id, project_id=request.project_id,
                     data_type=request.data_type, upload=str(Path(source).name),
                     content=_payload_report(prepared_source))
        validate_payload_kind(request.data_type, prepared_source, label)
        scans: list[dict] = []
        if normalize_data_type(request.data_type) in _RASTER_DATA_TYPES:
            store.set_task_progress(task_id, stage="INTEGRITY_SCAN", percent=15,
                                    detail="逐块读取栅格，检查源完整性")
            scans = probe_raster_readability(payload_rasters(prepared_source), audit_record, request.data_type)
            if scans:
                audit_record("PAYLOAD_RASTER_SCAN", actor=request.user_id, project_id=request.project_id,
                             data_type=request.data_type, scans=scans)
                warnings = [item for item in (source_integrity_warning(scan) for scan in scans) if item]
                app.state.task_store.set_task_integrity(
                    task_id, "INCOMPLETE" if warnings else "COMPLETE", warnings)
                if warnings:
                    # the frontend sees this on the task the moment the pre-flight verdict lands,
                    # while the (long) watermark job is still running
                    audit_record("PAYLOAD_INTEGRITY_WARNING", actor=request.user_id, project_id=request.project_id,
                                 task_id=task_id, data_type=request.data_type,
                                 blocks_total=warnings[0].get("blocks_total"),
                                 blocks_missing=warnings[0].get("blocks_missing"))
        mode = (getattr(request, "precision_mode", "NONE") or "NONE").upper()
        if mode != "NONE":
            store.set_task_progress(task_id, stage="PRECISION_REDUCTION", percent=25,
                                    detail=f"降精度处理（{mode}），在水印嵌入之前执行")
        if getattr(request, "watermark_enabled", True):
            store.set_task_progress(task_id, stage="WATERMARK_EMBEDDING", percent=35,
                                    detail="沙箱内嵌入水印（gds_sbx）")
        else:
            # 前端明确选了「不加防伪水印」，进度条就不能再显示"嵌入水印"，
            # 否则用户会以为系统偷偷加了水印，或者以为跳过了一步操作。
            store.set_task_progress(task_id, stage="STORING_PAYLOAD", percent=35,
                                    detail="按用户选择不加防伪水印，原样入库并登记存证")
        # 长任务必须持续回报进度：OSGB 24 GB / 28,581 文件实测跑 8 小时，
        # 期间 progress_percent 一直停在 35%，前端据此判定"卡死"。
        # 嵌入循环每 200 个文件回调一次，把 35%~85% 之间线性推进。
        _osgb_hook = None
        _set_osgb_hook = None
        if (str(request.data_type or "").upper() in {"OSGB", "MESH"}
                and getattr(request, "watermark_enabled", True)):
            try:
                from .osgb_texture_watermark import set_progress_hook as _set_osgb_hook

                def _osgb_hook(done: int, total: int, _tid: str = task_id, _store=store) -> None:
                    pct = 35 + int(50 * (done / max(total, 1)))
                    try:
                        _store.set_task_progress(_tid, stage="WATERMARK_EMBEDDING", percent=pct,
                                                 detail=f"水印嵌入 {done}/{total} 个文件")
                    except Exception:  # noqa: BLE001 - 进度回报失败不能影响任务
                        pass

                _set_osgb_hook(_osgb_hook)
            except Exception:  # noqa: BLE001
                _set_osgb_hook = None
        _t_process = time.perf_counter()
        artifact = None
        try:
            artifact = _process_task(request, prepared_source, app.state.processing_service, scans if 'scans' in locals() else None)
        finally:
            result_context = f" result={artifact.artifact_id}" if artifact is not None else ""
            print(f"[TIMING] task={task_id[:12]} process {time.perf_counter() - _t_process:.1f}s{result_context}", flush=True)
            if _set_osgb_hook is not None:
                try:
                    _set_osgb_hook(None)
                except Exception:  # noqa: BLE001
                    pass
        if getattr(request, "watermark_enabled", True):
            store.set_task_progress(task_id, stage="VERIFYING", percent=85, detail="校验水印可提取性")
        else:
            store.set_task_progress(task_id, stage="VERIFYING", percent=85, detail="登记成果清单与存证哈希")
        if hasattr(app.state.processing_service, "record_artifact_policy"):
            app.state.processing_service.record_artifact_policy(artifact, watermark_enabled=request.watermark_enabled, anchor_granularity=request.anchor_granularity)
        app.state.task_store.save_artifact(artifact, project_id=request.project_id, user_id=request.user_id)
        app.state.task_store.update_task_status(task_id, "COMPLETED", result_id=artifact.artifact_id)
        store.set_task_progress(task_id, stage="COMPLETED", percent=100, detail="任务完成，等待人工复核")
        audit_record("TASK_COMPLETED", actor=request.user_id, project_id=request.project_id,
                     data_type=request.data_type, task_id=task_id, result_id=artifact.artifact_id)
    except (UnsupportedDataTypeError, KeyError, ValueError) as exc:
        app.state.task_store.update_task_status(task_id, "FAILED", error=_humanize_task_error(exc))
        audit_record("TASK_REJECTED", actor=request.user_id, project_id=request.project_id,
                     data_type=request.data_type, task_id=task_id, reason=str(exc))
    except Exception as exc:
        app.state.task_store.update_task_status(task_id, "FAILED", error=_humanize_task_error(exc))
        audit_record("TASK_FAILED", actor=request.user_id, project_id=request.project_id,
                     data_type=request.data_type, task_id=task_id, reason=str(exc))
    finally:
        if acquired:
            _HEAVY_GATE.release()


def _metrics_from_manifest(manifest: dict) -> dict:
    """把 manifest 里已有的真实指标整理成前端可直接展示的结构。

    只做取值与改名，不计算、不估算：所有数字都能在 manifest.json 里找到原始位置。
    缺失的字段返回 None，由前端决定怎么显示，不用 0 冒充「没测」。
    """
    verification = manifest.get("verification") or {}
    extractor = verification.get("extractor") or {}
    partial = manifest.get("partial_processing") or {}

    carriers_total = partial.get("carriers_total")
    carriers_embedded = partial.get("carriers_embedded")
    if carriers_embedded is None:
        carriers_embedded = extractor.get("declared_carriers")
    coverage = verification.get("carrier_coverage")
    if coverage is None:
        coverage = manifest.get("carrier_coverage")

    per_carrier_ms = extractor.get("per_carrier_ms")
    samples = verification.get("samples") or extractor.get("samples")
    elapsed = manifest.get("elapsed_seconds") or manifest.get("processing_seconds")

    return {
        "result_id": manifest.get("result_id") or manifest.get("artifact_id"),
        "data_type": manifest.get("data_type"),
        "classification": manifest.get("classification"),
        "verified": verification.get("verified") if "verified" in verification else manifest.get("verified"),
        "carriers": {
            "total": carriers_total,
            "embedded": carriers_embedded,
            "checked": extractor.get("carriers_checked"),
            "failed": len(extractor.get("failed_carriers") or []),
            "skipped": partial.get("carriers_skipped"),
        },
        "coverage": coverage,
        "samples": samples,
        "timing": {
            "per_carrier_ms": per_carrier_ms,
            "total_seconds": elapsed,
            "estimated_seconds": (
                round(samples * per_carrier_ms / 1000.0, 1)
                if samples and per_carrier_ms else None
            ),
        },
        "watermark": {
            "expected": manifest.get("expected_watermark") or manifest.get("watermark"),
            "mode": manifest.get("watermark_mode"),
            "enabled": manifest.get("watermark_enabled"),
            "scheme": manifest.get("watermark_scheme") or manifest.get("watermark_method"),
        },
        "integrity": {
            "source_hash": manifest.get("source_hash"),
            "output_hash": manifest.get("output_hash"),
            "trace_record_hash": manifest.get("trace_record_hash"),
            "anchor_granularity": manifest.get("anchor_granularity_effective") or manifest.get("anchor_granularity"),
        },
        "files_processed": manifest.get("files_processed"),
        "source": "manifest.json",
    }


def create_app(
    workspace: Path,
    key: bytes,
    allowed_input_roots: list[Path] | None = None,
    processing_service: ProcessingService | None = None,
    audit_log_dir: Path | str | None = None,
    task_database_path: Path | str | None = None,
    loan_database_path: Path | str | None = None,
    circulation_database_path: Path | str | None = None,
    circulation_export_root: Path | str | None = None,
) -> FastAPI:
    app = FastAPI(title="Geo Data Security Server2")

    @app.exception_handler(RequestValidationError)
    async def _zh_request_validation_handler(request: Request, exc: RequestValidationError):
        # 保持 detail 为数组结构（前端按 detail[].msg 取文案），只把英文说明换成中文。
        return JSONResponse(status_code=422, content={"detail": _zh_validation_errors(exc)})

    pipeline = LocalProcessingPipeline(Path(workspace), key)
    service = processing_service or ProcessingService(Path(workspace), key)
    audit = None
    if audit_log_dir is not None and _HAS_AUDIT:
        audit = audit_log.AuditLog(audit_log_dir)
    app.state.pipeline = pipeline
    app.state.processing_service = service
    app.state.audit = audit
    task_store = TaskStore(task_database_path or (Path(workspace) / "state" / "server2.sqlite3"))
    app.state.task_store = task_store
    loan_store = LoanStore(loan_database_path or (Path(workspace) / "state" / "loans.sqlite3"))
    app.state.loan_store = loan_store
    # 《记录2》大类 1–3 的业务台账：账号/项目/三区资产/申请审批/授权归还/提醒告警/版本。
    circulation_store = CirculationStore(
        circulation_database_path or (Path(workspace) / "state" / "circulation.sqlite3"),
        export_root=circulation_export_root or (Path(workspace) / "exports" / "users"))
    app.state.circulation_store = circulation_store
    admin_auth = AdminAuthStore(task_store.database_path, Path(workspace) / "state" / "superadmin.bootstrap.txt")
    app.state.admin_auth = admin_auth
    sandbox_policy = SandboxPolicyStore(Path(workspace) / "state" / "sandbox_policy.sqlite3", Path(workspace))
    app.state.sandbox_policy = sandbox_policy
    sandbox_pool = SandboxPoolStore(Path(workspace) / "state" / "sandbox_pool.sqlite3", Path(workspace))
    app.state.sandbox_pool = sandbox_pool
    storage_policy = StoragePolicy(Path(workspace))
    app.state.storage_policy = storage_policy

    def _remount_pool_volumes() -> None:
        """启动后把数据卷挂回来（主卷 V: 由 gds_boot.ps1 先挂）。

        没有这一步，机器重启后数据卷全部 OFFLINE，上传静默挤回主卷，
        「加了卷却没变快变大」会表现为一个没人能立刻看懂的怪现象。
        放在后台线程：挂载要等 VeraCrypt，不该阻塞 API 起来。
        """
        try:
            results = sandbox_pool.ensure_all_mounted()
            if results:
                audit_record("SANDBOX_VOLUMES_STARTUP_MOUNT", actor="system", results=results)
        except Exception as exc:  # noqa: BLE001 - 启动挂载失败不能拖垮 API
            try:
                audit_record("SANDBOX_VOLUMES_STARTUP_MOUNT_FAILED", actor="system",
                             error=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass

    if os.environ.get("GDS_SANDBOX_SKIP_STARTUP_MOUNT") != "1":
        threading.Thread(target=_remount_pool_volumes, name="sandbox-volume-remount",
                         daemon=True).start()
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gds-task")
    app.state.task_executor = executor
    roots = [(Path(item).resolve()) for item in (allowed_input_roots or [Path.cwd()])]

    def audit_record(event: str, actor: str | None = None, **details) -> None:
        if audit is not None:
            audit.record(event, actor=actor, **details)

    def check_input_path(source_path: str, expect_dir: bool = False) -> Path:
        candidate = Path(source_path).resolve()
        exists_ok = candidate.is_dir() if expect_dir else candidate.is_file()
        if not exists_ok:
            raise HTTPException(status_code=404, detail="input path not found")
        if not any(candidate == root or root in candidate.parents for root in roots):
            raise HTTPException(status_code=403, detail="input path is outside allowed roots")
        return candidate

    def require_verified_watermark(data: dict) -> None:
        """Refuse to approve a result whose watermark could not be confirmed.

        ``is_partial`` keys on source integrity, so an engine failure (or a product whose embedded
        carriers do not re-extract) used to slide into the manual-review lane and could be approved
        as PARTIALLY_APPROVED - which hands the caller an unwatermarked raster. A failed
        verification is a re-run case, not a review case.
        """
        if data.get("verification_status") == "VERIFICATION_FAILED":
            raise HTTPException(
                status_code=409,
                detail=("result verification failed; the watermark could not be confirmed, so this result "
                        "cannot be approved as either a complete or a partial deliverable - re-run the task"),
            )

    def require_service_header(service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> None:
        try:
            _require_service_token(service_token)
        except HTTPException:
            audit_record("AUTH_DENIED", actor=service_token or "<missing>", action="internal_api")
            raise

    def require_admin_header(admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> None:
        configured = _admin_token()
        identity = admin_auth.authenticate_token(admin_token or "")
        legacy_ok = bool(configured and admin_token and secrets.compare_digest(admin_token, configured))
        if not identity and not legacy_ok:
            audit_record("ADMIN_AUTH_DENIED", actor=admin_token or "<missing>", action="admin_api")
            raise HTTPException(status_code=401, detail="admin authentication required")

    def admin_identity(admin_token: str | None) -> dict:
        identity = admin_auth.authenticate_token(admin_token or "")
        if identity:
            return identity
        configured = _admin_token()
        if configured and admin_token and secrets.compare_digest(admin_token, configured):
            return {"username": "legacy-admin-token", "role": "SUPERADMIN"}
        raise HTTPException(status_code=401, detail="admin authentication required")

    def require_superadmin(admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        identity = admin_identity(admin_token)
        if identity["role"] != "SUPERADMIN":
            raise HTTPException(status_code=403, detail="superadmin role required")
        return identity

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "role": "processing_storage_readonly_output"}

    # Server1-facing upload/task API. The sender never supplies a filesystem path.
    @app.post("/api/v1/ingest/sessions", status_code=201, dependencies=[Depends(require_service_header)])
    def create_upload(request: UploadCreateRequest) -> dict:
        # 顺序很关键：**先选卷，再按那个卷校验配额**。
        # 反过来的话，多卷部署里全局配额会先把请求挡在门外，新加的卷永远用不上。
        normalized = request.data_type.upper()
        aliases = {"DLG": "SHP", "DOM": "GEOTIFF", "MESH": "OSGB", "THEMATIC_VECTOR": "SHP"}
        if normalized in aliases:
            normalized = aliases[normalized]
        try:
            category = parse_classification(request.classification)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if not allowed_for_data_type(category, normalized):
            raise HTTPException(status_code=422, detail=f"classification {category.value} is incompatible with {request.data_type}")
        filename = Path(request.filename).name
        if filename != request.filename:
            raise HTTPException(status_code=422, detail="invalid filename; path components are not allowed")
        filename = _validate_windows_filename(filename)
        upload_id = "upl_" + secrets.token_hex(16)
        # 选卷：按容量+优先级挑一个放得下的卷。全部放不下就 507，并附上每卷实测余量，
        # 让调用方知道"该加卷"还是"该清理"，而不是收到一个笼统的磁盘错误。
        try:
            target_vol = sandbox_pool.select_volume(int(request.size_bytes),
                                                    prefer=getattr(request, "volume_id", None))
        except NoCapacityError as exc:
            audit_record("SANDBOX_NO_CAPACITY", actor=request.user_id, project_id=request.project_id,
                         requested_bytes=request.size_bytes, reason=str(exc))
            raise HTTPException(status_code=507, detail={"message": str(exc), "volumes": exc.detail}) from exc

        # 配额闸门针对**选中的那个卷**，不是全平台总量 —— 否则加多少卷都没用。
        allowed, reason = sandbox_policy.check_quota(int(request.size_bytes or 0),
                                                     volume_root=target_vol["root"])
        if not allowed:
            audit_record("SANDBOX_QUOTA_BLOCKED", actor=request.user_id, project_id=request.project_id,
                         requested_bytes=request.size_bytes, reason=reason,
                         volume_id=target_vol["id"])
            raise HTTPException(status_code=507, detail=reason)
        directory = sandbox_pool.area_path(target_vol["id"], "incoming", request.project_id, upload_id)
        directory.mkdir(parents=True, exist_ok=False)
        # 分片大小按调用方声明值落库：偏移量 = chunk_no * chunk_size，
        # 服务端**不能**擅自改这个数值，否则按 64KB 切的前端会被写成错位的文件，
        # 直到 complete 时 SHA-256 对不上才暴露，一次 1GB 上传整个白做。
        advice = _chunk_advice(int(request.chunk_size), int(request.size_bytes))
        if advice["too_small"]:
            audit_record("UPLOAD_CHUNK_SIZE_SMALL", actor=request.user_id,
                         project_id=request.project_id, upload_id=upload_id,
                         chunk_size=int(request.chunk_size),
                         recommended=int(advice["recommended_chunk_size"]),
                         estimated_mbps=advice["estimated_throughput_mbps"])
        item = {"upload_id": upload_id, "project_id": request.project_id, "user_id": request.user_id,
                "data_type": request.data_type.upper(), "classification": category.value, "filename": filename,
                "expected_bytes": request.size_bytes, "expected_sha256": request.sha256.lower(),
                "chunk_size": int(request.chunk_size), "status": "UPLOADING", "source_path": str(directory / filename)}
        task_store.save_upload(item)
        audit_record("UPLOAD_CREATED", actor=request.user_id, project_id=request.project_id,
                     upload_id=upload_id, data_type=request.data_type,
                     volume_id=target_vol["id"], volume_root=target_vol["root"])
        return {"upload_id": upload_id, "status": "UPLOADING", "chunk_size": int(request.chunk_size),
                "received_bytes": 0, "volume_id": target_vol["id"], "volume_root": target_vol["root"],
                **advice}

    @app.put("/api/v1/ingest/sessions/{upload_id}/chunks/{chunk_no}", dependencies=[Depends(require_service_header)])
    async def upload_chunk(upload_id: str, chunk_no: int, request: Request) -> dict:
        upload = task_store.get_upload(upload_id)
        if not upload or upload["status"] not in {"UPLOADING", "CREATED"}:
            raise HTTPException(status_code=404, detail="upload session not found or closed")
        if chunk_no < 0:
            raise HTTPException(status_code=422, detail="invalid chunk number")
        try:
            body = await request.body()
        except ClientDisconnect as exc:
            raise HTTPException(status_code=499, detail="client disconnected during chunk upload; retry the same chunk") from exc
        if not body or len(body) > upload["chunk_size"]:
            raise HTTPException(status_code=413, detail="invalid chunk size")
        destination = Path(upload["source_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        offset = chunk_no * upload["chunk_size"]
        if offset + len(body) > upload["expected_bytes"]:
            raise HTTPException(status_code=422, detail="chunk exceeds declared file size")

        def _write_and_record() -> int:
            """落盘 + 记账。放到线程池里跑，不能直接在这个协程里做。

            这里是同步阻塞 I/O：Windows 上的文件 seek/write，加上 SQLite 提交。
            以前直接在 `async def` 里调用，等于把整个事件循环按住 —— 前端开 4 个并发
            连接上传，请求却在一个线程里排队，并发形同虚设。
            """
            anchor = str(destination.anchor or destination.parent)
            free_bytes = _free_bytes_cached(anchor)
            if free_bytes is not None and free_bytes < len(body) + 64 * 1024 * 1024:
                raise HTTPException(status_code=507,
                                    detail="insufficient storage on Server2 workspace; clean failed uploads and retry")
            try:
                # 不做逐片 fsync：数据完整性由 complete 阶段的整文件 SHA-256 兜底，
                # 中途断电最多让这次上传 409 重传，不会产生"校验通过但内容错"的假成功。
                with destination.open("r+b" if destination.exists() else "wb") as stream:
                    stream.seek(offset); stream.write(body)
            except OSError as exc:
                if getattr(exc, "errno", None) == 28 or "No space left" in str(exc):
                    raise HTTPException(status_code=507,
                                        detail="insufficient storage on Server2 workspace; clean failed uploads and retry") from exc
                raise
            return task_store.record_upload_chunk(upload_id, chunk_no, len(body),
                                                  expected_bytes=upload["expected_bytes"])

        received = await run_in_threadpool(_write_and_record)
        return {"upload_id": upload_id, "chunk_no": chunk_no, "received": True, "received_bytes": received,
                "next_chunk": received // upload["chunk_size"]}

    @app.get("/api/v1/ingest/sessions/{upload_id}", dependencies=[Depends(require_service_header)])
    def upload_status(upload_id: str) -> dict:
        upload = task_store.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="upload session not found")
        return upload

    @app.post("/api/v1/ingest/sessions/{upload_id}/complete", dependencies=[Depends(require_service_header)])
    def complete_upload(upload_id: str) -> dict:
        upload = task_store.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="upload session not found")
        source = Path(upload["source_path"])
        if not source.is_file() or source.stat().st_size != upload["expected_bytes"]:
            task_store.mark_upload_failed(upload_id)
            raise HTTPException(status_code=409, detail="upload is incomplete")
        # 先算哈希（1GB 实测 3.3 秒，313 MB/s），再做一次性分片校验与状态落库。
        # 哈希读用 4MB 块：1MB 块在 1GB 上要慢约 0.1 秒，纯粹是 syscall 次数。
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
        actual = digest.hexdigest()
        if not secrets.compare_digest(actual, upload["expected_sha256"]):
            task_store.mark_upload_failed(upload_id)
            raise HTTPException(status_code=422, detail="sha256 mismatch")
        # 分片齐全性 + 大小校验 + 落 IMPORTED 合成一个事务（原来要 3 次以上提交）
        problem = task_store.finalize_upload(upload_id, upload["expected_bytes"],
                                             upload["chunk_size"], actual)
        if problem:
            raise HTTPException(status_code=409, detail=problem["error"])
        record = service.trace_chain.append("DATA_IMPORTED", project_id=upload["project_id"], classification=upload["classification"],
            source_label="server1", input_hash=actual, output_hash=actual, sandbox_user="gds_sbx", destination=str(source), upload_id=upload_id)
        audit_record("DATA_IMPORTED", actor=upload["user_id"], project_id=upload["project_id"], upload_id=upload_id, input_hash=actual)
        return {"upload_id": upload_id, "status": "IMPORTED", "source_path": str(source), "size_bytes": source.stat().st_size,
                "sha256": actual, "classification": upload["classification"], "trace_record_hash": record["record_hash"]}

    @app.post("/api/v1/tasks", status_code=202, dependencies=[Depends(require_service_header)])
    def submit_v1_task(request: V1TaskRequest) -> dict:
        upload = task_store.get_upload(request.upload_id)
        if not upload or upload["status"] != "IMPORTED":
            raise HTTPException(status_code=409, detail="upload is not imported")
        if upload["project_id"] != request.project_id or upload["user_id"] != request.user_id:
            raise HTTPException(status_code=403, detail="upload ownership mismatch")
        aliases = {"DLG": "SHP", "DOM": "GEOTIFF", "MESH": "OSGB", "THEMATIC_VECTOR": "SHP"}
        normalized = aliases.get(request.data_type.upper(), request.data_type.upper())
        source = Path(upload["source_path"])
        if not source.is_file() or source.stat().st_size != upload["expected_bytes"]:
            task_store.update_upload(request.upload_id, status="FAILED")
            raise HTTPException(status_code=409, detail="upload source is missing or incomplete; upload a new session")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
        if not secrets.compare_digest(digest.hexdigest(), upload["expected_sha256"]):
            task_store.update_upload(request.upload_id, status="FAILED")
            raise HTTPException(status_code=409, detail="upload source hash no longer matches; upload a new session")
        if source.suffix.lower() == ".zip":
            _validate_archive(source)
        profile = request.processing_profile
        if normalized == "OSGB" and profile == "PERFORMANCE":
            profile = "STRICT_PER_MEMBER"
        _validate_precision_request(request, normalized)
        _submit_precision_bounds(request, normalized, source)
        legacy = TaskRequest(project_id=request.project_id, user_id=request.user_id, data_type=normalized,
            source_path=str(source), watermark_text=request.watermark_text, classification=request.classification,
            precision_decimals=request.precision_decimals, processing_profile=profile,
            precision_mode=request.precision_mode, douglas_peucker_tolerance=request.douglas_peucker_tolerance,
            resample_scale=request.resample_scale, resample_width=request.resample_width,
            resample_height=request.resample_height, resample_method=request.resample_method,
            watermark_mode=request.watermark_mode, visible_watermark_text=request.visible_watermark_text,
            watermark_enabled=request.watermark_enabled, anchor_granularity=request.anchor_granularity)
        task_id = secrets.token_hex(16)
        task = {"task_id": task_id, "status": "QUEUED", "project_id": request.project_id, "user_id": request.user_id,
                "result_id": "", "classification": request.classification, "upload_id": request.upload_id,
                # 不写这两项时 save_task 会按默认值落库，页面就会把「没加水印」的任务显示成已加水印。
                "watermark_enabled": request.watermark_enabled, "anchor_granularity": request.anchor_granularity,
                # 重启中断后可原样重放：data_type 存规范化后的值，resume_payload 存全部入参。
                "data_type": normalized, "resume_payload": _task_resume_payload(request)}
        task_store.save_task(task)
        audit_record("TASK_SUBMITTED", actor=request.user_id, project_id=request.project_id, data_type=request.data_type, task_id=task_id, upload_id=request.upload_id)
        app.state.task_executor.submit(_run_task, app, legacy, source, task_id, audit_record)
        return {"task_id": task_id, "status": "QUEUED", "result_id": "", "upload_id": request.upload_id}

    @app.post("/api/v1/tasks/{task_id}/resume", status_code=202, dependencies=[Depends(require_service_header)])
    def resume_task(task_id: str) -> dict:
        """一键重提：把被服务重启中断（INTERRUPTED）的任务按原参数重放。

        为什么需要它：任务跑在进程内线程池里，没有持久化队列，服务一重启，
        执行线程就没了。以前这些行被直接标成 FAILED，前端只能让用户从头再传一遍。
        现在中断行保留了原始入参（resume_payload），这里原样重放，用户不用重传数据。

        契约要点：
        - 只接受 status=INTERRUPTED 的任务；其他状态返回 409，避免重复执行。
        - 复用同一个 upload_id，所以不会重复占空间、也不会产生第二份源文件。
        - 生成**新的** task_id，原 task_id 保留为历史记录（前端可展示"已重提"）。
        - upload 已失效（被清理/未导入）时返回 409 并说明原因，不静默重试。
        """
        original = task_store.get_task(task_id)
        if not original:
            raise HTTPException(status_code=404, detail="task not found")
        status = (original.get("status") or "").upper()
        if status != "INTERRUPTED":
            raise HTTPException(
                status_code=409,
                detail=f"task is {status}; only INTERRUPTED tasks can be resumed")

        upload_id = (original.get("upload_id") or "").strip()
        raw_payload = (original.get("resume_payload") or "").strip()
        if not upload_id or not raw_payload:
            raise HTTPException(
                status_code=409,
                detail="task has no resume payload (submitted before resume support); resubmit it manually")
        try:
            payload = json.loads(raw_payload)
        except ValueError:
            raise HTTPException(status_code=409, detail="task resume payload is corrupted; resubmit it manually")

        upload = task_store.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=409, detail="the source upload no longer exists; upload the data again")
        if upload["status"] != "IMPORTED":
            raise HTTPException(
                status_code=409,
                detail=f"the source upload is {upload['status']}; upload the data again")
        source = Path(upload["source_path"])
        if not source.is_file():
            raise HTTPException(status_code=409, detail="the source file is missing from the workspace; upload the data again")

        # 用原入参重建请求：字段缺失时按 V1TaskRequest 的默认值补齐。
        resumed = dict(payload)
        resumed["upload_id"] = upload_id
        resumed.setdefault("project_id", original.get("project_id"))
        resumed.setdefault("user_id", original.get("user_id"))
        resumed.setdefault("data_type", original.get("data_type") or upload["data_type"])
        resumed.setdefault("classification", original.get("classification") or upload["classification"])
        try:
            request = V1TaskRequest(**resumed)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"resume payload no longer matches the API contract: {exc}")

        new_task_id = secrets.token_hex(16)
        task_store.save_task({
            "task_id": new_task_id, "status": "QUEUED",
            "project_id": request.project_id, "user_id": request.user_id,
            "result_id": "", "classification": request.classification,
            "upload_id": upload_id, "data_type": original.get("data_type") or upload["data_type"],
            "watermark_enabled": request.watermark_enabled,
            "anchor_granularity": request.anchor_granularity,
            "resume_payload": _task_resume_payload(request),
        })
        audit_record("TASK_RESUMED", actor=request.user_id, project_id=request.project_id,
                     task_id=new_task_id, previous_task_id=task_id, upload_id=upload_id,
                     previous_stage=original.get("stage"), previous_progress=original.get("progress_percent"))

        profile = request.processing_profile
        aliases = {"DLG": "SHP", "DOM": "GEOTIFF", "MESH": "OSGB", "THEMATIC_VECTOR": "SHP"}
        normalized = aliases.get(request.data_type.upper(), request.data_type.upper())
        if normalized == "OSGB" and profile == "PERFORMANCE":
            profile = "STRICT_PER_MEMBER"
        legacy = TaskRequest(project_id=request.project_id, user_id=request.user_id, data_type=normalized,
            source_path=str(source), watermark_text=request.watermark_text, classification=request.classification,
            precision_decimals=request.precision_decimals, processing_profile=profile,
            precision_mode=request.precision_mode, douglas_peucker_tolerance=request.douglas_peucker_tolerance,
            resample_scale=request.resample_scale, resample_width=request.resample_width,
            resample_height=request.resample_height, resample_method=request.resample_method,
            watermark_mode=request.watermark_mode, visible_watermark_text=request.visible_watermark_text,
            watermark_enabled=request.watermark_enabled, anchor_granularity=request.anchor_granularity)
        app.state.task_executor.submit(_run_task, app, legacy, source, new_task_id, audit_record)
        return {"task_id": new_task_id, "status": "QUEUED", "result_id": "",
                "upload_id": upload_id, "resumed_from": task_id}

    @app.get("/api/v1/ingest/sessions/{upload_id}/integrity", dependencies=[Depends(require_service_header)])
    def v1_upload_integrity(upload_id: str) -> dict:
        """Pre-flight source integrity of an imported upload - read-only, never a gate.

        The platform accepts an incomplete source: the readable region is processed and the artifact
        is delivered as a partial result under manual review. This route exists so the caller can
        see that verdict (and the structured warning) right after import and decide whether to
        submit the watermarking task at all, instead of learning about it from the result manifest.
        """
        upload = task_store.get_upload(upload_id)
        if not upload:
            raise HTTPException(status_code=404, detail="upload not found")
        if upload["status"] != "IMPORTED":
            raise HTTPException(status_code=409, detail=f"upload is {upload['status']}; integrity is checked after import")
        source = Path(upload["source_path"])
        if not source.is_file():
            raise HTTPException(status_code=409, detail="uploaded source is missing from the workspace")
        response: dict = {
            "upload_id": upload_id,
            "filename": upload["filename"],
            "size_bytes": source.stat().st_size,
            "data_type": upload["data_type"],
            "partial_mode_enabled": _partial_mode_enabled(),
            "integrity_checked": False,
            "accepted": True,
            "source_integrity": None,
            "processing_plan": None,
            "can_submit_task": True,
            "warning": None,
            "scan": None,
        }
        if normalize_data_type(upload["data_type"]) not in _RASTER_DATA_TYPES:
            response["note"] = "完整性预检针对栅格数据；其他数据类按载荷类型与结构校验。"
            return response
        if os.environ.get("GDS_PREFLIGHT_RASTER_SCAN", "1") != "1" or not sandbox_compute.enabled():
            response["note"] = "预检扫描未启用（GDS_PREFLIGHT_RASTER_SCAN 或沙箱不可用），未做完整性判定。"
            return response
        try:
            detail = _cached_raster_scan(source, int(os.environ.get("GDS_PREFLIGHT_SCAN_TIMEOUT", "3600")))
        except RuntimeError as exc:
            warning = unreadable_source_warning(str(exc))
            response.update({"integrity_checked": True, "accepted": False, "can_submit_task": False,
                             "source_integrity": "UNREADABLE", "processing_plan": "REJECT", "warning": warning})
            audit_record("PAYLOAD_INTEGRITY_CHECK", actor=upload["user_id"], project_id=upload["project_id"],
                         upload_id=upload_id, source_integrity="UNREADABLE", reason=str(exc)[:400])
            return response
        detail = detail if isinstance(detail, dict) else {}
        shortfall = raster_carrier_shortfall(detail, upload["data_type"])
        if shortfall:
            response.update({
                "integrity_checked": True, "accepted": False, "can_submit_task": False,
                "source_integrity": "COMPLETE", "processing_plan": "REJECT",
                "warning": {"code": "SOURCE_BELOW_CARRIER_MINIMUM", "level": "ERROR",
                            "source_integrity": "COMPLETE", "message": shortfall},
            })
            return response
        integrity = _partial_scan_result(detail)
        warning = source_integrity_warning({**detail, **integrity})
        accepted = warning is None or _partial_mode_enabled()
        response.update({
            "integrity_checked": True,
            "accepted": accepted,
            "source_integrity": integrity["source_integrity"],
            "processing_plan": ("FULL" if warning is None else
                                ("PARTIAL_READABLE_REGION" if _partial_mode_enabled() else "REJECT")),
            "can_submit_task": accepted,
            "warning": warning,
            "scan": {key: detail.get(key) for key in
                     ("driver", "width", "height", "count", "dtype", "blocks_total", "blocks_read",
                      "blocks_missing", "failed_blocks") if key in detail},
        })
        if warning is None:
            response["scan"].pop("failed_blocks", None)
        audit_record("PAYLOAD_INTEGRITY_CHECK", actor=upload["user_id"], project_id=upload["project_id"],
                     upload_id=upload_id, source_integrity=integrity["source_integrity"],
                     blocks_total=integrity["blocks_total"], blocks_missing=integrity["blocks_missing"],
                     accepted=accepted)
        return response

    @app.get("/api/v1/tasks/{task_id}", dependencies=[Depends(require_service_header)])
    def v1_task_status(task_id: str) -> dict:
        task = task_store.get_task(task_id)
        if not task: raise HTTPException(status_code=404, detail="task not found")
        return _task_view(task)

    @app.get("/api/v1/tasks", dependencies=[Depends(require_service_header)])
    def v1_task_monitor(project_id: str | None = None, status: str | None = None,
                        limit: int = 50) -> dict:
        """工作台监测面板：按项目/状态列出任务及其阶段进度。

        前端「加水印/降精度操作按钮 + 进度提示 + 监测展示」的数据源：
        轮询本接口即可拿到每个任务的 stage / progress_percent / stage_detail。
        """
        rows = task_store.list_tasks(project_id=project_id, status=status)[: max(1, min(200, limit))]
        views = [_task_view(row) for row in rows]
        summary = {
            "total": len(views),
            "queued": sum(1 for v in views if v.get("status") == "QUEUED"),
            "running": sum(1 for v in views if v.get("status") == "RUNNING"),
            "completed": sum(1 for v in views if v.get("status") == "COMPLETED"),
            "failed": sum(1 for v in views if v.get("status") == "FAILED"),
        }
        return {"project_id": project_id, "count": len(views), "summary": summary, "tasks": views}

    @app.get("/api/v1/workspace/status", dependencies=[Depends(require_service_header)])
    def v1_workspace_status(project_id: str | None = None) -> dict:
        """工作台总览：水印状态 + 区块链流转记录，一次调用给前端可视化用。

        合并三块数据，避免前端拼多个接口：
          * tasks         —— 任务阶段进度（进度提示 / 监测）
          * watermarks    —— 每个成果的水印状态（是否嵌入、验证结果、覆盖率、降精度）
          * blockchain    —— 链完整性 + 需人工复核数 + 最近上链回执
        """
        tasks = [_task_view(row) for row in task_store.list_tasks(project_id=project_id)]
        artifacts = task_store.load_artifacts()
        watermark_rows = []
        for key, row in artifacts.items():
            if project_id and str(row.get("project_id")) != project_id:
                continue
            artifact = row["artifact"]
            manifest = _load_manifest(Path(artifact.output))
            verification = manifest.get("verification") or {}
            watermark_rows.append({
                "result_id": key,
                "project_id": row.get("project_id"),
                "data_type": manifest.get("data_type") or artifact.data_type,
                "approval_status": artifact.status,
                "watermark_enabled": bool(manifest.get("watermark_enabled", True)),
                "watermark_method": manifest.get("watermark_method"),
                "verification_status": manifest.get("verification_status"),
                "verified": verification.get("verified"),
                "carrier_coverage": verification.get("carrier_coverage"),
                "coverage_meets_bar": verification.get("coverage_meets_bar"),
                "precision_reduction": manifest.get("precision_reduction"),
                "source_integrity": (manifest.get("warnings") and "INCOMPLETE") or "COMPLETE",
                "trace_record_hash": manifest.get("trace_record_hash"),
                "anchor": app.state.processing_service.trace_chain.anchor_for(manifest.get("trace_record_hash")),
                "anchor_units": manifest.get("anchor_units", []),
            })
        chain = app.state.processing_service.trace_chain.verify()
        anchored = sum(1 for row in watermark_rows if row.get("anchor"))
        recent = [r for r in app.state.processing_service.trace_chain.search(project_id=project_id)][-10:]
        return {
            "project_id": project_id,
            "tasks": tasks,
            "watermarks": watermark_rows,
            "blockchain": {
                "chain": chain,
                "anchored_results": anchored,
                "pending_review": sum(1 for r in watermark_rows if r.get("approval_status") == "PENDING"),
                "recent_trace_records": recent,
            },
        }

    @app.get("/api/v1/results/{result_id}/verification", dependencies=[Depends(require_service_header)])
    def v1_verification(result_id: str) -> dict:
        record = task_store.get_artifact_record(result_id)
        if not record: raise HTTPException(status_code=404, detail="result not found")
        manifest_path = Path(record["artifact"].output); manifest_path = manifest_path / "manifest.json" if manifest_path.is_dir() else manifest_path.parent / "manifest.json"
        if not manifest_path.is_file(): raise HTTPException(status_code=404, detail="verification not found")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    @app.get("/api/v1/results/{result_id}/metrics", dependencies=[Depends(require_service_header)])
    def v1_metrics(result_id: str) -> dict:
        """成果处理与校验指标。

        Server1 的详情页在 tasks → provenance/query → verification 之后会拉这一条，
        用来展示载体覆盖率、耗时等。此前这条路由不存在，Server1 一直拿到 404
        （历史日志里可见对多个 result_id 反复 404）。
        """
        record = task_store.get_artifact_record(result_id)
        if not record:
            raise HTTPException(status_code=404, detail="result not found")
        output = Path(record["artifact"].output)
        path = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="manifest not found")
        return _metrics_from_manifest(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/v1/results/{result_id}/manifest", dependencies=[Depends(require_service_header)])
    def v1_manifest(result_id: str) -> dict:
        record = task_store.get_artifact_record(result_id)
        if not record: raise HTTPException(status_code=404, detail="result not found")
        output = Path(record["artifact"].output); path = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
        if not path.is_file(): raise HTTPException(status_code=404, detail="manifest not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/v1/results/{result_id}/approve", dependencies=[Depends(require_service_header)])
    def v1_approve(result_id: str, request: ApproveRequest) -> dict:
        record = task_store.get_artifact_record(result_id)
        if not record: raise HTTPException(status_code=404, detail="result not found")
        output = Path(record["artifact"].output); manifest_path = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="manifest not found")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # A recorded rejection is not undone by approving later; the data must be re-processed.
        if record["artifact"].status == APPROVAL_REJECTED:
            raise HTTPException(status_code=409,
                                detail="result was rejected by manual review; delivering it needs a new task")
        require_verified_watermark(data)
        is_partial = data.get("verification_status") == "VERIFIED_PARTIAL" or data.get("source_integrity") == "INCOMPLETE"
        if is_partial and not request.manual_review_confirmed:
            raise HTTPException(
                status_code=409,
                detail="partial result requires manual_review_confirmed=true and review_conclusion",
            )
        if is_partial and not request.review_conclusion:
            raise HTTPException(status_code=422, detail="review_conclusion is required for partial results")
        approval_status = "PARTIALLY_APPROVED" if is_partial else "APPROVED"
        data.update({
            "status": approval_status,
            "approved_by": request.reviewer_id,
        })
        if is_partial:
            data["manual_review"] = {
                "confirmed": True,
                "reviewer_id": request.reviewer_id,
                "conclusion": request.review_conclusion,
            }
            data["data_quality_conclusion"] = request.review_conclusion
            data["approval_policy"] = "MANUAL_REVIEW_COMPLETED_PARTIAL_RESULT"
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        task_store.set_artifact_status(result_id, approval_status)
        audit_record("RESULT_PARTIALLY_APPROVED" if is_partial else "RESULT_APPROVED", actor=request.reviewer_id, result_id=result_id, project_id=record["project_id"], conclusion=request.review_conclusion if is_partial else None)
        return {"result_id": result_id, "approval_status": approval_status, "data_quality_conclusion": request.review_conclusion if is_partial else None}

    @app.post("/api/v1/results/lookup", dependencies=[Depends(require_service_header)])
    def v1_results_lookup(request: ResultLookupRequest) -> dict:
        """Batch reconciliation for a caller whose own database may reference results this node no
        longer holds (workspace purge, storage rotation, a restored backup).

        Every id answers one of three states, which is what a caller needs to converge its records
        without guessing:

        * ``PRESENT`` - the artifact is still here; the node's own approval status is returned, which
          may legitimately differ from what the caller recorded.
        * ``PURGED`` - no row anymore, but the local trace chain still holds a record for this id, so
          the node *did* produce and record it; the record hash stays verifiable. The caller should
          mark its copy expired rather than re-request it.
        * ``NEVER_SEEN`` - no row and no trace record: the id never belonged to this node, so the
          caller's reference is a client-side error rather than a retention issue.

        Read-only; it never changes a status and never deletes anything.
        """
        requested = [str(item) for item in (request.result_ids or []) if str(item).strip()][:200]
        chain = app.state.processing_service.trace_chain
        rows: list[dict] = []
        for result_id in requested:
            record = task_store.get_artifact_record(result_id)
            if record:
                rows.append({
                    "result_id": result_id,
                    "state": "PRESENT",
                    "approval_status": record["artifact"].status,
                    "project_id": record["project_id"],
                    "data_type": record.get("data_type"),
                })
                continue
            # The chain is the durable record: a purged artifact still has its provenance entries.
            entries = chain.search(artifact_id=result_id)
            entry = {
                "result_id": result_id,
                "state": "PURGED" if entries else "NEVER_SEEN",
                "approval_status": None,
                "project_id": (entries[-1].get("project_id") if entries else None),
                "data_type": None,
            }
            if entries:
                last = entries[-1]
                entry.update({
                    "trace_event": last.get("event_type"),
                    "trace_record_hash": last.get("record_hash"),
                    "trace_at": last.get("timestamp"),
                    "trace_records": len(entries),
                })
            rows.append(entry)
        return {
            "count": len(rows),
            "present": sum(1 for row in rows if row["state"] == "PRESENT"),
            "purged": sum(1 for row in rows if row["state"] == "PURGED"),
            "never_seen": sum(1 for row in rows if row["state"] == "NEVER_SEEN"),
            "results": rows,
        }

    @app.post("/api/v1/results/{result_id}/reject", dependencies=[Depends(require_service_header)])
    def v1_reject(result_id: str, request: RejectRequest) -> dict:
        """Reject a result after manual review - the counterpart of approve.

        The decision is written to the manifest and the artifact status, so every delivery route
        refuses the data and the frontend can list it as rejected.

        A *partial* approval is explicitly not a complete acceptance (it carries a data-quality
        caveat and a reviewer conclusion), so it can still be rejected afterwards: the reviewer's
        later judgement supersedes the earlier conditional one, and the superseded decision is kept
        in the manifest. A *complete* approval is terminal instead - flipping it is not a status
        edit but a new processing task, because the delivered artifact was already accepted.
        """
        record = task_store.get_artifact_record(result_id)
        if not record:
            raise HTTPException(status_code=404, detail="result not found")
        current = record["artifact"].status
        if current == APPROVAL_APPROVED:
            raise HTTPException(
                status_code=409,
                detail="result is already APPROVED (complete acceptance); rejecting a complete "
                       "deliverable needs a new task, not a status flip",
            )
        if current == APPROVAL_REJECTED:
            raise HTTPException(status_code=409, detail="result is already rejected")
        output = Path(record["artifact"].output)
        manifest_path = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="manifest not found")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        # A rejection that supersedes a partial approval keeps what was decided earlier, so the
        # sequence stays auditable instead of the earlier conclusion simply disappearing.
        superseded = None
        if current == APPROVAL_PARTIAL:
            superseded = {
                "status": APPROVAL_PARTIAL,
                "approved_by": data.get("approved_by"),
                "approved_at": data.get("approved_at"),
                "data_quality_conclusion": data.get("data_quality_conclusion"),
            }
        data.update({
            "status": APPROVAL_REJECTED,
            "rejected_by": request.reviewer_id,
            "rejection_conclusion": request.review_conclusion,
            "rejected_at": now_local(),
            "approval_policy": "REJECTED_BY_REVIEWER",
        })
        if superseded:
            data["superseded_approval"] = superseded
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        task_store.set_artifact_status(result_id, APPROVAL_REJECTED)
        audit_record("RESULT_REJECTED", actor=request.reviewer_id, result_id=result_id,
                     project_id=record["project_id"], conclusion=request.review_conclusion,
                     superseded_status=current if superseded else None)
        return {"result_id": result_id, "approval_status": APPROVAL_REJECTED,
                "rejected_by": request.reviewer_id, "rejection_conclusion": request.review_conclusion,
                "superseded_approval": superseded}

    @app.get("/api/v1/results/pending-review", dependencies=[Depends(require_service_header)])
    def v1_pending_review(project_id: str | None = None, limit: int = 50) -> dict:
        """Results waiting for a manual review decision, newest first.

        A result is listed while it carries no decision and its manifest asks for a human (partial
        source, partial verdict, or failed verification). Approving or rejecting it takes it out of
        the list, so a reviewer's queue cannot grow stale.
        """
        limit = max(1, min(int(limit), 200))
        rows = sorted(task_store.load_artifacts().values(),
                      key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        results: list[dict] = []
        for row in rows:
            if len(results) >= limit:
                break
            if project_id and row.get("project_id") != project_id:
                continue
            if row["artifact"].status in DECIDED_STATES:
                continue
            output = Path(row["artifact"].output)
            manifest_path = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not needs_manual_review(manifest):
                continue
            verification = manifest.get("verification") or {}
            partial = manifest.get("partial_processing") or {}
            results.append({
                "result_id": row["artifact_id"],
                "project_id": row.get("project_id"),
                "user_id": row.get("user_id"),
                "data_type": row.get("data_type"),
                "status": row["artifact"].status,
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "source_integrity": manifest.get("source_integrity"),
                "verification_status": manifest.get("verification_status"),
                "approval_policy": manifest.get("approval_policy"),
                # the approve call requires manual_review_confirmed + review_conclusion exactly when
                # the result is partial, so the frontend can render the right form up front
                "requires_manual_review": bool(
                    manifest.get("source_integrity") == "INCOMPLETE"
                    or manifest.get("verification_status") == "VERIFIED_PARTIAL"
                ),
                "verification": {
                    "verified": verification.get("verified"),
                    "method": verification.get("method"),
                    "samples": verification.get("samples"),
                    "matched": verification.get("matched"),
                    "carrier_coverage": verification.get("carrier_coverage"),
                    "coverage_bar": verification.get("coverage_bar"),
                    "coverage_meets_bar": verification.get("coverage_meets_bar"),
                },
                "partial_processing": {
                    key: partial.get(key) for key in
                    ("carriers_total", "carriers_embedded", "carriers_skipped", "valid_h",
                     "blocks_total", "blocks_read", "blocks_missing", "psnr_db")
                    if key in partial
                },
                "data_quality_conclusion": manifest.get("data_quality_conclusion"),
                "expected_watermark": manifest.get("expected_watermark"),
                # the same warning record the task and the ingest check carry, so a reviewer sees
                # the reason in one place
                "warnings": manifest.get("warnings") or [],
            })
        return {"count": len(results), "results": results}

    @app.get("/api/v1/results/{result_id}/files/{filename}", dependencies=[Depends(require_service_header)])
    def v1_file(result_id: str, filename: str) -> FileResponse:
        record = task_store.get_artifact_record(result_id)
        if not record:
            raise HTTPException(status_code=404, detail="approved result not found")
        rejected_result_guard(record["artifact"].status)
        if not _deliverable(record):
            raise HTTPException(status_code=404, detail="approved result not found")
        output = Path(record["artifact"].output); root = output if output.is_dir() else output.parent
        candidate = (root / filename).resolve()
        if candidate.parent != root.resolve() or not candidate.is_file(): raise HTTPException(status_code=404, detail="file not found")
        audit_record("RESULT_FILE_DOWNLOADED", actor=record["user_id"], result_id=result_id, project_id=record["project_id"], filename=filename)
        return FileResponse(candidate, filename=candidate.name)

    @app.post("/api/v1/provenance/query", dependencies=[Depends(require_service_header)])
    def provenance_query(result_id: str, expected_watermark: str | None = None) -> dict:
        record = task_store.get_artifact_record(result_id)
        if not record: raise HTTPException(status_code=404, detail="result not found")
        output = Path(record["artifact"].output); manifest = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
        if not manifest.is_file(): raise HTTPException(status_code=404, detail="manifest not found")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if expected_watermark and str(data.get("expected_watermark")) != expected_watermark: raise HTTPException(status_code=404, detail="watermark not found")
        data["artifact_id"] = result_id; data["chain"] = service.trace_chain.verify()
        return data

    def _deliverable(record) -> bool:
        """该成果是否可交付（下载/导出）。

        三种放行口径，全部基于已有证据、不改接口契约：
        1. 已审批（APPROVED / PARTIALLY_APPROVED）—— 原有口径；
        2. 部分处理但水印达标（覆盖率 ≥95% 且水印可再抽取）—— 缺块本是空白像素，
           不影响溯源与交付；
        3. 其余仍按原样拒绝。
        """
        status = record["artifact"].status
        if status in APPROVED_STATES:
            return True
        if status in DECIDED_STATES:      # REJECTED：人工已否决，不再翻转
            return False
        manifest = _load_manifest(Path(getattr(record["artifact"], "output", "") or ""))
        return bool(manifest) and partial_watermark_acceptable(manifest)

    @app.get("/api/v1/results/{result_id}/export", dependencies=[Depends(require_service_header)])
    def export_result(result_id: str) -> FileResponse:
        """Export any approved result (single file, SHP set, GeoTIFF, GeoJSON, DEM, or Mesh tree) as one ZIP."""
        record = task_store.get_artifact_record(result_id)
        if not record or not _deliverable(record):
            if record:
                rejected_result_guard(record["artifact"].status)
            raise HTTPException(status_code=404, detail="approved result not found")
        output = Path(record["artifact"].output)
        root = output if output.is_dir() else output.parent
        export_dir = Path(workspace) / "exports" / result_id
        export_dir.mkdir(parents=True, exist_ok=True)
        archive_base = export_dir / f"{result_id}"
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=root))
        # 流式计算摘要：早前用 archive_path.read_bytes()，会把整个 ZIP 一次性读进内存
        # ——真实 DOM 成果的 ZIP 有 1.2 GB，直接吃满内存。改成分块读。
        digest = hashlib.sha256()
        with archive_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        audit_record("RESULT_EXPORTED", actor=record["user_id"], result_id=result_id,
                     project_id=record["project_id"], archive=str(archive_path),
                     sha256=digest.hexdigest())
        return FileResponse(archive_path, filename=archive_path.name, media_type="application/zip")

    def _credential_value(manifest: dict, export_id: str) -> str:
        watermark = str(manifest.get("expected_watermark", ""))
        source_hash = str(manifest.get("source_hash", ""))
        output_hash = str(manifest.get("output_hash", ""))
        return hashlib.sha256(f"{watermark}|{source_hash}|{output_hash}|{export_id}".encode("utf-8")).hexdigest()

    def _export_units(root: Path, manifest: dict, granularity: str) -> list[dict]:
        if granularity == "UPLOAD":
            return [{"unit": "UPLOAD", "sha256": manifest.get("output_hash", "")}]
        ignored = {"manifest.json", ".gds_extract_complete", "payload_inventory.json"}
        files = sorted(item for item in root.rglob("*") if item.is_file() and item.name not in ignored)
        units: list[dict] = []
        for item in files:
            if granularity == "TILE" and item.suffix.lower() in {".tif", ".tiff", ".img", ".vrt"}:
                units.extend(app_tile_units(item, root))
            else:
                units.append({"unit": str(item.relative_to(root)).replace("\\", "/"),
                              "sha256": hashlib.sha256(item.read_bytes()).hexdigest()})
        return units

    def _build_export(app, export_id: str, result_id: str, root: Path, archive_path: Path, record: dict, granularity: str, manifest: dict) -> None:
        app.state.task_store.update_export_job(export_id, status="RUNNING")
        try:
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.make_archive(str(archive_path.with_suffix("")), "zip", root_dir=root)
            archive_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            credential = _credential_value(manifest, export_id)
            units = _export_units(root, manifest, granularity)
            events = []
            for unit in units:
                events.append(app.state.processing_service.trace_chain.append("RESULT_EXPORTED", export_id=export_id, artifact_id=result_id, project_id=record["project_id"], credential_value=credential, anchor_granularity=granularity, unit=unit["unit"], unit_hash=unit["sha256"], output_hash=archive_hash, sandbox_user="gds_sbx" if sandbox_compute.enabled() else "direct"))
            trace_hash = events[-1]["record_hash"] if events else ""
            anchor = app.state.processing_service.trace_chain.anchor_for(trace_hash)
            app.state.task_store.update_export_job(export_id, status="COMPLETED", archive_path=str(archive_path), credential_value=credential, trace_record_hash=trace_hash, anchor_json=json.dumps({"local_trace": True, "onchain": bool(anchor and anchor.get("onchain", anchor.get("status") == 1)), "receipt": anchor}, ensure_ascii=False))
        except Exception as exc:
            app.state.task_store.update_export_job(export_id, status="FAILED", error=f"{type(exc).__name__}: {exc}")

    @app.post("/api/v1/results/{result_id}/export", status_code=202, dependencies=[Depends(require_service_header)])
    def create_export(result_id: str, request: ExportCreateRequest | None = None) -> dict:
        record = task_store.get_artifact_record(result_id)
        if not record or not _deliverable(record):
            if record:
                rejected_result_guard(record["artifact"].status)
            raise HTTPException(status_code=404, detail="approved result not found")
        output = Path(record["artifact"].output); root = output if output.is_dir() else output.parent
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        granularity = (request.anchor_granularity if request else None) or manifest.get("anchor_granularity", "UPLOAD")
        export_id = "exp_" + secrets.token_hex(16)
        archive_path = Path(workspace) / "exports" / result_id / f"{export_id}.zip"
        task_store.save_export_job({"export_id": export_id, "result_id": result_id,
                                    "project_id": record["project_id"], "user_id": record["user_id"],
                                    "status": "QUEUED", "archive_path": str(archive_path)})
        app.state.task_executor.submit(_build_export, app, export_id, result_id, root, archive_path, record, granularity, manifest)
        return {"export_id": export_id, "result_id": result_id, "status": "QUEUED"}

    @app.get("/api/v1/exports/{export_id}", dependencies=[Depends(require_service_header)])
    def export_status(export_id: str) -> dict:
        job = task_store.get_export_job(export_id)
        if not job:
            raise HTTPException(status_code=404, detail="export job not found")
        return job

    @app.get("/api/v1/exports/{export_id}/download", dependencies=[Depends(require_service_header)])
    def export_download(export_id: str) -> FileResponse:
        job = task_store.get_export_job(export_id)
        if not job or job["status"] != "COMPLETED":
            raise HTTPException(status_code=409, detail="export is not completed")
        path = Path(job["archive_path"])
        if not path.is_file():
            raise HTTPException(status_code=404, detail="export archive not found")
        return FileResponse(path, filename=path.name, media_type="application/zip")

    def _query_source(request: WatermarkQueryRequest) -> tuple[Path, str]:
        normalized = normalize_data_type(request.data_type)
        # Uploads arrive as a ZIP for every data class, while the verifiers operate on
        # the controlled extracted payload directory. Resolve that directory before the
        # type checks; passing the ZIP to a raster/tree verifier produced opaque errors.
        candidate = Path(request.source_path).resolve()
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            _validate_archive(candidate)
            payload = zip_payload_root(candidate)
            if not payload.is_dir() or not any(payload.iterdir()):
                raise HTTPException(status_code=409,
                                    detail="archive has not been processed yet; submit a task first so the payload is extracted")
            source = check_input_path(str(payload), expect_dir=True)
            # 矢量借出单允许格式转换：目录里可能是 GeoJSON / Shapefile，而不是原容器。
            if normalized in _VECTOR_DECLARED:
                return resolve_declared_carrier(normalized, source)
            validate_payload_kind(normalized, source, f"extracted payload of {candidate.name}")
            if normalized in _FILE_BASED_TYPES:
                return _payload_data_file(normalized, source), normalized
            return source, normalized
        # 直接交来的矢量文件按自身后缀选提取器。SHP 声明不能把 .json 挡成“必须是目录”。
        if candidate.is_file() and _vector_type_of_file(candidate):
            source = check_input_path(request.source_path, expect_dir=False)
            return source, carrier_type_for_file(normalized, source)
        if normalized == "OSGB":
            tree = check_input_path(request.source_path, expect_dir=True)
            if not any(tree.rglob("*.osgb")):
                raise HTTPException(status_code=422, detail="no .osgb files found in extracted payload")
            return tree, normalized
        # 调用方也可能直接给解包后的目录（例如 S1 传的是 payload 目录而不是 ZIP）
        if candidate.is_dir() and normalized in _VECTOR_DECLARED:
            directory = check_input_path(request.source_path, expect_dir=True)
            return resolve_declared_carrier(normalized, directory)
        if normalized in _FILE_BASED_TYPES and candidate.is_dir():
            directory = check_input_path(request.source_path, expect_dir=True)
            return _payload_data_file(normalized, directory), normalized
        source = check_input_path(request.source_path,
                                  expect_dir=normalized in {"SHP", "OSGB", "OSGB_TEXTURES", "TEXTURES", "DEM"})
        if source.is_file():
            return source, carrier_type_for_file(normalized, source)
        return source, normalized

    def _verify_uploaded_watermark(request: WatermarkQueryRequest,
                                   reference: Path | None = None) -> dict:
        source, normalized = _query_source(request)
        raw_expected = (request.watermark_text or "").strip()
        # "0" 和空串是调用方"没有登记水印号"时的占位值。拿它去逐条比对永远不匹配，
        # 会把明明带我们水印的成果判成"不是我们的"。文本类改为只检出、回报实际值；
        # 栅格与模型类必须给出具体号码，否则明确拒绝而不是给出错误结论。
        raster_like = {"GEOTIFF", "GTIFF", "DEM", "OSGB", "OSGB_TEXTURES", "TEXTURES"}
        if raw_expected in {"", "0"}:
            if normalized in raster_like:
                raise HTTPException(
                    status_code=422,
                    detail="该成果未登记水印号，无法核验栅格或模型数据；请提供 watermark_text 后再核验")
            expected: object = ""
        else:
            expected = _parse_watermark_int(raw_expected) if normalized in raster_like else raw_expected
        if normalized == "OSGB":
            if not sandbox_compute.enabled():
                raise HTTPException(status_code=503, detail="OSGB verification requires configured sandbox")
            result = sandbox_compute.run("osgb_dct_verify_tree", str(source), f"{expected:08X}", timeout_s=3600)
            result.update({"data_type": normalized, "classification": request.classification,
                           "source_path": str(source), "query_only": True})
            return result
        # 栅格类必须走有界判定：verify_artifact 会逐载体全量解码，遇到裁剪/旋转过的探针
        # 每个载体都要跑满 12 次旋转重试（实测 702 ms/载体），2 GB 正射图直接超时。
        # check_any 用的是抽样 + 几何还原，带参考图时能认裁剪/缩放/旋转/转 PNG。
        if normalized in {"GEOTIFF", "GTIFF", "DEM"}:
            # 有参考图时走 TrustMark 有界还原：裁剪/缩放/旋转/转 PNG 都能认出。
            # 不用 check_any 里的 geo-restore —— 它把探针整幅读进内存做 SIFT，
            # 实测 1014 Mpx 成果上超过 34,800 CPU 秒不返回，接口必然超时。
            if reference is not None:
                try:
                    from .trustmark_engine import (TrustMarkGeoTIFFEngine,
                                                   TrustMarkEngineUnavailableError)
                    try:
                        _eng = TrustMarkGeoTIFFEngine()
                    except TrustMarkEngineUnavailableError:
                        _eng = None
                    if _eng is not None:
                        _rr = _eng._call(["geo-restore-bounded", str(reference), str(source),
                                          f"{int(expected):08X}"], timeout=300)
                        if _rr.get("found"):
                            _hits = _rr.get("hits") or 0
                            _total = _rr.get("total") or 0
                            return {
                                "verified": True, "samples": _total, "matched": _hits,
                                "method": "bounded_trustmark_restore",
                                "channel": "RASTER_TRUSTMARK", "modified": True,
                                "conclusion": "仍是本系统数据，但已被修改（裁剪/缩放/旋转/格式转换）",
                                "detail": {"hits": _hits, "total": _total,
                                           "rate": _rr.get("rate"), "tile": _rr.get("tile"),
                                           "carriers": _total, "matched_carriers": _hits},
                                "data_type": normalized, "classification": request.classification,
                                "source_path": str(source), "query_only": True,
                            }
                except Exception:  # noqa: BLE001 - 有界通道失败仍要退回下面的判定
                    pass
            try:
                from . import provenance_any as pa
                root = Path(loan_database_path).parent.parent if loan_database_path else Path(workspace)
                pa._prune_scratch_dirs(root / "provenance_any")
                work = root / "provenance_any" / ("ret_" + __import__("uuid").uuid4().hex)
                work.mkdir(parents=True, exist_ok=True)
                chk = pa.check_any(source, int(expected), reference, work)
                det = chk.get("detail")
                det = det if isinstance(det, dict) else {}
                found = bool(chk.get("found"))
                return {
                    "verified": found, "samples": det.get("carriers"),
                    "matched": det.get("matched_carriers"),
                    "method": "bounded_" + str(chk.get("via") or "any"),
                    "channel": chk.get("channel"),
                    "modified": found,
                    "conclusion": chk.get("conclusion"),
                    "detail": chk.get("detail"),
                    "data_type": normalized, "classification": request.classification,
                    "source_path": str(source), "query_only": True,
                }
            except Exception:  # noqa: BLE001 - 有界通道异常时退回原通道，不让核验直接 500
                pass
        try:
            result = verify_artifact(normalized, source, expected, service.key, service.trustmark_engine)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            # "没有水印字段"是确定结论，不是核验失败：这份数据确实没加过水印。
            # 报成无法核验会让归还方以为要修数据，实际结论就是"不是我们的"。
            if "watermark field not found" in message or "no _wm" in message.lower():
                return {"verified": False, "samples": 0, "matched": 0,
                        "method": f"{normalized.lower()}_attribute_extract",
                        "reason": "载体内没有水印字段，说明未加过水印",
                        "data_type": normalized, "classification": request.classification,
                        "source_path": str(source), "query_only": True}
            raise HTTPException(status_code=422, detail=f"watermark verification failed: {message}") from exc
        result.update({"data_type": normalized, "classification": request.classification,
                       "source_path": str(source), "query_only": True})
        return result

    @app.post("/api/v1/watermarks/detect", dependencies=[Depends(require_service_header)])
    def detect_watermark(request: WatermarkQueryRequest) -> dict:
        result = _verify_uploaded_watermark(request)
        result["detected"] = bool(result.get("verified") or result.get("pixel_matched", 0) > 0 or result.get("matched", 0) > 0)
        audit_record("WATERMARK_DETECTED", actor=request.user_id, project_id=request.project_id,
                     data_type=request.data_type, source_path=request.source_path,
                     detected=result["detected"])
        return result

    @app.post("/api/v1/watermarks/verify", dependencies=[Depends(require_service_header)])
    def verify_watermark(request: WatermarkVerifyRequest) -> dict:
        result = _verify_uploaded_watermark(request)
        if request.expected_watermark and str(request.expected_watermark) != str(request.watermark_text):
            result["verified"] = False
            result["reason"] = "expected_watermark does not match query watermark"
        result["provenance"] = service.trace_chain.search(project_id=request.project_id,
                                                           classification=request.classification)
        audit_record("WATERMARK_VERIFIED_QUERY", actor=request.user_id, project_id=request.project_id,
                     data_type=request.data_type, verified=bool(result.get("verified")))
        return result

    @app.get("/api/v1/provenance/search", dependencies=[Depends(require_service_header)])
    def provenance_search(project_id: str | None = None, artifact_id: str | None = None,
                          classification: str | None = None, input_hash: str | None = None) -> dict:
        filters = {"project_id": project_id, "artifact_id": artifact_id,
                   "classification": classification, "input_hash": input_hash}
        rows = service.trace_chain.search(**filters)
        return {"count": len(rows), "records": rows, "chain": service.trace_chain.verify()}

    @app.get("/api/v1/provenance/flow", dependencies=[Depends(require_service_header)])
    def provenance_flow(project_id: str | None = None, task_id: str | None = None, artifact_id: str | None = None) -> dict:
        tasks = task_store.load_tasks()
        if project_id:
            tasks = {key: row for key, row in tasks.items() if str(row.get("project_id")) == project_id}
        if task_id:
            tasks = {key: row for key, row in tasks.items() if key == task_id}
        artifacts = task_store.load_artifacts()
        if artifact_id:
            artifacts = {key: row for key, row in artifacts.items() if key == artifact_id}
        nodes = [{"id": "server1", "type": "server", "label": "Server1"},
                 {"id": "server2", "type": "server", "label": "Server2"},
                 {"id": "sandbox", "type": "sandbox", "label": "gds_sbx"},
                 {"id": "trace_chain", "type": "ledger", "label": "Trace Chain"},
                 {"id": "besu", "type": "blockchain", "label": "Besu"}]
        edges = [{"from": "server1", "to": "server2", "event": "DATA_TRANSFER"},
                 {"from": "server2", "to": "sandbox", "event": "PROCESSING"},
                 {"from": "sandbox", "to": "trace_chain", "event": "TRACE"},
                 {"from": "trace_chain", "to": "besu", "event": "ANCHOR"}]
        records = service.trace_chain.search(project_id=project_id)
        for task in tasks.values():
            nodes.append({"id": f"task:{task['task_id']}", "type": "task", "status": task["status"], "project_id": task["project_id"], "watermark_enabled": bool(task.get("watermark_enabled", 1)), "anchor_granularity": task.get("anchor_granularity", "UPLOAD")})
            if task.get("result_id"):
                edges.append({"from": f"task:{task['task_id']}", "to": f"artifact:{task['result_id']}", "event": "RESULT"})
        for artifact in artifacts.values():
            aid = artifact["artifact_id"]
            manifest_path = Path(artifact["artifact"].output) / "manifest.json" if Path(artifact["artifact"].output).is_dir() else Path(artifact["artifact"].output).parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            nodes.append({"id": f"artifact:{aid}", "type": "artifact", "status": artifact["artifact"].status, "project_id": artifact["project_id"], "watermark_enabled": manifest.get("watermark_enabled", True), "anchor_granularity": manifest.get("anchor_granularity", "UPLOAD"), "trace_record_hash": manifest.get("trace_record_hash"), "anchor": service.trace_chain.anchor_for(manifest.get("trace_record_hash")), "anchor_units": manifest.get("anchor_units", [])})
        return {"project_id": project_id, "nodes": nodes, "edges": edges, "records": records, "chain": service.trace_chain.verify()}

    # ══════════════════════════════════════════════════════════════════════
    # 数据外借 / 归还 / 期限 与归还核验
    # ══════════════════════════════════════════════════════════════════════
    def _loan_chain():
        return app.state.processing_service.trace_chain

    def _missing_result_http(result_id: str) -> HTTPException:
        """Say WHY a result is unavailable instead of a bare 404.

        The trace chain keeps a record for every result this node once produced, so the two cases
        are distinguishable: a purged result is a retention fact the caller can act on (410), while
        an id with no trace record never belonged here (404). Same split as
        /api/v1/results/lookup, kept in one helper so both endpoints tell the same story.
        """
        try:
            entries = app.state.processing_service.trace_chain.search(artifact_id=result_id)
        except Exception:
            entries = []
        if not entries:
            return HTTPException(
                status_code=404,
                detail=(f"本节点没有成果编号 {result_id} 的任何记录。请确认编号是否正确，"
                        f"或改选状态为已审批的成果。"),
            )
        last = entries[-1]
        when = str(last.get("timestamp") or "")[:19].replace("T", " ")
        event = last.get("event_type") or "产出"
        return HTTPException(
            status_code=410,
            detail=(f"成果 {result_id} 已被清理：本节点曾于 {when} 产出并记录该成果（事件 {event}），"
                    f"但成果记录与原始文件已不存在，无法借出。请重新加工生成新成果后再登记借出。"),
        )

    def _artifact_watermark_text(result_id: str | None) -> str:
        """成果清单里登记的水印号。借出单没登记时用它兜底。

        归还核验要拿"这份数据应该带什么水印"去比对。前端建借出单时不一定会填，
        结果核验端只能拿空串或 "0" 去比，全部要素都不匹配，最后误判成非本系统成果。
        """
        if not result_id:
            return ""
        try:
            record = task_store.get_artifact_record(result_id)
        except Exception:
            return ""
        artifact = record.get("artifact") if record else None
        if artifact is None:
            return ""
        manifest = Path(str(getattr(artifact, "output", ""))).parent / "manifest.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return str(data.get("expected_watermark") or "")

    @app.post("/api/v1/loans", status_code=201, dependencies=[Depends(require_service_header)])
    def create_loan(request: LoanCreateRequest) -> dict:
        """登记一次数据外借：绑定结果 + 借用人 + 归还期限。"""
        record = task_store.get_artifact_record(request.result_id)
        if record is None:
            raise _missing_result_http(request.result_id)
        try:
            due_at = default_due(request.due_days) if request.due_days else request.due_at
            loan = loan_store.create(
                result_id=request.result_id, project_id=request.project_id,
                borrower=request.borrower, due_at=due_at,
                classification=request.classification or record.get("classification", "") or "",
                data_type=request.data_type, purpose=request.purpose,
                operator=request.operator,
                expected_watermark=request.expected_watermark or _artifact_watermark_text(request.result_id),
                carrier_hash=loan_store.hash_carrier(request.carrier_path) if request.carrier_path else "",
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _loan_chain().append("DATA_LENT", loan_id=loan["loan_id"], artifact_id=loan["result_id"],
                             project_id=loan["project_id"], borrower=loan["borrower"],
                             due_at=loan["due_at"], purpose=loan["purpose"],
                             operator=loan["operator"], carrier_hash=loan["carrier_hash"])
        audit_record("DATA_LENT", actor=loan["operator"] or loan["borrower"],
                     project_id=loan["project_id"], artifact_id=loan["result_id"],
                     borrower=loan["borrower"], due_at=loan["due_at"])
        return loan

    @app.get("/api/v1/loans", dependencies=[Depends(require_service_header)])
    def list_loans(status: str | None = None, result_id: str | None = None,
                   project_id: str | None = None, borrower: str | None = None,
                   include_closed: bool = True) -> dict:
        """借出单列表 + 汇总（工作台用）。顺带把过期的刷成 OVERDUE。"""
        newly_overdue = loan_store.refresh_overdue()
        for item in newly_overdue:
            _loan_chain().append("LOAN_OVERDUE", loan_id=item["loan_id"],
                                 artifact_id=item["result_id"], project_id=item["project_id"],
                                 borrower=item["borrower"], due_at=item["due_at"],
                                 overdue_days=item["overdue_days"])
            audit_record("LOAN_OVERDUE", actor=item["borrower"], project_id=item["project_id"],
                         loan_id=item["loan_id"], overdue_days=item["overdue_days"])
        rows = loan_store.list(status=status, result_id=result_id, project_id=project_id,
                               borrower=borrower, include_closed=include_closed)
        fresh = {r["loan_id"] for r in newly_overdue}
        rows = [loan_store.get(r["loan_id"]) if r["loan_id"] in fresh else r for r in rows]
        return {"count": len(rows), "loans": rows, "summary": summarize_loans(rows),
                "newly_overdue": newly_overdue}

    @app.get("/api/v1/loans/summary", dependencies=[Depends(require_service_header)])
    def loans_summary(project_id: str | None = None) -> dict:
        """工作台总览：在借/超期/已归还计数 + 最近到期 + 超期明细。"""
        loan_store.refresh_overdue()
        rows = loan_store.list(project_id=project_id)
        return {"project_id": project_id, "summary": summarize_loans(rows),
                "chain": _loan_chain().verify()}

    @app.get("/api/v1/loans/export-candidates", dependencies=[Depends(require_service_header)])
    def loan_export_candidates(project_id: str | None = None,
                               include_lent: bool = False) -> dict:
        """借出/导出候选：列出「已审批且文件确实存在」的成果，供前端下拉选择。

        为什么需要这个接口：借出登记要求载体路径，但用户不该知道也不该手填
        V:\\ 之类的服务器绝对路径。这里由后端把候选连路径一起给出，前端只负责
        显示文件名，用户选中后路径自动带出。

        三条过滤规则，缺一不可：
          1. artifacts.status == APPROVED（未审批的不能外借）
          2. 磁盘上 output 真实存在（否则借出去也核验不了）
          3. 默认排除已有未归还借出单的成果（include_lent=true 时才列出）
        """
        records = list(task_store.load_artifacts().values())
        if project_id:
            records = [r for r in records if r.get("project_id") == project_id]

        open_loans = {loan["result_id"]: loan for loan in
                      loan_store.list(include_closed=False) if loan.get("status") != STATUS_RETURNED}

        items = []
        skipped = {"not_approved": 0, "missing_on_disk": 0, "already_lent": 0}
        for record in records:
            artifact = record.get("artifact") if isinstance(record, dict) else None
            if artifact is None:
                continue
            artifact_id = getattr(artifact, "artifact_id", None) or record.get("artifact_id")
            status = getattr(artifact, "status", None) or record.get("status")
            if status != "APPROVED":
                skipped["not_approved"] += 1
                continue
            output = Path(getattr(artifact, "output", "") or "")
            if not output.exists():
                skipped["missing_on_disk"] += 1
                continue
            loan = open_loans.get(artifact_id)
            if loan and not include_lent:
                skipped["already_lent"] += 1
                continue
            carrier = output
            if output.is_dir():
                inner = sorted(p for p in output.rglob("*") if p.is_file()
                               and p.suffix.lower() in {".gpkg", ".shp", ".tif", ".tiff", ".geojson"})
                if inner:
                    carrier = inner[0]
            items.append({
                "result_id": artifact_id,
                "project_id": getattr(artifact, "project_id", None) or record.get("project_id"),
                "data_type": getattr(artifact, "data_type", None) or record.get("data_type"),
                "classification": (loan.get("classification") if loan else "") or "",
                "filename": carrier.name,
                "carrier_path": str(carrier),
                "size_bytes": carrier.stat().st_size if carrier.is_file() else None,
                "output_is_dir": output.is_dir(),
                "open_loan_id": loan["loan_id"] if loan else None,
                "open_loan_status": loan["status"] if loan else None,
            })
        items.sort(key=lambda x: (x["project_id"] or "", x["result_id"]))
        return {"count": len(items), "items": items, "skipped": skipped,
                "note": "carrier_path 由后端给出，前端只需展示 filename，不要让用户手填服务器路径"}

    @app.get("/api/v1/loans/{loan_id}/export-preflight", dependencies=[Depends(require_service_header)])
    def loan_export_preflight(loan_id: str) -> dict:
        """借出导出预检：说明这张借出单能不能导出、导什么、有没有风险。

        只做判定不做导出，让前端在点「导出」前就能给出明确提示。
        """
        try:
            loan = loan_store.get(loan_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown loan_id: {loan_id}")
        record = task_store.get_artifact_record(loan["result_id"])
        artifact = record.get("artifact") if record else None
        output = Path(getattr(artifact, "output", "") or "") if artifact else None

        blockers: list[str] = []
        warnings: list[str] = []
        if artifact is None:
            blockers.append("借出单关联的成果已不存在，无法导出")
        elif getattr(artifact, "status", None) != "APPROVED":
            blockers.append(f"成果当前状态为 {getattr(artifact, 'status', 'UNKNOWN')}，未通过审批不可导出")
        if output is not None and not output.exists():
            blockers.append("成果文件在磁盘上已缺失，无法导出")
        if loan["status"] == STATUS_RETURNED:
            warnings.append("该借出单已归还，导出的是归档副本")
        overdue, days = loan_store._due_passed(loan["due_at"]) if loan["status"] != STATUS_RETURNED else (False, 0)
        if overdue:
            warnings.append(f"该借出单已超期 {days} 天，导出行为会被记入审计")

        carrier = None
        if output is not None and output.exists():
            carrier = output
            if output.is_dir():
                inner = sorted(p for p in output.rglob("*") if p.is_file()
                               and p.suffix.lower() in {".gpkg", ".shp", ".tif", ".tiff", ".geojson"})
                carrier = inner[0] if inner else output

        return {
            "loan_id": loan_id,
            "result_id": loan["result_id"],
            "project_id": loan["project_id"],
            "data_type": loan.get("data_type"),
            "filename": carrier.name if carrier else None,
            "carrier_path": str(carrier) if carrier else None,
            "size_bytes": carrier.stat().st_size if carrier and carrier.is_file() else None,
            "exportable": not blockers,
            "blockers": blockers,
            "warnings": warnings,
            "overdue": overdue,
            "overdue_days": days if overdue else 0,
        }

    @app.get("/api/v1/loans/{loan_id}/exports", dependencies=[Depends(require_service_header)])
    def loan_exports(loan_id: str) -> dict:
        """该借出单的导出记录：从审计与链上事件里还原导出历史。

        本节点不单独维护导出表，导出动作会写审计（LOAN_EXPORTED）与链上事件，
        这里把它们聚合成前端可直接展示的列表，避免前端自己拼两个数据源。
        """
        try:
            loan = loan_store.get(loan_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown loan_id: {loan_id}")
        events = [e for e in _loan_chain().search(loan_id=loan_id)
                  if e.get("event_type") in {"LOAN_EXPORTED", "LOAN_EXPORT_REQUESTED"}]
        items = [{
            "event": e.get("event_type"),
            "at": e.get("timestamp"),
            "record_hash": e.get("record_hash"),
            "operator": e.get("operator") or e.get("actor") or e.get("borrower") or "",
            "filename": e.get("filename") or "",
        } for e in events]
        return {"loan_id": loan_id, "result_id": loan["result_id"],
                "count": len(items), "exports": items,
                "note": "导出记录来源于链上事件；若为空表示该借出单尚无导出动作"}

    @app.get("/api/v1/loans/{loan_id}/export-download", dependencies=[Depends(require_service_header)])
    def loan_export_download(loan_id: str):
        """按借出单直接下载成果文件。

        与 /api/v1/exports/{export_id}/download 的区别：那条要先建导出任务拿 export_id，
        这条给「借出归还」场景用，用户已经在借出单上下文里，直接给文件。
        """
        try:
            loan = loan_store.get(loan_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown loan_id: {loan_id}")
        record = task_store.get_artifact_record(loan["result_id"])
        artifact = record.get("artifact") if record else None
        if artifact is None:
            raise HTTPException(status_code=404, detail="借出单关联的成果已不存在")
        if getattr(artifact, "status", None) != "APPROVED":
            raise HTTPException(status_code=403,
                                detail=f"成果状态为 {getattr(artifact, 'status', 'UNKNOWN')}，未通过审批不可导出")
        output = Path(getattr(artifact, "output", "") or "")
        if not output.exists():
            raise HTTPException(status_code=404, detail="成果文件在磁盘上已缺失")
        carrier = output
        if output.is_dir():
            inner = sorted(p for p in output.rglob("*") if p.is_file()
                           and p.suffix.lower() in {".gpkg", ".shp", ".tif", ".tiff", ".geojson"})
            if not inner:
                raise HTTPException(status_code=404, detail="成果目录内没有可下载的数据文件")
            carrier = inner[0]
        _loan_chain().append("LOAN_EXPORTED", loan_id=loan_id, artifact_id=loan["result_id"],
                             project_id=loan["project_id"], borrower=loan["borrower"],
                             filename=carrier.name)
        audit_record("LOAN_EXPORTED", actor=loan["borrower"], project_id=loan["project_id"],
                     loan_id=loan_id, artifact_id=loan["result_id"], filename=carrier.name)
        return FileResponse(carrier, filename=carrier.name)

    @app.get("/api/v1/loans/{loan_id}", dependencies=[Depends(require_service_header)])
    def get_loan(loan_id: str) -> dict:
        try:
            loan = loan_store.get(loan_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown loan_id: {loan_id}")
        overdue, days = loan_store._due_passed(loan["due_at"]) if loan["status"] != STATUS_RETURNED \
            else (False, 0)
        return {**loan, "overdue": overdue, "overdue_days": days if overdue else 0,
                "events": [r for r in _loan_chain().search(loan_id=loan_id)],
                "chain": _loan_chain().verify()}

    @app.post("/api/v1/loans/{loan_id}/check-in", dependencies=[Depends(require_service_header)])
    def check_in_loan(loan_id: str, request: LoanCheckInRequest) -> dict:
        """归还核验：现场检水印 → 比对借出记录/结果库 → 判身份与期限 → 落结论并上链。"""
        try:
            loan = loan_store.get(loan_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown loan_id: {loan_id}")
        if loan["status"] == STATUS_RETURNED:
            raise HTTPException(
                status_code=409,
                detail=f"loan {loan_id} already checked in at {loan['returned_at']} "
                       f"(verdict {loan['return_verdict']})")
        if loan["status"] == "CANCELLED":
            raise HTTPException(status_code=409, detail=f"loan {loan_id} is cancelled")

        # 期望水印号：请求优先，否则用借出单登记的
        expected = (request.watermark_text or loan.get("expected_watermark")
                    or _artifact_watermark_text(loan["result_id"]) or "")
        probe = WatermarkQueryRequest(
            project_id=loan["project_id"], user_id=request.user_id or "checkin",
            data_type=request.data_type, classification=request.classification,
            source_path=request.source_path,
            watermark_text=expected or "0",
        )
        # 先把载体路径验明，再进水印判定。路径不存在、越出允许根目录、压缩包未解包、
        # 类型与声明不符，都属于「调用方传错了」，必须直接返回给调用方。
        # 若不在此处拦下，下面的 except 会把它们一并吞成 detected=False，最终落成
        # RETURNED_NOT_OURS——把「路径写错」误报成「这不是本系统产出的数据」。
        try:
            _query_source(probe)
        except HTTPException as exc:
            if exc.status_code == 404:
                raise HTTPException(
                    status_code=422,
                    detail=(f"归还核验的载体路径不可用：{request.source_path}"
                            "（服务器上找不到该文件或目录，请确认文件已上传并落盘、路径完整）"),
                ) from exc
            raise
        detected = False
        matched: str | None = None
        probe_error = ""
        try:
            # 借出单登记了 result_id：把原始成果当作几何参考图传进去，
            # 裁剪/缩放/旋转/转 PNG 才能被还原后认出，而不是逐载体死等超时。
            reference_path = None
            try:
                _rec = task_store.get_artifact_record(loan["result_id"])
                _art = (_rec or {}).get("artifact")
                if isinstance(_art, dict):
                    _out = _art.get("output") or _art.get("destination")
                    if _out and Path(str(_out)).exists():
                        reference_path = Path(str(_out))
            except Exception:  # noqa: BLE001 - 参考图取不到就退回无参考图的有界判定
                reference_path = None
            probe_result = _verify_uploaded_watermark(probe, reference=reference_path)
            detected = bool(probe_result.get("verified")
                            or probe_result.get("pixel_matched", 0) > 0
                            or probe_result.get("matched", 0) > 0)
            matched = probe_result.get("watermark_text") or probe_result.get("expected_watermark") \
                or (expected if detected else None)
        except HTTPException as exc:
            probe_result = {"error": str(exc.detail), "status": exc.status_code}
            probe_error = str(exc.detail)
            detected = False
        except Exception as exc:
            probe_result = {"error": f"{type(exc).__name__}: {exc}"}
            probe_error = probe_result["error"]
            detected = False

        # 归位：优先用借出单登记的 result_id；水印库能反查到别的结果时如实报出。
        result_id_of_found = loan["result_id"] if detected else None
        carrier_hash = ""
        try:
            carrier_hash = loan_store.hash_carrier(request.source_path)
        except OSError:
            carrier_hash = ""

        outcome = loan_store.check_in(
            loan_id, expected_watermark=expected or None, matched_watermark=matched,
            identity_found=detected, result_id_of_found=result_id_of_found,
            carrier_hash=carrier_hash, operator=request.operator, detected=detected,
            note=request.note, probe_failed=probe_error,
        )
        outcome["watermark_probe"] = probe_result
        if loan.get("carrier_hash") and carrier_hash:
            outcome["carrier_hash_match"] = (loan["carrier_hash"] == carrier_hash)
            outcome["detail"].append(
                "载体哈希与借出时登记的一致" if outcome["carrier_hash_match"]
                else "⚠ 载体哈希与借出时登记的不一致（内容被改动过）")

        _loan_chain().append("DATA_RETURNED", loan_id=loan_id, artifact_id=loan["result_id"],
                             project_id=loan["project_id"], borrower=loan["borrower"],
                             verdict=outcome["verdict"], identity_confirmed=outcome["identity_confirmed"],
                             overdue=outcome["overdue"], overdue_days=outcome["overdue_days"],
                             carrier_hash=carrier_hash, operator=request.operator,
                             source_path=str(request.source_path), data_type=request.data_type,
                             probe_error=probe_error, detail=outcome["detail"])
        if outcome["overdue"]:
            _loan_chain().append("LOAN_OVERDUE", loan_id=loan_id, artifact_id=loan["result_id"],
                                 project_id=loan["project_id"], borrower=loan["borrower"],
                                 due_at=loan["due_at"], overdue_days=outcome["overdue_days"],
                                 at_return=True)
        audit_record("DATA_RETURNED", actor=request.operator or request.user_id,
                     project_id=loan["project_id"], artifact_id=loan["result_id"],
                     loan_id=loan_id, verdict=outcome["verdict"],
                     identity_confirmed=outcome["identity_confirmed"],
                     overdue=outcome["overdue"], source_path=str(request.source_path),
                     data_type=request.data_type, probe_error=probe_error)
        outcome["chain"] = _loan_chain().verify()
        return outcome

    @app.get("/api/v1/provenance/check", dependencies=[Depends(require_service_header)])
    def provenance_check(source_path: str, data_type: str, classification: str,
                         watermark_text: str | None = None,
                         project_id: str | None = None) -> dict:
        """不针对某张借出单的"这份数据是不是我们的"查询。

        返回：水印是否检出 + 命中哪些借出流转记录 + 当前是否在借/已归还/超期。
        """
        probe = WatermarkQueryRequest(
            project_id=project_id or "provenance", user_id="provenance-check",
            data_type=data_type, classification=classification,
            source_path=source_path, watermark_text=watermark_text or "0",
        )
        try:
            probe_result = _verify_uploaded_watermark(probe)
            error = None
        except HTTPException as exc:
            probe_result, error = {}, {"status": exc.status_code, "detail": str(exc.detail)}
        detected = bool(probe_result.get("verified")
                        or probe_result.get("pixel_matched", 0) > 0
                        or probe_result.get("matched", 0) > 0)

        loan_store.refresh_overdue()
        # 用检出到的水印号与项目过滤借出记录，作为"流转记录"证据
        loans = loan_store.list(project_id=project_id) if project_id else loan_store.list()
        related = [r for r in loans if watermark_text and
                   str(r.get("expected_watermark") or "") == str(watermark_text)]
        chain_hits = _loan_chain().search(project_id=project_id) if project_id else []
        chain_hits = [r for r in chain_hits
                      if r.get("event_type") in {"DATA_LENT", "DATA_RETURNED", "LOAN_OVERDUE"}]

        open_now = [r for r in related if r["status"] in {STATUS_LENT, STATUS_OVERDUE}]
        return {
            "is_probably_ours": detected,
            "watermark_detected": detected,
            "watermark_probe": probe_result,
            "probe_error": error,
            "matching_loans": related,
            "currently_lent": bool(open_now),
            "open_loans": open_now,
            "loan_events": chain_hits,
            "chain": _loan_chain().verify(),
            "conclusion": (
                "载体上检出水印，且存在对应借出流转记录" if detected and related else
                "载体上检出水印，但查不到借出流转记录（可能是系统内部产物，未外借）" if detected else
                "未检出水印：无法证明是本系统数据（无借出记录可佐证）"
            ),
        }

    @app.get("/api/v1/provenance/check-geometric", dependencies=[Depends(require_service_header)])
    def provenance_check_geometric(source_path: str, reference_path: str,
                                   expected_watermark: str,
                                   project_id: str | None = None) -> dict:
        """溯源查询（抗几何变形版）：数据被裁剪/旋转/缩放后仍能判定来源。

        与 `/provenance/check` 的区别：
          本端点要求提供**系统内登记的参考图**（加水印后的原图或其登记副本），
          先用 SIFT 配准把可疑数据还原到参考几何，再走标准水印提取。

        `expected_watermark` 必须是系统登记的载荷值（十进制或十六进制）。

        返回：是否命中、解出的载荷、检测到的几何变换（角度/缩放/内点数）、
              使用的还原候选、以及对应的流转记录结论。
        """
        from . import geo_restore_extract as geo

        src = Path(source_path)
        ref = Path(reference_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"source not found: {source_path}")
        if not ref.exists():
            raise HTTPException(status_code=404, detail=f"reference not found: {reference_path}")

        try:
            expected = _parse_watermark_int(expected_watermark)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400,
                                detail=f"invalid expected_watermark: {exc}") from exc

        # workspace 是 create_app 的入参（局部变量），不是 service 属性；
        # 从 loan_store 的库路径反推根目录，避免依赖不存在的属性。
        root = Path(loan_database_path).parent.parent if loan_database_path else Path(workspace)
        work = root / "provenance_geom" / __import__("uuid").uuid4().hex
        _prune_scratch_dirs(root / "provenance_geom")
        work.mkdir(parents=True, exist_ok=True)
        # 网格标签从「加水印的参考图」继承；若参考图本身不带标签，
        # 退化到参考图自身标签（提取器会用其 transform 兜底）。
        tag_source = ref if geo._ref_tags(ref) else None

        result = geo.extract_with_restore(ref, src, expected, work,
                                          tag_source=tag_source)
        geometry = result.get("geometry") or {}

        # QIM 通道没命中时，按 TrustMark 载体通道再试一次。
        # 这两条是并存的两套嵌入方案：DOM/正射成果走 TrustMark（256px 载体 + SIFT 还原），
        # 只有 DEM/早期 QIM 成果才带 geo_restore_extract 能读的格网。用 QIM 去测 TrustMark
        # 成果必然 0 命中（实测 1024² 原图在 check-any 能检出、在 check-geometric 判未检出），
        # 前端看到的就是"带参考图反而检不出来"。
        if not result.get("matches"):
            # 必须走沙箱子进程：torch 只装在 GDS_TRUSTMARK_PYTHON 指向的 venv 里，
            # API 进程本身（.deploy-venv）没有 torch，在进程内调 restore_and_decode
            # 会直接 ModuleNotFoundError 被吞掉，表现就是"带了参考图仍然未检出"。
            try:
                from .trustmark_engine import (TrustMarkGeoTIFFEngine,
                                               TrustMarkEngineUnavailableError)
                try:
                    eng = TrustMarkGeoTIFFEngine()
                except TrustMarkEngineUnavailableError as exc:
                    # 构造失败也要留痕：否则响应里看不出"回退没跑"，
                    # 前端只看到"未检出"，无法区分引擎不可用与确实没水印。
                    eng = None
                    result["trustmark_fallback"] = {
                        "attempted": False, "found": False,
                        "error": f"engine_unavailable: {str(exc)[:200]}"}
                if eng is not None:
                    rr = eng._call(["geo-restore", str(ref), str(src), f"{expected:08X}"],
                                   timeout=1800)
                    if rr.get("found"):
                        result = {
                            "matches": True,
                            "watermark": f"{expected:08X}",
                            "via": "trustmark:" + str(rr.get("via") or "restore"),
                        }
                        geometry = rr.get("geometry") or {
                            "scale": rr.get("scale"), "hits": rr.get("hits"),
                            "total": rr.get("total"), "restored": True,
                        }
                        result["geometry"] = geometry
                    else:
                        result["trustmark_fallback"] = {
                            "attempted": True, "found": False,
                            "via": rr.get("via"), "hits": rr.get("hits"),
                            "total": rr.get("total"), "error": rr.get("error"),
                        }
            except Exception as exc:  # noqa: BLE001
                # 不能静默吞掉：回退通道失败必须能在响应里看到，否则前端只会拿到
                # "未检出"而不知道是缺参考图、引擎不可用，还是回退本身报错。
                result["trustmark_fallback"] = {
                    "attempted": True, "found": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }

        loan_store.refresh_overdue()
        loans = loan_store.list(project_id=project_id) if project_id else loan_store.list()
        related = [r for r in loans
                   if str(r.get("expected_watermark") or "") == str(expected)]

        detected = bool(result.get("matches"))
        open_now = [r for r in related if r["status"] in {STATUS_LENT, STATUS_OVERDUE}]
        return {
            "is_probably_ours": detected,
            "watermark_detected": detected,
            "expected_watermark": expected,
            "resolved_watermark": result.get("watermark"),
            "matched": detected,
            "geometry": {
                "angle_deg": geometry.get("angle"),
                "scale": geometry.get("scale"),
                "inliers": geometry.get("inliers"),
                "matches": geometry.get("matches"),
                "restored": geometry.get("restored"),
            },
            "restored_via": result.get("via"),
            "trustmark_fallback": result.get("trustmark_fallback"),
            "work_dir": str(work),
            "matching_loans": related,
            "currently_lent": bool(open_now),
            "open_loans": open_now,
            "conclusion": (
                "数据经几何变换（裁剪/旋转/缩放）后仍检出水印，判定为本系统流出"
                if detected else
                "未检出登记水印：无法证明该数据由本系统流出"
            ),
        }

    @app.get("/api/v1/provenance/check-any", dependencies=[Depends(require_service_header)])
    def provenance_check_any(source_path: str, expected_watermark: str,
                             reference_path: str | None = None,
                             project_id: str | None = None) -> dict:
        """溯源查询（四类数据统一入口）：按文件类型自动分派到对应检测通道。

        分派规则与各通道的真实边界（实测口径，不跨类外推）：

          · GeoTIFF/栅格 → TrustMark 载体 + SIFT 配准还原
              参考图可选；不传则用 GDS_TM_PAYLOAD 标签自校验。
              抗旋转任意角度、缩放 0.5–2.0x、裁剪 10–25%。
          · DEM (含 GDS_DEM_WATERMARK 标签) → 整数值 LSB + SIFT 配准还原
              只有**放大**或**旋转**能还原；缩小会因重采样把 LSB 抹掉，无法还原。
          · GPKG → 属性字段 `_wm` 存在性
              几何任意变换不影响；字段被删除则失效。
          · OSGB → 内嵌 JPEG 纹理 DCT
              改节点矩阵/重编码 JPEG 不失效；转 PNG 或纹理缩小会失效。

        `reference_path` 只对 GeoTIFF/DEM 有意义，GPKG/OSGB 忽略。
        """
        from . import geo_restore_extract as geo
        from . import provenance_any as pa

        # 先校验水印值，再查文件：参数错误应返回 400，不该被文件检查掩盖成 404。
        try:
            expected = _parse_watermark_int(expected_watermark)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400,
                                detail=f"invalid expected_watermark: {exc}") from exc

        src = Path(source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"source not found: {source_path}")

        root = Path(loan_database_path).parent.parent if loan_database_path else Path(workspace)
        work = root / "provenance_any" / __import__("uuid").uuid4().hex
        _prune_scratch_dirs(root / "provenance_any")
        work.mkdir(parents=True, exist_ok=True)

        result = pa.check_any(src, expected, Path(reference_path) if reference_path else None, work)

        loan_store.refresh_overdue()
        loans = loan_store.list(project_id=project_id) if project_id else loan_store.list()
        related = [r for r in loans
                   if str(r.get("expected_watermark") or "") == str(expected)]

        detected = bool(result.get("found"))
        open_now = [r for r in related if r["status"] in {STATUS_LENT, STATUS_OVERDUE}]
        return {
            "is_probably_ours": detected,
            "watermark_detected": detected,
            "data_class": result.get("data_class"),
            "channel": result.get("channel"),
            "expected_watermark": expected,
            "resolved_watermark": result.get("decoded"),
            "matched": detected,
            "capability": result.get("capability"),
            "detail": result.get("detail"),
            "work_dir": str(work),
            "matching_loans": related,
            "currently_lent": bool(open_now),
            "open_loans": open_now,
            "conclusion": result.get("conclusion"),
        }

    @app.get("/api/v1/provenance/check-block-sync", dependencies=[Depends(require_service_header)])
    def provenance_check_block_sync(source_path: str, reference_path: str) -> dict:
        """溯源查询（块内自同步通道）：块被裁/被重采样时仍有较大概率恢复。

        与 `/provenance/check-geometric` 的区别：
          · check-geometric 走参考图配准 + QIM 直接提取，块必须完整
          · 本端点走块内自同步：每块自带序号与量化步长，配准只为求偏移

        实测能力（真实 Landsat8bit 2048²，载荷 20261016）：
          裁剪 ≥35%、缩放 0.5/0.75/1.5、旋转 15/30/45/90° → 检出 9/10
          裁剪 25%（有效块仅 4 个且在边缘）→ 检不出

        **必须提供 reference_path**：对齐容差实测仅 ±1 像素，
        没有参考图无法确定块边界（盲搜已实测不可行）。
        """
        from . import block_sync_watermark as bsync

        src = Path(source_path)
        ref = Path(reference_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"source not found: {source_path}")
        if not ref.exists():
            raise HTTPException(status_code=404, detail=f"reference not found: {reference_path}")

        result = bsync.extract_with_registration(src, ref)
        return {
            "is_probably_ours": bool(result.get("found")),
            "watermark_detected": result.get("watermark"),
            "matched": bool(result.get("found")),
            "votes": result.get("votes"),
            "tiles_located": result.get("tiles_located"),
            "offset_guess": result.get("offset_guess"),
            "geometry": result.get("geometry"),
            "candidates": result.get("payloads"),
            "reason": result.get("reason"),
            "channel": "block-sync",
            "conclusion": (
                "块内自同步水印命中，判定为本系统流出"
                if result.get("found") else
                "未在块内同步通道检出水印：该数据可能已被破坏性处理"
            ),
        }

    @app.get("/api/v1/provenance/check-multi-domain", dependencies=[Depends(require_service_header)])
    def provenance_check_multi_domain(source_path: str, reference_path: str | None = None) -> dict:
        """溯源查询（多域冗余通道）：频域被破坏时由空域兜底。

        两个域互相独立：
          · 频域（DCT-SVD）—— 抗几何变换，但对强模糊/重采样敏感
          · 空域（块内均值 QIM）—— 抗模糊，但只承载低 16 位载荷

        返回命中的通道与对应载荷。空域命中时载荷是登记值的低 16 位，
        调用方需按 `(expected & 0xFFFF)` 比对。

        `reference_path` 可选：不提供时只走空域通道（无需配准）。
        """
        from . import multi_domain_watermark as md

        src = Path(source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"source not found: {source_path}")
        ref = Path(reference_path) if reference_path else None
        if ref is not None and not ref.exists():
            raise HTTPException(status_code=404, detail=f"reference not found: {reference_path}")

        result = md.extract(src, ref)
        return {
            "is_probably_ours": bool(result.get("found")),
            "watermark_detected": result.get("watermark"),
            "matched": bool(result.get("found")),
            "channel": result.get("channel"),
            "frequency": result.get("frequency"),
            "spatial": result.get("spatial"),
            "note": result.get("note"),
            "conclusion": (
                f"多域水印命中（通道 {result.get('channel')}），判定为本系统流出"
                if result.get("found") else
                "两个域均未检出：数据可能已被破坏性处理"
            ),
        }

    @app.get("/api/v1/collusion/codebook", dependencies=[Depends(require_service_header)])
    def collusion_generate(seed: str, users: str, length: int = 4096) -> dict:
        """生成合谋抗性码字库（Tardos）。users 用逗号分隔。"""
        from . import collusion_code as cc

        user_list = [u.strip() for u in users.split(",") if u.strip()]
        if not user_list:
            raise HTTPException(status_code=422, detail="users must not be empty")
        cb = cc.generate_codebook(user_list, length=length, seed=seed)
        # 必须落盘：identify 依赖 codebooks.sqlite3 复现码字库，
        # 只生成不保存会让「生成 → 定位」流程断开（真实缺陷，已修）。
        store = cc.CodebookStore(Path(workspace) / "state" / "codebooks.sqlite3")
        store.save(cb)
        return {"seed": cb["seed"], "length": cb["length"], "t": cb["t"],
                "users": list(cb["codes"]), "saved": True,
                "note": "码字库已落盘到 state/codebooks.sqlite3，可直接用于 identify"}

    @app.get("/api/v1/collusion/identify", dependencies=[Depends(require_service_header)])
    def collusion_identify(seed: str, extracted_bits: str,
                           threshold: float | None = None) -> dict:
        """用提取出的码字定位合谋者。extracted_bits 为 0/1 字符串。"""
        from . import collusion_code as cc

        codebook_path = Path(workspace) / "state" / "codebooks.sqlite3"
        store = cc.CodebookStore(codebook_path)
        cb = store.load(seed)
        if cb is None:
            raise HTTPException(status_code=404, detail=f"codebook not found: {seed}")

        bits = [int(ch) for ch in extracted_bits if ch in "01"]
        if len(bits) != cb["length"]:
            raise HTTPException(
                status_code=422,
                detail=f"extracted_bits length {len(bits)} != codebook length {cb['length']}")
        return cc.identify(cb, bits,
                           threshold=cc._COLLUSION_Z if threshold is None else threshold)

    # ── 外部编号映射（服务器1 fileId → 本系统 artifact） ─────────────
    @app.post("/api/v1/external-file-ids", status_code=201,
              dependencies=[Depends(require_service_header)])
    def link_external_file_id(external_id: str, artifact_id: str,
                              note: str | None = None) -> dict:
        """把服务器1 侧的 fileId 登记到本系统的成果上。

        服务器1 有自己的编号体系（如 `file_1792552013127`），本系统不认识。
        登记一次之后，沙箱接口就可以直接用那个编号调用，无需改造对方前端。
        """
        record = task_store.get_artifact_record(artifact_id)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=(f"artifact not found: {artifact_id}. "
                        "artifact_id 取自 GET /api/v1/watermark-sandbox/candidates"))
        out = task_store.link_external_file_id(external_id, artifact_id, "server1", note)
        out["artifact"] = {
            "data_type": record.get("data_type"),
            "project_id": record.get("project_id"),
            "output": record.get("output"),
        }
        return out

    @app.get("/api/v1/external-file-ids", dependencies=[Depends(require_service_header)])
    def list_external_file_ids(limit: int = 200) -> dict:
        """列出已登记的外部编号映射。"""
        items = task_store.list_external_file_ids(limit)
        return {"count": len(items), "items": items}

    @app.get("/api/v1/watermark-sandbox/candidates", dependencies=[Depends(require_service_header)])
    def watermark_sandbox_candidates(limit: int = 50, project_id: str | None = None,
                                     include_inbox: bool = True) -> dict:
        """列出沙箱可检测的文件：**有水印的和无水印的都在列**。

        两个来源：
          1. 系统内已登记成果的 output（成品，通常带水印）
          2. 沙箱样本目录 `sandbox_inbox`（外部投递的攻击样/原件，可能无水印）

        每条都带 `has_watermark` 判定字段，前端可据此分组展示。
        「无水印」是有效用例 —— 用来验证检测器不会误报，不是异常数据。
        """
        rows = list(task_store.load_artifacts().values())
        items = []
        for record in rows:
            if project_id and record.get("project_id") != project_id:
                continue
            carrier = Path(str(record.get("output") or record.get("path") or ""))
            if not carrier.is_file():
                continue
            items.append({
                "fileId": f"file_{record.get('artifact_id', '')}",
                "artifact_id": record.get("artifact_id"),
                "project_id": record.get("project_id"),
                "filename": carrier.name,
                "absolute_path": str(carrier),
                "size_bytes": carrier.stat().st_size,
                "watermark_text": record.get("watermark_text"),
                # 登记成果默认按「本系统产出」处理；实际有无水印由 detect/verify 判定
                "has_watermark": None,
                "origin": "artifact",
                "watermarkSandboxReady": True,
            })
            if len(items) >= limit:
                break

        if include_inbox and len(items) < limit:
            inbox = _SANDBOX_INBOX
            if inbox.exists():
                for p in sorted(inbox.iterdir()):
                    if not p.is_file() or p.suffix.lower() not in _SUFFIX_DATA_TYPE:
                        continue
                    if project_id:
                        continue  # 沙箱目录不归属具体项目
                    items.append({
                        "fileId": str(p),
                        "artifact_id": None,
                        "project_id": None,
                        "filename": p.name,
                        "absolute_path": str(p),
                        "size_bytes": p.stat().st_size,
                        "watermark_text": None,
                        "has_watermark": None,
                        "origin": "sandbox_inbox",
                        "watermarkSandboxReady": True,
                    })
                    if len(items) >= limit:
                        break

        return {"count": len(items), "items": items,
                "inbox": str(_SANDBOX_INBOX),
                "note": ("两类文件都可检测：origin=artifact 是系统产出的成品（通常带水印），"
                         "origin=sandbox_inbox 是外部投递的攻击样（可能无水印）。"
                         "has_watermark 为 null 表示需调用 detect/verify 才能判定；"
                         "无水印文件的正确结果是 samples=0 / verified=false，属有效用例。")}

    @app.get("/api/v1/files/{file_id}/watermark-sandbox/preview",
             dependencies=[Depends(require_service_header)])
    def watermark_sandbox_preview(file_id: str) -> dict:
        """沙箱预览：确认文件可读、可解析，并给出检测前的元信息。

        `file_id` 形如 `file_<result_id>`，或直接传绝对路径（以盘符开头）。
        必须在秒级返回 —— 服务器1报告过 preview 超时，这里只做轻量探测。
        """
        path, record = _resolve_sandbox_target(file_id, task_store)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"file not found: {path}")

        info = {
            "fileId": file_id,
            "absolute_path": str(path),
            "size_bytes": path.stat().st_size,
            "readable": os.access(path, os.R_OK),
            "suffix": path.suffix.lower(),
            "sandbox_enabled": sandbox_compute.enabled(),
            "watermarkSandboxReady": True,
        }
        # 若调用方传的是归档或目录，这里已经是解包后的真实数据文件，
        # 明确告知原本传入的与最终解析到的不是同一个路径。
        original = (file_id or "").strip()
        if original and len(original) > 2 and original[1] == ":" and Path(original) != path:
            info["requested_path"] = original
            info["resolved_from_archive"] = True
        if record:
            info.update({
                "result_id": record.get("result_id"),
                "project_id": record.get("project_id"),
                "data_type": record.get("data_type"),
                "classification": record.get("classification"),
                "watermark_text": record.get("watermark_text"),
                "registered": True,
            })
        else:
            info["registered"] = False
            info["note"] = "该路径不在系统登记成果中，只做只读探测"

        # 大文件不在这里做全量解析，只抽样确认可读
        try:
            with open(path, "rb") as fh:
                fh.read(4096)
            info["sample_read_ok"] = True
        except OSError as exc:
            info["sample_read_ok"] = False
            info["read_error"] = str(exc)
        return info

    @app.post("/api/v1/files/{file_id}/watermark-sandbox/detect",
              dependencies=[Depends(require_service_header)])
    def watermark_sandbox_detect(file_id: str, watermark_text: str | None = None) -> dict:
        """沙箱检测：在受限账户下判断文件是否携带本系统水印。

        与 `provenance/check` 的区别：本端点不查流转记录，只回答「有没有水印」，
        供服务器1的沙箱页做快速判定。深度比对仍走 provenance/check。
        """
        path, record = _resolve_sandbox_target(file_id, task_store)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"file not found: {path}")

        expected = watermark_text or (record or {}).get("watermark_text")
        dtype, dclass = _guess_type(path, record)
        request = WatermarkQueryRequest(
            project_id=str((record or {}).get("project_id") or "unregistered"),
            user_id="watermark-sandbox",
            source_path=str(path),
            data_type=dtype,
            classification=dclass,
            watermark_text=expected or "0",
        )
        result = _verify_uploaded_watermark(request)
        return {
            "fileId": file_id,
            "absolute_path": str(path),
            "suspected": bool(result.get("verified")),
            "verified": bool(result.get("verified")),
            "method": result.get("method"),
            "samples": result.get("samples"),
            "matched": result.get("matched"),
            "detail": result,
            "conclusion": ("检出本系统水印" if result.get("verified")
                           else "未检出本系统水印"),
        }

    @app.post("/api/v1/files/{file_id}/watermark-sandbox/verify",
              dependencies=[Depends(require_service_header)])
    def watermark_sandbox_verify(file_id: str, watermark_text: str | None = None) -> dict:
        """沙箱验证：不止判断有无，还要与登记水印号逐条比对并给出结论。"""
        path, record = _resolve_sandbox_target(file_id, task_store)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"file not found: {path}")

        expected = watermark_text or (record or {}).get("watermark_text")
        if not expected:
            raise HTTPException(
                status_code=422,
                detail="watermark_text required: file is not registered, pass the expected watermark")

        dtype, dclass = _guess_type(path, record)
        request = WatermarkQueryRequest(
            project_id=str((record or {}).get("project_id") or "unregistered"),
            user_id="watermark-sandbox",
            source_path=str(path),
            data_type=dtype,
            classification=dclass,
            watermark_text=expected,
        )
        result = _verify_uploaded_watermark(request)
        verified = bool(result.get("verified"))
        return {
            "fileId": file_id,
            "absolute_path": str(path),
            "expected_watermark": expected,
            "verified": verified,
            "method": result.get("method"),
            "samples": result.get("samples"),
            "matched": result.get("matched"),
            "detail": result,
            "conclusion": (
                f"水印验证通过：载体与登记号 {expected} 一致"
                if verified else
                f"水印验证未通过：载体与登记号 {expected} 不一致或无法读取"),
        }

    # ── 沙箱攻击样投递（2026-10-22 新增） ───────────────────────────
    # 背景：开发机没有 V: 盘，无法把攻击样本拷到服务器2 的绝对路径下。
    # 此前只能由运维手工落盘再告知路径，测试方无法自助。
    # 下面两条提供稳定入口：投递 → 落到固定目录 → 返回服务器2 绝对路径，
    # 该路径可直接用于 provenance/check 与沙箱 detect/verify。
    # 目录常量 _SANDBOX_INBOX 定义在模块级，供 candidates 复用。

    @app.post("/api/v1/sandbox-samples", status_code=201,
              dependencies=[Depends(require_service_header)])
    async def sandbox_sample_upload(file: UploadFile = File(...),
                                    note: str | None = None) -> dict:
        """投递一个攻击样本到固定目录，返回服务器2 侧绝对路径。

        上传后落盘为 `V:\\geo_data_security\\sandbox_inbox\\<原文件名>`，
        同名文件会被覆盖（方便同一攻击场景反复重测）。
        """
        _SANDBOX_INBOX.mkdir(parents=True, exist_ok=True)
        name = Path(file.filename or "sample.bin").name  # 只取文件名，杜绝路径穿越
        if not name or name in {".", ".."}:
            raise HTTPException(status_code=422, detail="invalid filename")
        target = _SANDBOX_INBOX / name
        size = 0
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with tmp.open("wb") as fh:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    size += len(chunk)
            tmp.replace(target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        digest = _hash_file(target)
        return {
            "filename": name,
            "absolute_path": str(target),
            "size_bytes": size,
            "sha256": digest,
            "note": note,
            "inbox": str(_SANDBOX_INBOX),
            "usage": "把 absolute_path 原样传给 provenance/check 或沙箱 detect/verify",
        }

    @app.get("/api/v1/sandbox-samples", dependencies=[Depends(require_service_header)])
    def sandbox_sample_list() -> dict:
        """列出沙箱样本目录，给出每个文件的服务器2 绝对路径。"""
        items = []
        if _SANDBOX_INBOX.exists():
            for p in sorted(_SANDBOX_INBOX.iterdir()):
                if p.is_file():
                    items.append({
                        "filename": p.name,
                        "absolute_path": str(p),
                        "size_bytes": p.stat().st_size,
                    })
        return {"inbox": str(_SANDBOX_INBOX), "count": len(items), "items": items}

    @app.get("/api/v1/precision/bounds", dependencies=[Depends(require_service_header)])
    def precision_bounds_endpoint(upload_id: str | None = None, data_type: str | None = None,
                                  width: int | None = None, height: int | None = None,
                                  features: int | None = None, points: int | None = None,
                                  watermark_enabled: bool = True,
                                  watermark_mode: str = "FORENSIC") -> dict:
        """降精度安全范围：前端据此渲染"最小—最大"选择器。

        给了 `upload_id` 就自动读该上传的载体统计（栅格尺寸 / 要素数）；也可手工传。
        返回的 ``min_*`` / ``max_*`` 都是**保证水印仍可嵌入**的范围，不是形式校验的边界。
        """
        from . import precision_bounds

        resolved_type = data_type
        counts: dict[str, Any] = {}
        if upload_id:
            upload = task_store.get_upload(upload_id)
            if upload is None:
                raise HTTPException(status_code=404, detail=f"unknown upload_id: {upload_id}")
            resolved_type = resolved_type or upload.get("data_type")
            stats = _carrier_stats(upload)
            counts = {k: v for k, v in stats.items() if v is not None}

        w = width if width is not None else counts.get("width")
        h = height if height is not None else counts.get("height")
        f = features if features is not None else counts.get("features")
        p = points if points is not None else counts.get("points")

        bounds = precision_bounds.bounds_for(resolved_type or "", width=w, height=h,
                                             features=f, points=p)
        payload: dict[str, Any] = {
            "data_type": resolved_type,
            "upload_id": upload_id,
            "carrier_counts": {"width": w, "height": h, "features": f, "points": p},
            "bounds": bounds,
            "watermark_safe": bool(bounds.get("feasible", bounds.get("known") is False)),
        }
        blocked_reason = _policy_blocked("DECIMALS", watermark_enabled, watermark_mode)
        payload["blocked"] = blocked_reason is not None
        if blocked_reason:
            # 受限时仍然返回真实载体边界：前端要能画出「关掉水印后的合法区间」。
            # 只把 message 带上，不抹掉 bounds 与 selectable。
            payload["message"] = blocked_reason
        # 把"客户可选"的两种模式口径一并给出，前端不用自己拼
        if resolved_type and resolved_type.upper() in {"GEOTIFF", "GTIFF", "TIF", "TIFF", "DOM", "DEM"}:
            payload["selectable"] = {
                "mode": "RESAMPLE",
                "scale_min": bounds.get("min_scale"),
                "scale_max": bounds.get("max_scale"),
                "width_min": bounds.get("min_width"),
                "height_min": bounds.get("min_height"),
                "unit": "scale (0–1]",
            }
        elif resolved_type and resolved_type.upper() in {"SHP", "DLG", "GEOJSON", "GPKG", "GEOPACKAGE"}:
            payload["selectable"] = {
                "modes": ["DECIMALS", "DOUGLAS_PEUCKER"],
                "decimals_min": bounds.get("decimals_min"),
                "decimals_max": bounds.get("decimals_max"),
                "min_features_required": bounds.get("min_features_required"),
                "min_vertices_required": bounds.get("min_vertices_required"),
                "unit": "decimals (int) / tolerance (float)",
            }
        elif resolved_type and resolved_type.upper() == "POINT_CLOUD":
            payload["selectable"] = {
                "slots_required": bounds.get("slots_required"),
                "note": "每比特占 1 个坐标槽；槽位 = 48 + 8 × 水印载荷字节数",
                "unit": "coordinate slots",
            }
        return payload

    def _iter_jsonl(path: Path):
        """逐行解析 JSONL，跳过坏行。

        证据文件由后台线程并发 append，写中断会留下半行。一行坏数据不能让整个
        链状态看板 500（2026-11-12 实测：锚定回执里一行孤立的 `}` 让
        `/api/v1/chain/status` 直接崩溃）。跳过的行数打到日志，便于事后核对。
        """
        skipped = 0
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except (ValueError, TypeError):
                skipped += 1
                continue
            if isinstance(row, dict):
                yield row
        if skipped:
            print(f"warning: skipped {skipped} unparsable line(s) in {path}", flush=True)

    def _chain_file() -> Path:
        return Path(app.state.processing_service.trace_chain.path)

    def _anchor_rows(limit: int = 200) -> list[dict]:
        path = _chain_file().with_suffix(".anchors.jsonl")
        if not path.exists():
            return []
        # Background anchoring appends from several threads; a torn line must not turn the
        # chain status board into a 500. Read tolerantly - see trace_chain._iter_json_lines.
        rows: list[dict] = list(_iter_jsonl(path))
        return rows[-limit:]

    def _trace_index() -> dict[str, dict]:
        index: dict[str, dict] = {}
        path = _chain_file()
        if not path.exists():
            return index
        for record in _iter_jsonl(path):
            index[str(record.get("record_hash", "")).removeprefix("0x")] = record
        return index

    def _rpc(method: str, params: list | None = None) -> object:
        """直连 Besu JSON-RPC，只用于展示链状态，任何异常都不抛给调用方。"""
        url = os.environ.get("GDS_BESU_RPC_URL", "http://127.0.0.1:18545")
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
        try:
            request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=5) as response:
                body = json.loads(response.read().decode())
            return body.get("result")
        except Exception:
            return None

    @app.get("/api/v1/chain/status", dependencies=[Depends(require_service_header)])
    def chain_status() -> dict:
        """区块链状态：节点是否可达、链 ID、最新区块、已锚定条数、本地哈希链校验结果。

        这是"到底有没有上链"的唯一权威口径：node_reachable 为真且有锚定记录，
        才说明写链成功；只有本地哈希链而链上无记录，不算上链。
        """
        anchors = _anchor_rows(limit=100000)
        chain_id = _rpc("eth_chainId")
        block = _rpc("eth_blockNumber")
        latest = anchors[-1] if anchors else None
        return {
            "anchor_enabled": os.environ.get("GDS_BESU_ANCHOR_ENABLED", "0") == "1",
            "rpc_url": os.environ.get("GDS_BESU_RPC_URL", "http://127.0.0.1:18545"),
            "gateway_url": os.environ.get("GDS_BESU_ANCHOR_URL", ""),
            "node_reachable": bool(block),
            "chain_id": int(chain_id, 16) if isinstance(chain_id, str) else None,
            "block_number": int(block, 16) if isinstance(block, str) else None,
            "anchored_records": len(anchors),
            "last_anchor": None if not latest else {
                "record_hash": latest.get("record_hash"),
                "event_type": latest.get("event_type"),
                "timestamp": latest.get("timestamp"),
                "tx_hash": (latest.get("anchor") or {}).get("tx_hash"),
                "block_number": (latest.get("anchor") or {}).get("block_number"),
                "onchain": (latest.get("anchor") or {}).get("onchain"),
            },
            "local_chain": app.state.processing_service.trace_chain.verify(),
        }

    @app.get("/api/v1/chain/anchors", dependencies=[Depends(require_service_header)])
    def chain_anchors(project_id: str | None = None, result_id: str | None = None,
                      event_type: str | None = None, limit: int = 50, offset: int = 0) -> dict:
        """上链记录列表，按时间倒序。可按项目、成果、事件类型过滤。"""
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        rows = list(reversed(_anchor_rows(limit=100000)))
        index = _trace_index()
        items = []
        for row in rows:
            record = index.get(str(row.get("record_hash", "")).removeprefix("0x"), {})
            if project_id and record.get("project_id") != project_id:
                continue
            if result_id and record.get("artifact_id") != result_id:
                continue
            if event_type and row.get("event_type") != event_type:
                continue
            anchor = row.get("anchor") or {}
            items.append({
                "record_hash": row.get("record_hash"),
                "prev_hash": record.get("prev_hash"),
                "event_type": row.get("event_type"),
                "event_time": row.get("timestamp"),
                "artifact_id": record.get("artifact_id"),
                "project_id": record.get("project_id"),
                "tx_hash": anchor.get("tx_hash"),
                "block_number": anchor.get("block_number"),
                "onchain": anchor.get("onchain"),
                "events": anchor.get("events"),
                "status": anchor.get("status"),
            })
        total = len(items)
        return {"total": total, "limit": limit, "offset": offset,
                "items": items[offset:offset + limit]}

    @app.get("/api/v1/results/{result_id}/chain-proof", dependencies=[Depends(require_service_header)])
    def result_chain_proof(result_id: str) -> dict:
        """单个成果的上链凭据：是否上链、交易哈希、区块高度、本地链校验结果。

        归还核验、交付验收、客户稽核都用这一条来回答"这份数据到底上没上链"。
        """
        record = task_store.get_artifact_record(result_id)
        if record is None:
            raise _missing_result_http(result_id)
        artifact = record.get("artifact")
        output = Path(getattr(artifact, "output", "") or "")
        manifest = _load_manifest(output)
        record_hash = manifest.get("trace_record_hash") or ""
        # 锚定回执有两个来源：链旁的回执文件（最新一次）和成果清单里写入时的那次
        receipt = app.state.processing_service.trace_chain.anchor_for(record_hash) or manifest.get("besu_anchor")
        onchain = bool(receipt and receipt.get("onchain") is True and receipt.get("status") == 1)
        reason = ""
        if not onchain:
            if not manifest:
                reason = "成果清单缺失，无法确认上链状态"
            elif not record_hash:
                reason = "该成果没有生成链上事件记录"
            elif not receipt:
                reason = "本地有哈希链记录，但链上没有对应的锚定交易"
            else:
                reason = "锚定交易存在但未确认成功"
        return {
            "result_id": result_id,
            "project_id": record.get("project_id"),
            "data_type": (manifest.get("data_type") or getattr(artifact, "data_type", "")),
            "classification": manifest.get("classification") or record.get("classification"),
            "watermark_enabled": bool(manifest.get("watermark_enabled", True)),
            "expected_watermark": manifest.get("expected_watermark"),
            "on_chain": onchain,
            "reason": reason,
            "record_hash": record_hash,
            "anchor": None if not receipt else {
                "tx_hash": receipt.get("tx_hash"),
                "block_number": receipt.get("block_number"),
                "onchain": receipt.get("onchain"),
                "status": receipt.get("status"),
                "events": receipt.get("events"),
            },
            "local_chain": app.state.processing_service.trace_chain.verify(),
        }

    def _chain_view_allowed(request: Request) -> bool:
        """默认只放开本机与内网；GDS_CHAIN_VIEW_REQUIRE_TOKEN=1 时一律要令牌。"""
        if os.environ.get("GDS_CHAIN_VIEW_REQUIRE_TOKEN") == "1":
            return False
        host = (request.client.host if request.client else "") or ""
        if host in {"127.0.0.1", "::1", ""}:
            return True
        try:
            return ipaddress.ip_address(host).is_private
        except ValueError:
            return False

    def _chain_view_guard(request: Request, token: str | None) -> None:
        if _chain_view_allowed(request):
            return
        supplied = request.headers.get("X-Service-Token") or token or ""
        try:
            _require_service_token(supplied)
        except HTTPException:
            raise HTTPException(
                status_code=401,
                detail="查看存证看板需要令牌：本机或内网地址可直接打开，其他来源请在地址后加 ?token=...")

    def _chain_payload(limit_results: int = 200, limit_events: int = 15) -> dict:
        """看板数据：链状态 + 每份成果是否上链 + 最近链上事件。"""
        artifacts = task_store.load_artifacts()
        rows = []
        for key, row in list(artifacts.items())[-limit_results:][::-1]:
            artifact = row["artifact"]
            manifest = _load_manifest(Path(artifact.output))
            record_hash = manifest.get("trace_record_hash")
            receipt = app.state.processing_service.trace_chain.anchor_for(record_hash) or manifest.get("besu_anchor")
            anchored = bool(receipt and receipt.get("onchain") is True and receipt.get("status") == 1)
            reason = ""
            if not anchored:
                reason = "未生成链上事件" if not record_hash else "链上无对应交易"
            rows.append({
                "result_id": key,
                "project_id": row.get("project_id"),
                "data_type": manifest.get("data_type") or getattr(artifact, "data_type", ""),
                "watermark_enabled": bool(manifest.get("watermark_enabled", True)),
                "anchor": None if not receipt else {
                    "tx_hash": receipt.get("tx_hash"),
                    "block_number": receipt.get("block_number"),
                    "onchain": receipt.get("onchain"),
                    "status": receipt.get("status"),
                },
                "reason": reason,
            })
        return {
            "status": chain_status(),
            "results": rows,
            "events": chain_anchors(limit=limit_events).get("items", []),
            "generated_at": now_local(millis=True),
        }

    @app.get("/chain-view", include_in_schema=False)
    def chain_view(request: Request, token: str | None = None) -> HTMLResponse:
        """区块链存证图：区块之间的链接关系用图形展示，打开即可看。"""
        _chain_view_guard(request, token)
        return HTMLResponse(render_chain_graph(_chain_payload(limit_events=12)))

    @app.get("/chain-view/table", include_in_schema=False)
    def chain_view_table(request: Request, token: str | None = None) -> HTMLResponse:
        """存证明细表：逐份成果列出是否上链、区块高度与交易哈希。"""
        _chain_view_guard(request, token)
        return HTMLResponse(render_chain_board(_chain_payload()))

    @app.get("/chain-view/data", include_in_schema=False)
    def chain_view_data(request: Request, token: str | None = None) -> dict:
        """看板自动刷新用的数据接口，口径与页面一致。"""
        _chain_view_guard(request, token)
        return _chain_payload()

    @app.get("/api/v1/anchor/granularity", dependencies=[Depends(require_service_header)])
    def anchor_granularity_endpoint() -> dict:
        """客户可选的上链粒度：从"整包一个区块"到"每个栅格块一个区块"。

        真实可选的只有下面这三档；`TILE` 在导出时才展开成具体块数，
        所以这里给的是**口径**，不是保证的数量。
        """
        return {
            "granularities": [
                {"value": "UPLOAD", "label": "整个上传（最大）", "unit_count": 1,
                 "detail": "一次上传的全部内容算作一个区块，只锚定整体输出哈希"},
                {"value": "UNIT", "label": "每个文件（默认）", "unit_count": "= 成果内文件数",
                 "detail": "成果目录里的每个文件各自上链（逐文件 DATA_UNIT_ANCHORED）"},
                {"value": "TILE", "label": "每个栅格块（最小）", "unit_count": "= 成果内 GeoTIFF 的内部块数",
                 "detail": "按 GeoTIFF 内部 block_windows 切分，每块单独算摘要上链；"
                           "读不出块的类型自动退回按文件上链，绝不假装切过"},
            ],
            "default": "UPLOAD",
            "chosen_at": "task submit (anchor_granularity) or export (override)",
            "event_types": ["DATA_UNIT_ANCHORED", "RESULT_EXPORTED"],
            "note": "上链对象选择只影响链上记录的数量与粒度，不改变水印嵌入结果",
        }

    @app.post("/api/v1/precision/check", dependencies=[Depends(require_service_header)])
    def precision_check_endpoint(request: PrecisionCheckRequest) -> dict:
        """干跑校验：这个降精度强度会不会把水印载体压没。返回 ok + 可用的安全范围。"""
        from . import precision_bounds

        blocked_reason = _precision_block_reason(request)

        resolved_type = request.data_type
        counts: dict[str, Any] = {}
        if request.upload_id:
            upload = task_store.get_upload(request.upload_id)
            if upload is None:
                raise HTTPException(status_code=404, detail=f"unknown upload_id: {request.upload_id}")
            resolved_type = resolved_type or upload.get("data_type")
            counts = {k: v for k, v in _carrier_stats(upload).items() if v is not None}

        w = request.width if request.width is not None else counts.get("width")
        h = request.height if request.height is not None else counts.get("height")
        f = request.features if request.features is not None else counts.get("features")

        mode = (request.precision_mode or "NONE").upper()
        if mode == "RESAMPLE":
            verdict = precision_bounds.check_resample(
                w, h, cross_tile=(resolved_type or "").upper() == "DEM",
                scale=request.resample_scale,
                target_width=request.resample_width, target_height=request.resample_height)
        elif mode == "DECIMALS":
            verdict = precision_bounds.check_decimals(f, decimals=request.precision_decimals)
        elif mode == "DOUGLAS_PEUCKER":
            bounds = precision_bounds.vector_bounds(f)
            verdict = {"ok": bool(bounds.get("feasible", True)), "bounds": bounds,
                       "reason": bounds.get("reason")}
        else:
            bounds = precision_bounds.bounds_for(resolved_type or "", width=w, height=h, features=f)
            verdict = {"ok": True, "bounds": bounds, "reason": None}

        final: dict[str, Any] = {
            "data_type": resolved_type,
            "precision_mode": mode,
            "carrier_counts": {"width": w, "height": h, "features": f},
            "blocked": blocked_reason is not None,
        }
        if blocked_reason:
            # 策略拒绝优先：此时参数可能完全合法，仍要给出载体边界供前端展示
            final["watermark_safe"] = False
            final["message"] = blocked_reason
            final["check"] = {
                "ok": False,
                "reason": blocked_reason,
                "bounds": verdict.get("bounds") or {"known": False},
                "bounds_ok": bool(verdict.get("ok")),
                "bounds_reason": verdict.get("reason"),
            }
        else:
            final["check"] = verdict
            final["watermark_safe"] = bool(verdict.get("ok"))
            if not verdict.get("ok"):
                # 越界自有一句（哪一项越界），不复用策略文案
                final["message"] = verdict.get("reason")
        return final

    @app.post("/internal/import", status_code=201, dependencies=[Depends(require_service_header)])
    async def import_file(project_id: str = Form(...), classification: str = Form(...), file: UploadFile = File(...)) -> dict:
        try:
            category = parse_classification(classification)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        filename = Path(file.filename or "upload.bin").name
        if filename in {"", ".", ".."}:
            raise HTTPException(status_code=422, detail="invalid upload filename")
        batch = Path(workspace) / "incoming" / project_id / secrets.token_hex(16)
        batch.mkdir(parents=True, exist_ok=False)
        destination = batch / filename
        digest = __import__("hashlib").sha256()
        try:
            with destination.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    stream.write(chunk); digest.update(chunk)
        finally:
            await file.close()
        file_hash = digest.hexdigest()
        record = app.state.processing_service.trace_chain.append("DATA_IMPORTED", project_id=project_id,
            classification=category.value, source_label="internal_upload", input_hash=file_hash,
            output_hash=file_hash, sandbox_user="gds_sbx", destination=str(destination))
        audit_record("DATA_IMPORTED", actor="internal_upload", project_id=project_id,
                     classification=category.value, source_path=str(destination), input_hash=file_hash)
        return {"source_path": str(destination), "classification": category.value,
                "sha256": file_hash, "trace_record_hash": record["record_hash"]}

    @app.get("/internal/capabilities")
    def capabilities(service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> dict:
        _require_service_token(service_token)
        from . import precision_bounds
        return {
            "classifications": [{"classification": item.value, **CAPABILITIES[item]} for item in Classification],
            # 跨类型能力（与数据类型无关），让前端不用猜
            "precision_bounds": {
                "endpoints": ["GET /api/v1/precision/bounds", "POST /api/v1/precision/check"],
                "enforced_at_submit": True,
                "carrier_floors": {
                    "raster_min_side_px": precision_bounds.RASTER_MIN_SIDE,
                    "dem_cross_tile_min_window_px": precision_bounds.DEM_CROSS_TILE_MIN_WIDTH,
                    "vector_min_features": precision_bounds.VECTOR_MIN_FEATURES,
                    "vector_min_decimals": precision_bounds.VECTOR_MIN_DECIMALS,
                    "point_cloud_header_slots": precision_bounds.POINT_CLOUD_HEADER_SLOTS,
                    "point_cloud_slots_formula": "48 + 8 × 水印载荷字节数（每比特 1 个坐标槽）",
                },
                "unknown_size_behaviour": "known=false + reason, never invent min/max",
            },
            "anchor_granularity": {
                "endpoint": "GET /api/v1/anchor/granularity",
                "values": ["UPLOAD", "UNIT", "TILE"],
                "default": "UPLOAD",
                "tile_unit_source": "rasterio block_windows (real GeoTIFF blocks)",
                "fallback_is_reported": True,
            },
        }

    @app.get("/internal/trace/verify")
    def trace_verify(service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> dict:
        _require_service_token(service_token)
        return app.state.processing_service.trace_chain.verify()

    @app.get("/internal/trace/artifact/{artifact_id}")
    def trace_artifact(artifact_id: str, service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> dict:
        _require_service_token(service_token)
        record = app.state.task_store.get_artifact_record(artifact_id)
        if not record:
            raise HTTPException(status_code=404, detail="result not found")
        output = Path(record["artifact"].output)
        manifest_path = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="trace manifest not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"artifact_id": artifact_id, "project_id": record["project_id"],
                "classification": manifest.get("classification"), "source_hash": manifest.get("source_hash"),
                "output_hash": manifest.get("output_hash"), "trace_record_hash": manifest.get("trace_record_hash"),
                # Additive only: on-chain receipt for this record when anchoring is enabled, so the
                # response shape for callers that do not read it is unchanged.
                "anchor": app.state.processing_service.trace_chain.anchor_for(manifest.get("trace_record_hash")),
                "chain": app.state.processing_service.trace_chain.verify()}

    # ── 审计日志存储配置（前端「日志存储」页） ───────────────────────────────
    # 说明：路径用 /api/v1/audit/retention，Server1 需要加同名转发；字段统一驼峰，
    # 与 Server1 现有 /api/audit/retention 的返回保持一致，前端不用写两套解析。
    @app.get("/api/v1/audit/retention", dependencies=[Depends(require_service_header)])
    def get_retention() -> dict:
        return app.state.storage_policy.view()

    @app.put("/api/v1/audit/retention", dependencies=[Depends(require_service_header)])
    def put_retention(request: RetentionUpdateRequest) -> dict:
        patch = request.model_dump(exclude_none=True)
        operator = patch.pop("operator", "") or ""
        try:
            app.state.storage_policy.update(patch, actor=operator)
        except PolicyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if app.state.audit is not None:
            app.state.audit.record("RETENTION_POLICY_UPDATED", actor=operator or "系统",
                                   changes=patch)
        return app.state.storage_policy.view()

    @app.post("/api/v1/audit/retention/cleanup", dependencies=[Depends(require_service_header)])
    def post_retention_cleanup(request: RetentionCleanupRequest) -> dict:
        report = app.state.storage_policy.cleanup(
            dry_run=bool(request.dryRun), actor=request.operator or "",
            targets=request.targets, confirm_capacity=bool(request.confirmCapacity))
        if not request.dryRun and app.state.audit is not None:
            app.state.audit.record("RETENTION_CLEANUP_EXECUTED", actor=request.operator or "系统",
                                   deleted_audit_files=report["auditFiles"]["deleted"],
                                   deleted_es_indices=report["esIndices"]["deleted"],
                                   cutoff=report["cutoff"])
        return report

    @app.get("/api/v1/audit/retention/history", dependencies=[Depends(require_service_header)])
    def get_retention_history(limit: int = 20) -> dict:
        limit = max(1, min(int(limit), 200))
        return {"items": app.state.storage_policy.history(limit), "limit": limit}

    @app.get("/internal/audit/retention")
    def audit_retention(service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> dict:
        _require_service_token(service_token)
        if audit is None:
            return {"policy_configured": False, "policy_days": 180, "reason": "audit log not mounted"}
        return audit.retention_evidence()

    def _admin_storage_summary() -> dict:
        root = Path(workspace)
        totals = {"files": 0, "bytes": 0}
        by_area = {}
        for name in ("incoming", "projects", "exports", "audit", "state"):
            path = root / name
            files = 0; size = 0
            if path.exists():
                for child in path.rglob("*"):
                    if child.is_file():
                        files += 1
                        try: size += child.stat().st_size
                        except OSError: pass
            by_area[name] = {"path": str(path), "files": files, "bytes": size}
            totals["files"] += files; totals["bytes"] += size
        return {"workspace": str(root), "totals": totals, "areas": by_area}

    def _admin_besu_status() -> dict:
        install_root = Path(os.environ.get("GDS_INSTALL_ROOT", r"D:\GeoDataSecurity"))
        cfg_path = install_root / "state" / "besu_trace_config.json"
        receipt_path = install_root / "chain" / "anchor" / "anchor_receipts.json"
        if not receipt_path.is_file():
            legacy = install_root / "besu" / "anchor" / "anchor_receipts.json"
            if legacy.is_file():
                receipt_path = legacy
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        receipts = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
        rpc = cfg.get("rpc_url", "")
        rpc_result = {}
        if rpc:
            try:
                body = json.dumps({"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}).encode()
                req = urllib.request.Request(rpc, data=body, headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    rpc_result = json.loads(response.read().decode("utf-8"))
            except Exception as exc:
                rpc_result = {"error": f"{type(exc).__name__}: {exc}"}
        return {"config": cfg, "rpc": rpc_result, "receipts": receipts,
                "relationship": {"local_trace_event": "record_hash", "besu_anchor": "record_hash -> transaction -> block -> event", "raw_data_onchain": False}}

    @app.post("/admin/auth/login")
    def admin_login(request: AdminLoginRequest) -> dict:
        identity = admin_auth.authenticate_password(request.username, request.password)
        if not identity:
            raise HTTPException(status_code=401, detail="invalid username or password")
        token = admin_auth.issue_token(identity["username"])
        audit_record("ADMIN_LOGIN", actor=identity["username"], role=identity["role"])
        return {"access_token": token, "token_type": "admin", "username": identity["username"], "role": identity["role"]}

    @app.post("/admin/auth/change-password", dependencies=[Depends(require_admin_header)])
    def admin_change_password(request: AdminPasswordChangeRequest,
                              admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        identity = admin_identity(admin_token)
        try:
            admin_auth.change_password(identity["username"], request.old_password, request.new_password)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        audit_record("ADMIN_PASSWORD_CHANGED", actor=identity["username"])
        return {"changed": True, "username": identity["username"], "reauthenticate": True}

    @app.get("/admin/users", dependencies=[Depends(require_admin_header)])
    def admin_users() -> dict:
        return {"users": admin_auth.list_users()}

    @app.post("/admin/users", status_code=201, dependencies=[Depends(require_superadmin)])
    def admin_create_user(request: AdminUserCreateRequest,
                          admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        identity = admin_identity(admin_token)
        try:
            created = admin_auth.create_user(request.username, request.role, request.password)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="username already exists") from exc
        audit_record("ADMIN_USER_CREATED", actor=identity["username"], username=request.username, role=request.role)
        return created

    @app.post("/admin/users/{username}/token", dependencies=[Depends(require_superadmin)])
    def admin_issue_user_token(username: str,
                               admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        identity = admin_identity(admin_token)
        try:
            token = admin_auth.issue_token(username)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        audit_record("ADMIN_TOKEN_ISSUED", actor=identity["username"], username=username)
        return {"username": username, "access_token": token, "token_type": "admin"}

    @app.post("/admin/users/{username}/disable", dependencies=[Depends(require_superadmin)])
    def admin_disable_user(username: str,
                           admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        identity = admin_identity(admin_token)
        admin_auth.set_active(username, False)
        audit_record("ADMIN_USER_DISABLED", actor=identity["username"], username=username)
        return {"username": username, "active": False}

    @app.post("/admin/users/{username}/enable", dependencies=[Depends(require_superadmin)])
    def admin_enable_user(username: str,
                          admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        identity = admin_identity(admin_token)
        admin_auth.set_active(username, True)
        audit_record("ADMIN_USER_ENABLED", actor=identity["username"], username=username)
        return {"username": username, "active": True}

    @app.post("/admin/users/{username}/revoke-tokens", dependencies=[Depends(require_superadmin)])
    def admin_revoke_tokens(username: str,
                            admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        identity = admin_identity(admin_token)
        admin_auth.revoke_tokens(username)
        audit_record("ADMIN_TOKENS_REVOKED", actor=identity["username"], username=username)
        return {"username": username, "revoked": True}

    @app.get("/admin/overview", dependencies=[Depends(require_admin_header)])
    def admin_overview() -> dict:
        counts = task_store.counts()
        chain = service.trace_chain.verify()
        retention = audit.retention_evidence() if audit is not None else {"policy_configured": False}
        return {"service": "server2", "api_role": "processing_storage_readonly_output",
                "counts": counts, "storage": _admin_storage_summary(),
                "trace_chain": chain, "audit": retention, "besu": _admin_besu_status()}

    @app.get("/admin/storage", dependencies=[Depends(require_admin_header)])
    def admin_storage() -> dict:
        return _admin_storage_summary()

    @app.get("/admin/database/{table}", dependencies=[Depends(require_admin_header)])
    def admin_database(table: str, limit: int = 100, offset: int = 0) -> dict:
        try:
            rows = task_store.list_rows(table, limit, offset)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"table": table, "limit": limit, "offset": offset, "count": len(rows), "rows": rows}

    @app.get("/admin/trace/chain", dependencies=[Depends(require_admin_header)])
    def admin_trace_chain() -> dict:
        return {"chain": service.trace_chain.verify(), "records": service.trace_chain.search()}

    @app.get("/admin/audit", dependencies=[Depends(require_admin_header)])
    def admin_audit() -> dict:
        return audit.retention_evidence() if audit is not None else {"policy_configured": False}

    @app.get("/admin/besu", dependencies=[Depends(require_admin_header)])
    def admin_besu() -> dict:
        return _admin_besu_status()

    @app.get("/admin/sandbox/volumes", dependencies=[Depends(require_admin_header)])
    def admin_volumes() -> dict:
        """卷池全景：每个卷的挂载状态、容量、剩余、优先级、是否主卷。全部当场实测。"""
        return sandbox_pool.status()

    @app.post("/admin/sandbox/volumes", status_code=202, dependencies=[Depends(require_admin_header)])
    def admin_volume_create(payload: AdminVolumeCreateRequest,
                            admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """新建一个沙箱卷。**异步**：立即返回 202 + job_id，用 GET /admin/sandbox/volumes/jobs/{job_id} 查进度。

        为什么必须异步：`VeraCrypt Format.exe` 是无命令行静默模式的 GUI 程序，
        建 100GB 卷实测要几分钟，同步 HTTP 请求必然超时（浏览器/网关先断开，
        服务端还在写，结果页面看不到、状态也丢了）。
        """
        identity = admin_identity(admin_token)
        letter = (payload.letter or "").upper()
        if not letter:
            free = free_drive_letters()
            if not free:
                raise HTTPException(status_code=409, detail="没有空闲盘符可分配给新卷")
            letter = free[0]
        if Path(f"{letter}:\\").exists():
            raise HTTPException(status_code=409, detail=f"{letter}: 已被占用，请换一个盘符")
        if len(letter) != 1 or letter not in "DEFGHIJKLMNOPQRSTUVWXYZ":
            raise HTTPException(status_code=422, detail="盘符必须在 D-Z 之间")

        password = payload.password or sandbox_pool.volume_password()
        if not password:
            raise HTTPException(
                status_code=422,
                detail="缺少卷口令：请在请求里给 password（≥20 字符），或在 server2.env 配置 GDS_SANDBOX_VOLUME_PASSWORD")

        job_id = sandbox_pool.start_create_job(
            size_gb=payload.size_gb, letter=letter,
            label=payload.label or f"{payload.size_gb}GB 卷",
            priority=payload.priority, password=password,
            actor=identity["username"], now=now_local())
        audit_record("SANDBOX_VOLUME_CREATE_STARTED", actor=identity["username"],
                     job_id=job_id, letter=letter, size_gb=payload.size_gb)
        return {"accepted": True, "job_id": job_id, "letter": letter,
                "size_gb": payload.size_gb,
                "poll": f"/admin/sandbox/volumes/jobs/{job_id}"}

    @app.post("/admin/sandbox/volumes/remount", dependencies=[Depends(require_admin_header)])
    def admin_volumes_remount(admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """把池里所有未挂载的数据卷挂回来。重启后自动执行，这里提供手动入口。"""
        identity = admin_identity(admin_token)
        results = sandbox_pool.ensure_all_mounted()
        audit_record("SANDBOX_VOLUMES_REMOUNTED", actor=identity["username"], results=results)
        return {"results": results, "pool": sandbox_pool.status()}

    @app.get("/admin/sandbox/volumes/jobs", dependencies=[Depends(require_admin_header)])
    def admin_volume_jobs(limit: int = 20) -> dict:
        """最近的建卷任务（含进行中的），供页面轮询显示进度。"""
        return {"jobs": sandbox_pool.jobs(limit)}

    @app.get("/admin/sandbox/volumes/jobs/{job_id}", dependencies=[Depends(require_admin_header)])
    def admin_volume_job(job_id: str) -> dict:
        job = sandbox_pool.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        if job.get("volume_id"):
            vol = sandbox_pool.get(job["volume_id"])
            if vol is not None:
                job["volume"] = sandbox_pool.volume_state(vol)
        return job

    @app.post("/admin/sandbox/volumes/{volume_id}/unmount", dependencies=[Depends(require_admin_header)])
    def admin_volume_unmount(volume_id: str, payload: AdminVolumeActionRequest,
                             admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """卸载（下线）一个卷。数据保留，卷仍在池中但标记为不可用，上传不再落到它上面。

        默认只卸挂载、不禁用；要彻底不让它接新数据用 `disable`。
        """
        identity = admin_identity(admin_token)
        vol = sandbox_pool.get(volume_id)
        if vol is None:
            raise HTTPException(status_code=404, detail="卷不存在")
        if vol.primary:
            raise HTTPException(status_code=409, detail="主卷不能卸载：state/ 与哈希链在上面")
        if not vol.letter:
            raise HTTPException(status_code=422, detail="该卷没有盘符，不是本平台管理的容器")
        ok = unmount_letter(vol.letter)
        if not ok:
            raise HTTPException(status_code=409,
                                detail=f"{vol.letter}: 卸载失败：可能仍有文件被占用，请先停止使用该卷上的任务")
        audit_record("SANDBOX_VOLUME_UNMOUNTED", actor=identity["username"],
                     volume_id=volume_id, letter=vol.letter, reason=payload.reason)
        return {"unmounted": True, "volume": sandbox_pool.volume_state(vol)}

    @app.post("/admin/sandbox/volumes/{volume_id}/mount", dependencies=[Depends(require_admin_header)])
    def admin_volume_mount(volume_id: str, payload: AdminVolumeActionRequest,
                           admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """重新挂载（上线）一个卷，让它重新接新上传。"""
        identity = admin_identity(admin_token)
        vol = sandbox_pool.get(volume_id)
        if vol is None:
            raise HTTPException(status_code=404, detail="卷不存在")
        if not vol.container:
            raise HTTPException(status_code=422, detail="该卷不是容器卷，无法由平台挂载")
        password = sandbox_pool.volume_password()
        if not password:
            raise HTTPException(status_code=422,
                                detail="缺少卷口令：请在 server2.env 配置 GDS_SANDBOX_VOLUME_PASSWORD")
        if not mount_container(vol.container, password, vol.letter):
            raise HTTPException(status_code=500, detail=f"{vol.letter}: 挂载失败")
        sandbox_pool._ensure_skeleton(vol)
        audit_record("SANDBOX_VOLUME_MOUNTED", actor=identity["username"],
                     volume_id=volume_id, letter=vol.letter, reason=payload.reason)
        return {"mounted": True, "volume": sandbox_pool.volume_state(vol)}

    @app.post("/admin/sandbox/volumes/{volume_id}/enable", dependencies=[Depends(require_admin_header)])
    def admin_volume_enable(volume_id: str, payload: AdminVolumeActionRequest,
                            admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        identity = admin_identity(admin_token)
        try:
            vol = sandbox_pool.set_enabled(volume_id, True)
        except KeyError:
            raise HTTPException(status_code=404, detail="卷不存在")
        audit_record("SANDBOX_VOLUME_ENABLED", actor=identity["username"], volume_id=volume_id)
        return {"enabled": True, "volume": sandbox_pool.volume_state(vol)}

    @app.post("/admin/sandbox/volumes/{volume_id}/disable", dependencies=[Depends(require_admin_header)])
    def admin_volume_disable(volume_id: str, payload: AdminVolumeActionRequest,
                             admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """停用：不再把新上传分到它上面。已在其上的数据仍可读。"""
        identity = admin_identity(admin_token)
        vol = sandbox_pool.get(volume_id)
        if vol is None:
            raise HTTPException(status_code=404, detail="卷不存在")
        if vol.primary:
            raise HTTPException(status_code=409, detail="主卷不能停用：state/ 与哈希链在上面")
        vol = sandbox_pool.set_enabled(volume_id, False)
        audit_record("SANDBOX_VOLUME_DISABLED", actor=identity["username"], volume_id=volume_id,
                     reason=payload.reason)
        return {"enabled": False, "volume": sandbox_pool.volume_state(vol)}

    @app.delete("/admin/sandbox/volumes/{volume_id}", dependencies=[Depends(require_admin_header)])
    def admin_volume_remove(volume_id: str,
                            admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """从池中移除。只摘登记，**不删容器文件、不删数据**。"""
        identity = admin_identity(admin_token)
        vol = sandbox_pool.get(volume_id)
        if vol is None:
            raise HTTPException(status_code=404, detail="卷不存在")
        if vol.primary:
            raise HTTPException(status_code=409, detail="主卷不能移除")
        # 移出池但盘还挂着 → 留下一个"挂着却没人管"的盘符，以后建卷时
        # free_drive_letters() 会一直跳过它。所以先在池内卸掉挂载。
        unmounted = False
        if vol.container and vol.letter:
            try:
                unmounted = unmount_letter(vol.letter)
            except Exception:
                unmounted = False
        try:
            sandbox_pool.remove(volume_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        audit_record("SANDBOX_VOLUME_REMOVED", actor=identity["username"], volume_id=volume_id,
                     container=vol.container, unmounted=unmounted)
        return {"removed": True, "volume_id": volume_id, "container_kept": vol.container,
                "unmounted": unmounted,
                "note": "容器文件与数据均保留；重新登记即可恢复使用"}

    @app.get("/sandbox-view", include_in_schema=False)
    def sandbox_view(request: Request, token: str | None = None) -> HTMLResponse:
        """沙箱容量管理页：打开即可看容量，需管理员口令才能改配额/清理。"""
        _chain_view_guard(request, token)
        status = sandbox_policy.status()
        return HTMLResponse(render_sandbox_console(
            status, sandbox_policy.area_usage(), now_local(), sandbox_pool.status()))

    @app.get("/provenance-view", include_in_schema=False)
    def provenance_view(request: Request, token: str | None = None) -> HTMLResponse:
        """溯源查询控制台：打开即可检测四类数据是否为本系统流出。

        页面本身不含凭据；提交检测走同源 ``/provenance-view/check``，
        该端点和本页共用同一道看板守卫（本机/内网直开，其他来源需 ?token=）。
        """
        _chain_view_guard(request, token)
        from .provenance_console import render_html as render_provenance_console
        return HTMLResponse(render_provenance_console(generated_at=now_local()))

    @app.get("/provenance-view/check", include_in_schema=False)
    def provenance_view_check(request: Request, source_path: str, expected_watermark: str,
                              reference_path: str | None = None) -> dict:
        """控制台专用的同源查询端点：鉴权方式与页面一致，避免在页面里注入服务令牌。"""
        _chain_view_guard(request, None)
        from . import provenance_any as pa
        # 先校验水印值，再查文件：参数错误应返回 400，不该被文件检查掩盖成 404。
        try:
            expected = _parse_watermark_int(expected_watermark)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400,
                                detail=f"invalid expected_watermark: {exc}") from exc
        src = Path(source_path)
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"source not found: {source_path}")
        root = Path(loan_database_path).parent.parent if loan_database_path else Path(workspace)
        work = root / "provenance_any" / __import__("uuid").uuid4().hex
        _prune_scratch_dirs(root / "provenance_any")
        work.mkdir(parents=True, exist_ok=True)
        result = pa.check_any(src, expected,
                              Path(reference_path) if reference_path else None, work)
        return {
            "is_probably_ours": bool(result.get("found")),
            "source_path": source_path,
            "data_class": result.get("data_class"),
            "channel": result.get("channel"),
            "expected_watermark": expected,
            "resolved_watermark": result.get("decoded"),
            "capability": result.get("capability"),
            "detail": result.get("detail"),
            "via": result.get("via"),
            "ledger": result.get("ledger", "hash_chain"),
            "work_dir": str(work),
            "conclusion": result.get("conclusion"),
        }

    @app.get("/admin/sandbox", dependencies=[Depends(require_admin_header)])
    def admin_sandbox() -> dict:
        """沙箱容量与占用：策略值与实测值分开返回，不互相推导。"""
        return sandbox_policy.status()

    @app.get("/admin/sandbox/usage", dependencies=[Depends(require_admin_header)])
    def admin_sandbox_usage() -> dict:
        return {"volume": sandbox_policy.volume(), "areas": sandbox_policy.area_usage()}

    @app.post("/admin/sandbox/quota", dependencies=[Depends(require_admin_header)])
    def admin_sandbox_quota(payload: AdminSandboxQuotaRequest,
                            admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """设置沙箱容量上限。只改策略，不删数据、不碰 V: 卷。"""
        identity = admin_identity(admin_token)
        try:
            result = sandbox_policy.update_policy(
                quota_bytes=payload.quota_bytes,
                actor=identity["username"], reason=payload.reason or "", now=now_local())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        audit_record("SANDBOX_QUOTA_UPDATED", actor=identity["username"],
                     quota_bytes=payload.quota_bytes, reason=payload.reason)
        return {"updated": True, "policy": {k: v for k, v in result.items() if k != "history"},
                "status": sandbox_policy.status()}

    @app.post("/admin/sandbox/cleanup", dependencies=[Depends(require_admin_header)])
    def admin_sandbox_cleanup(payload: AdminSandboxCleanupRequest,
                              admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """清空沙箱数据区。默认 dry_run=true，只统计不删除。

        安全约束（与 ``references/workspace-wipe-and-reset.md`` 一致）：
        * 有 RUNNING/PROCESSING 任务时拒绝执行；
        * 永远不动 ``state/``（链与管理员凭据）、``audit/``、``sandbox/worker/``；
        * 只删数据区内容，保留目录本身，从而保留其 ACL。
        """
        identity = admin_identity(admin_token)
        running = task_store.count_by_status("tasks", ("RUNNING", "PROCESSING", "QUEUED"))
        if running and not payload.force:
            raise HTTPException(
                status_code=409,
                detail=f"有 {running} 个任务正在运行；清理会破坏这些任务，请等它们结束或显式强制清理")
        areas = payload.areas or list(SANDBOX_CLEANABLE_AREAS)
        unknown = [a for a in areas if a not in SANDBOX_CLEANABLE_AREAS]
        if unknown:
            raise HTTPException(status_code=422, detail=f"不支持清理的区域: {', '.join(unknown)}")
        root = Path(workspace)
        before = sandbox_policy.volume()
        removed: dict[str, dict] = {}
        for area in areas:
            base = root / area
            files = 0
            size = 0
            if base.exists():
                for dirpath, _dirs, filenames in os.walk(base):
                    for fn in filenames:
                        files += 1
                        try:
                            size += os.path.getsize(os.path.join(dirpath, fn))
                        except OSError:
                            pass
            removed[area] = {"files": files, "bytes": size}
            if not payload.dry_run and base.exists():
                for child in base.iterdir():
                    try:
                        if child.is_dir() and not child.is_symlink():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink()
                    except OSError:
                        pass
        if not payload.dry_run:
            # 队列行一并清理，否则列表里会剩下一堆指向空目录的记录。
            con = sqlite3.connect(str(task_store.database_path), timeout=30)
            try:
                for table in ("tasks", "upload_sessions", "artifacts", "export_jobs", "upload_chunks"):
                    try:
                        con.execute(f"DELETE FROM {table}")
                    except sqlite3.Error:
                        pass
                con.commit()
            finally:
                con.close()
            audit_record("SANDBOX_CLEANED", actor=identity["username"], areas=areas,
                         removed=removed, dry_run=False)
        after = sandbox_policy.volume()
        return {"dry_run": payload.dry_run, "areas": removed,
                "volume_before": before, "volume_after": after,
                "freed_bytes": max(0, (after.get("free_bytes") or 0) - (before.get("free_bytes") or 0))
                if not payload.dry_run else sum(v["bytes"] for v in removed.values())}

    @app.get("/admin/relationship", dependencies=[Depends(require_admin_header)])
    def admin_relationship() -> dict:
        return {"nodes": [
                    {"id":"server1","type":"client","label":"Server1 Frontend/Client"},
                    {"id":"server2_api","type":"api","label":"Server2 API :9081"},
                    {"id":"incoming","type":"storage","label":"V: incoming"},
                    {"id":"sandbox","type":"compute","label":"gds_sbx Sandbox"},
                    {"id":"trace_chain","type":"ledger","label":"Local Trace Chain"},
                    {"id":"besu","type":"blockchain","label":"Besu QBFT"},
                    {"id":"exports","type":"storage","label":"Approved Exports"}],
                "edges": [
                    {"from":"server1","to":"server2_api","label":"9081 API"},
                    {"from":"server2_api","to":"incoming","label":"controlled import"},
                    {"from":"incoming","to":"sandbox","label":"gds_sbx processing"},
                    {"from":"sandbox","to":"trace_chain","label":"hash + verification"},
                    {"from":"trace_chain","to":"besu","label":"record hash anchor"},
                    {"from":"server2_api","to":"exports","label":"approved export"},
                    {"from":"exports","to":"server1","label":"download"}]}

    # Authentication dependency is defined above so upload and task routes share it.

    @app.post("/internal/tasks", status_code=202, dependencies=[Depends(require_service_header)])
    def submit_task(request: TaskRequest) -> dict[str, str]:
        normalized = request.data_type.upper()
        if not request.classification.strip():
            legacy_defaults = {"SHP": "DLG_LINEWORK", "GEOJSON": "THEMATIC_VECTOR", "GEOTIFF": "REMOTE_SENSING_RASTER", "GTIFF": "REMOTE_SENSING_RASTER", "TEXTURES": "OBLIQUE_MESH", "OSGB_TEXTURES": "OBLIQUE_MESH", "OSGB": "OBLIQUE_MESH"}
            request = request.model_copy(update={"classification": legacy_defaults.get(normalized, "")})
        try:
            classification = parse_classification(request.classification)
        except ValueError as exc:
            audit_record("TASK_REJECTED", actor=request.user_id, project_id=request.project_id,
                         data_type=request.data_type, reason=str(exc))
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if normalized not in {"GEOJSON", "SHP", "DLG", "GEOTIFF", "GTIFF", "DEM", "POINT_CLOUD", "TEXTURES", "OSGB_TEXTURES", "OSGB"}:
            audit_record("TASK_REJECTED", actor=request.user_id, project_id=request.project_id,
                         data_type=request.data_type, reason="unsupported data type")
            raise HTTPException(status_code=422, detail=f"task data type not implemented: {request.data_type}")
        if not allowed_for_data_type(classification, "SHP" if normalized == "DLG" else normalized):
            raise HTTPException(status_code=422, detail=f"classification {classification.value} is incompatible with {normalized}")
        candidate = Path(request.source_path).resolve()
        archive_pending = candidate.is_file() and candidate.suffix.lower() == ".zip"
        if archive_pending:
            # Front-end contract: every data class may arrive as one ZIP archive whose
            # real content decides the handler. Extraction and payload validation run in
            # the worker thread so the HTTP request is never blocked by a large ZIP.
            _validate_archive(candidate)
            source = candidate
        else:
            source = check_input_path(request.source_path, expect_dir=normalized in {"SHP", "DLG", "DEM", "TEXTURES", "OSGB_TEXTURES", "OSGB", "POINT_CLOUD"})
        if not archive_pending:
            if normalized in {"SHP", "DLG"} and not any(source.rglob("*.shp")) and not any(source.rglob("*.gpkg")):
                raise HTTPException(status_code=422, detail="no .shp layers or .gpkg dataset found")
            if normalized in {"TEXTURES", "OSGB_TEXTURES"} and not any(
                path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in source.iterdir()
            ):
                raise HTTPException(status_code=422, detail="no JPG/PNG textures found")
            if normalized == "OSGB" and not any(source.rglob("*.osgb")):
                raise HTTPException(status_code=422, detail="no .osgb files found")
            if normalized == "DEM" and source.is_file() and source.suffix.lower() not in {".tif", ".tiff"}:
                raise HTTPException(status_code=422, detail="DEM requires a GeoTIFF source or a directory containing GeoTIFF files")
            if normalized == "DEM" and source.is_dir() and not any(path.is_file() and path.suffix.lower() in {".tif", ".tiff"} for path in source.rglob("*")):
                raise HTTPException(status_code=422, detail="DEM payload contains no .tif/.tiff files")
            if normalized == "POINT_CLOUD":
                from .point_cloud import discover
                if not discover(source):
                    raise HTTPException(status_code=422, detail="no supported point-cloud files found")
        _validate_precision_request(request, normalized)
        task_id = secrets.token_hex(16)
        task = {"task_id": task_id, "status": "QUEUED", "project_id": request.project_id,
                "user_id": request.user_id, "result_id": "", "classification": classification.value,
                "watermark_enabled": request.watermark_enabled, "anchor_granularity": request.anchor_granularity,
                "upload_id": getattr(request, "upload_id", "") or "",
                "data_type": normalized,
                "resume_payload": _task_resume_payload(request)}
        app.state.task_store.save_task(task)
        audit_record("TASK_SUBMITTED", actor=request.user_id, project_id=request.project_id,
                     data_type=request.data_type, task_id=task_id, source_path=str(source))
        app.state.task_executor.submit(
            _run_task, app, request, source, task_id, audit_record
        )
        return {"task_id": task_id, "status": "QUEUED", "result_id": ""}

    @app.get("/internal/tasks/{task_id}")
    def task_status(task_id: str, service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> dict:
        _require_service_token(service_token)
        task = app.state.task_store.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="task not found")
        return _task_view(task)
    @app.post("/internal/results/{result_id}/approve")
    def approve(result_id: str, request: ApproveRequest, service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> dict:
        _require_service_token(service_token)
        artifact_record = app.state.task_store.get_artifact_record(result_id)
        if not artifact_record:
            raise HTTPException(status_code=404, detail="result not found")
        artifact = artifact_record["artifact"]
        if artifact.status == APPROVAL_REJECTED:
            raise HTTPException(status_code=409,
                                detail="result was rejected by manual review; delivering it needs a new task")
        manifest_path = Path(artifact.output) / "manifest.json" if Path(artifact.output).is_dir() else Path(artifact.output).parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        require_verified_watermark(manifest)
        is_partial = manifest.get("verification_status") == "VERIFIED_PARTIAL" or manifest.get("source_integrity") == "INCOMPLETE"
        if is_partial and not request.manual_review_confirmed:
            raise HTTPException(status_code=409, detail="partial result requires manual_review_confirmed=true and review_conclusion")
        if is_partial and not request.review_conclusion:
            raise HTTPException(status_code=422, detail="review_conclusion is required for partial results")
        approval_status = "PARTIALLY_APPROVED" if is_partial else "APPROVED"
        artifact_record["artifact"] = artifact.__class__(artifact.artifact_id, artifact.data_type, artifact.source, artifact.output, artifact.files_processed, approval_status)
        app.state.task_store.set_artifact_status(result_id, approval_status)
        if manifest_path.is_file():
            manifest.update({"status": approval_status, "approved_by": request.reviewer_id})
            if is_partial:
                manifest["manual_review"] = {"confirmed": True, "reviewer_id": request.reviewer_id, "conclusion": request.review_conclusion}
                manifest["data_quality_conclusion"] = request.review_conclusion
                manifest["approval_policy"] = "MANUAL_REVIEW_COMPLETED_PARTIAL_RESULT"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        audit_record("RESULT_PARTIALLY_APPROVED" if is_partial else "RESULT_APPROVED", actor=request.reviewer_id, result_id=result_id, project_id=artifact_record["project_id"], conclusion=request.review_conclusion if is_partial else None)
        return {"result_id": result_id, "approval_status": approval_status, "data_quality_conclusion": request.review_conclusion if is_partial else None}

    @app.get("/readonly/results/{result_id}/manifest")
    def manifest(result_id: str, user_id: str, project_id: str) -> dict:
        artifact_record = app.state.task_store.get_artifact_record(result_id)
        if not artifact_record or artifact_record["user_id"] != user_id or artifact_record["project_id"] != project_id:
            raise HTTPException(status_code=404, detail="result not found")
        if artifact_record["artifact"].status not in APPROVED_STATES:
            rejected_result_guard(artifact_record["artifact"].status)
            raise HTTPException(status_code=403, detail="result is not approved")
        audit_record("RESULT_MANIFEST_READ", actor=user_id, result_id=result_id, project_id=project_id)
        response = artifact_record["artifact"].__dict__.copy()
        manifest_path = Path(artifact_record["artifact"].output)
        manifest_path = manifest_path / "manifest.json" if manifest_path.is_dir() else manifest_path.parent / "manifest.json"
        if manifest_path.is_file():
            response["verification"] = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        return response

    @app.get("/internal/results/{result_id}/metrics")
    def internal_metrics(result_id: str, service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> dict:
        """内部同源口径的成果指标；与 /api/v1/results/{id}/metrics 返回同一结构。"""
        _require_service_token(service_token)
        artifact_record = app.state.task_store.get_artifact_record(result_id)
        if not artifact_record:
            raise HTTPException(status_code=404, detail="result not found")
        output = Path(artifact_record["artifact"].output)
        path = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="manifest not found")
        return _metrics_from_manifest(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/internal/results/{result_id}/verification")
    def verification(result_id: str, service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None) -> dict:
        _require_service_token(service_token)
        artifact_record = app.state.task_store.get_artifact_record(result_id)
        if not artifact_record:
            raise HTTPException(status_code=404, detail="result not found")
        output = Path(artifact_record["artifact"].output)
        manifest_path = output / "manifest.json" if output.is_dir() else output.parent / "manifest.json"
        if not manifest_path.is_file():
            raise HTTPException(status_code=404, detail="verification manifest not found")
        audit_record("RESULT_VERIFICATION_READ", actor="service", result_id=result_id, project_id=artifact_record["project_id"])
        return __import__("json").loads(manifest_path.read_text(encoding="utf-8"))

    @app.get("/readonly/results/{result_id}/file")
    def result_file(result_id: str, user_id: str, project_id: str, filename: str) -> FileResponse:
        artifact_record = app.state.task_store.get_artifact_record(result_id)
        if not artifact_record or artifact_record["user_id"] != user_id or artifact_record["project_id"] != project_id:
            raise HTTPException(status_code=404, detail="result not found")
        artifact = artifact_record["artifact"]
        if artifact.status not in APPROVED_STATES:
            rejected_result_guard(artifact.status)
            raise HTTPException(status_code=403, detail="result is not approved")
        output_path = Path(artifact.output)
        result_dir = output_path if output_path.is_dir() else output_path.parent
        candidate = (result_dir / filename).resolve()
        if candidate.parent != result_dir.resolve() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        audit_record("RESULT_FILE_DOWNLOADED", actor=user_id, result_id=result_id, project_id=project_id, filename=filename)
        return FileResponse(candidate, filename=candidate.name)

    # 启动自愈：把上次进程终止时遗留的「非终态」任务标记为 INTERRUPTED。
    # 任务跑在进程内的 ThreadPoolExecutor 里，没有持久化队列；服务被停止或崩溃后，
    # 这些任务的线程已经不存在，但库里的 status 仍是 QUEUED/RUNNING，
    # 前端会永远看到一个卡在 35%（沙箱嵌入水印）的进度条。
    # 这里在应用启动时统一收尾，让前端立刻看到明确状态，而不是无限等待。
    # 注意：中断不等于失败——数据还在，入参已留档，前端可调 resume 原样重放。
    try:
        stale = task_store.fail_stale_tasks(
            reason=("服务重启导致任务中断（任务运行在进程内线程池，未持久化队列）。"
                    "源数据仍然有效，可调用 POST /api/v1/tasks/{task_id}/resume 原样重提。"))
        if stale:
            audit_record("TASKS_MARKED_INTERRUPTED", actor="system", count=len(stale),
                         details=stale[:50],
                         resumable=[item["task_id"] for item in stale if item.get("resumable")][:50])
    except Exception as exc:  # 清理失败不能阻止服务启动
        audit_record("TASKS_STALE_CLEANUP_FAILED", actor="system", reason=str(exc))

    app.include_router(build_circulation_router(
        circulation_store,
        admin_identity=admin_identity,
        require_service=lambda value: _require_service_token(value),
        audit_record=audit_record,
        export_root=Path(circulation_export_root) if circulation_export_root
        else Path(workspace) / "exports" / "users",
        asset_source=_asset_source_resolver(task_store),
        result_status=_result_status_resolver(task_store),
    ))

    return app


def _submit_precision_bounds(request, normalized: str, source: Path) -> None:
    """提交时用**真实载荷**判一次降精度是否越界，越界立刻 422。

    与处理阶段（_apply_precision）用的是同一条规则 `precision_bounds`：
    提交时能读到大小的就当场拒，不让客户拿到一个 202 之后才 FAILED 的任务。
    """
    from . import precision_bounds
    from .precision import PrecisionError

    mode = (getattr(request, "precision_mode", "NONE") or "NONE").upper()
    if mode not in {"RESAMPLE", "DOUGLAS_PEUCKER"}:
        return
    try:
        # ZIP 载荷：解包根如果已存在就用它，否则尝试现解一次；解不出就交给处理阶段。
        payload = source
        if source.is_file() and source.suffix.lower() == ".zip":
            try:
                extracted = zip_payload_root(source)
                if extracted.is_dir():
                    payload = extracted
            except Exception:
                return
        source = payload
        if mode == "RESAMPLE":
            raster = None
            if source.is_file() and source.suffix.lower() in {".tif", ".tiff"}:
                raster = source
            elif source.is_dir():
                found = sorted(p for p in source.rglob("*")
                               if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
                raster = found[0] if found else None
            if raster is None:
                return
            try:
                import rasterio
                with rasterio.open(raster) as handle:
                    width, height = handle.width, handle.height
            except Exception:
                return
            verdict = precision_bounds.check_resample(
                width, height, cross_tile=normalized == "DEM",
                scale=getattr(request, "resample_scale", None),
                target_width=getattr(request, "resample_width", None),
                target_height=getattr(request, "resample_height", None))
            if not verdict["ok"]:
                raise HTTPException(status_code=422, detail=verdict["reason"])
        else:
            features = None
            try:
                if normalized in {"SHP", "DLG"} and source.is_dir():
                    import shapefile
                    shapes = sorted(p for p in source.rglob("*.shp") if p.is_file())
                    if shapes:
                        with shapefile.Reader(str(shapes[0])) as reader:
                            features = len(reader)
                elif normalized == "GEOJSON":
                    if source.is_file():
                        features = len(json.loads(source.read_text(encoding="utf-8")).get("features") or [])
                    else:
                        first = next((p for p in source.rglob("*.geojson") if p.is_file()), None)
                        if first:
                            features = len(json.loads(first.read_text(encoding="utf-8")).get("features") or [])
            except Exception:
                features = None
            if features is None:
                return
            bounds = precision_bounds.vector_bounds(features)
            if not bounds.get("feasible", True):
                raise HTTPException(status_code=422, detail=bounds["reason"])
    except HTTPException:
        raise
    except PrecisionError as error:  # pragma: no cover - 防御
        raise HTTPException(status_code=422, detail=str(error)) from error


# 产品策略：当前采用不可见取证水印，降精度会破坏后续溯源所需的几何载体，
# 因此对外声明「暂不支持降精度」。如需放开，设置
# GDS_ALLOW_PRECISION_WITH_WATERMARK=1，逻辑无需改动。
PRECISION_BLOCKED_WITH_WATERMARK = (
    "降精度功能受限：由于当前采用不可见取证技术，系统暂不支持对数据几何精度进行降低处理，"
    "否则将导致后续水印溯源失败。"
)
_PRECISION_OPEN_ENV = "GDS_ALLOW_PRECISION_WITH_WATERMARK"


def _precision_policy_enabled() -> bool:
    """策略开关：默认开启（禁止降精度）。环境变量置 1 可临时放开。"""
    return os.environ.get(_PRECISION_OPEN_ENV, "").strip().lower() not in {"1", "true", "yes", "on"}


def _policy_blocked(precision_mode, watermark_enabled: bool = True,
                    watermark_mode: str = "FORENSIC") -> str | None:
    """策略判断的唯一入口，三个接口共用，避免各写一套出现口径不一致。

    只有「水印开着 + 不可见取证（FORENSIC / HYBRID） + 选了降精度」才拒绝。
    水印关闭、纯可见水印、放行开关打开，都不拦。
    """
    if not _precision_policy_enabled():
        return None
    if (precision_mode or "NONE").upper() == "NONE":
        return None
    if bool(watermark_enabled) and (watermark_mode or "FORENSIC").upper() in {"FORENSIC", "HYBRID"}:
        return PRECISION_BLOCKED_WITH_WATERMARK
    return None


def _precision_block_reason(request) -> str | None:
    return _policy_blocked(getattr(request, "precision_mode", "NONE"),
                           getattr(request, "watermark_enabled", True),
                           getattr(request, "watermark_mode", "FORENSIC"))


def _validate_precision_request(request, normalized: str) -> None:
    """提交时立刻校验降精度参数，把「空转」变成显式 422。

    规则（与 precision.apply_precision_plan 一致）：
      * DECIMALS        仅矢量（GEOJSON/SHP/DLG/GPKG），且必须给 precision_decimals
      * DOUGLAS_PEUCKER 仅矢量（GEOJSON/SHP/DLG/GPKG），且必须给 douglas_peucker_tolerance
      * RESAMPLE        仅栅格（GEOTIFF/DEM），且必须给 resample_scale 或 resample_width/height
      * NONE            不校验
    """
    from .precision import PRECISION_MODES, PrecisionError

    # watermark_mode：FORENSIC（不可见）/ VISIBLE（仅可见）/ HYBRID（两者叠加）。
    # 可见水印目前只在栅格（GEOTIFF/DOM）路径实现——矢量/点云叠加可见文字需要各自的渲染管线，
    # 因此这些类型传 VISIBLE/HYBRID 仍显式 422，不静默降级为纯不可见。
    watermark_mode = (getattr(request, "watermark_mode", "FORENSIC") or "FORENSIC").upper()
    if watermark_mode not in {"FORENSIC", "VISIBLE", "HYBRID"}:
        raise HTTPException(status_code=422, detail=f"unsupported watermark_mode: {watermark_mode}")
    if watermark_mode != "FORENSIC" and normalized not in {"GEOTIFF", "GTIFF", "DOM"}:
        raise HTTPException(
            status_code=422,
            detail=f"watermark_mode={watermark_mode} is implemented for raster (GeoTIFF/DOM) only; got data_type={normalized}",
        )
    if watermark_mode in {"VISIBLE", "HYBRID"} and not (getattr(request, "visible_watermark_text", None) or "").strip():
        raise HTTPException(status_code=422, detail=f"watermark_mode={watermark_mode} requires visible_watermark_text")

    mode = (getattr(request, "precision_mode", "NONE") or "NONE").upper()
    if mode not in PRECISION_MODES:
        raise HTTPException(status_code=422, detail=f"unsupported precision_mode: {mode}; use one of {list(PRECISION_MODES)}")
    blocked = _precision_block_reason(request)
    if blocked:
        raise HTTPException(status_code=422, detail=blocked)
    if mode == "NONE":
        return
    vector_kinds = {"GEOJSON", "SHP", "GPKG", "GEOPACKAGE", "DLG"}
    raster_kinds = {"GEOTIFF", "GTIFF", "DOM", "DEM", "TIF", "TIFF"}
    if mode == "DECIMALS":
        if getattr(request, "precision_decimals", None) is None:
            raise HTTPException(status_code=422, detail="precision_mode=DECIMALS requires precision_decimals")
        if normalized not in vector_kinds:
            raise HTTPException(status_code=422, detail=f"precision_mode=DECIMALS is only implemented for vector data; got data_type={normalized}")
    elif mode == "DOUGLAS_PEUCKER":
        if getattr(request, "douglas_peucker_tolerance", None) is None:
            raise HTTPException(status_code=422, detail="precision_mode=DOUGLAS_PEUCKER requires douglas_peucker_tolerance")
        if normalized not in vector_kinds:
            raise HTTPException(status_code=422, detail=f"precision_mode=DOUGLAS_PEUCKER is only implemented for vector data (GeoJSON/SHP/GPKG); got data_type={normalized}")
    elif mode == "RESAMPLE":
        if normalized not in raster_kinds:
            raise HTTPException(status_code=422, detail=f"precision_mode=RESAMPLE is only implemented for raster data; got data_type={normalized}")
        if getattr(request, "resample_scale", None) is None and not (
            getattr(request, "resample_width", None) and getattr(request, "resample_height", None)
        ):
            raise HTTPException(status_code=422, detail="precision_mode=RESAMPLE requires resample_scale, or both resample_width and resample_height")

    # 载体安全边界：降精度在水印之前，降过头会让水印无处可落。
    # 这里只做「客户给的参数是否越界」的静态判定；栅格的真实尺寸要到 _apply_precision
    # 阶段读到实际数据才能算，那一层用同一条规则再判一次（见 check_resample）。
    from . import precision_bounds
    if mode == "DECIMALS":
        d = getattr(request, "precision_decimals", None)
        if d is not None and int(d) < precision_bounds.VECTOR_MIN_DECIMALS:
            raise HTTPException(
                status_code=422,
                detail=(f"precision_decimals={d} is below the watermark-safe minimum "
                        f"{precision_bounds.VECTOR_MIN_DECIMALS}; 0 decimals collapses coordinates "
                        f"and removes the watermark carrier"),
            )


def _count_gpkg_features(path: Path) -> int:
    """统计 GeoPackage 全部要素表的要素总数，用于降精度前的载体量边界检查。

    读不出来时返回 0 交由调用方判定，绝不编造数字。
    """
    import sqlite3

    total = 0
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'")]
        for table in tables:
            total += connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    finally:
        connection.close()
    return total


def _humanize_task_error(exc: BaseException) -> str:
    """把底层异常翻译成「用户能据此自助处理」的说明，同时保留原始技术细节。

    设计取舍：任务失败的 `error` 字段是直接展示给前端用户的，把
    `PermissionError: [Errno 13] ...` 这类原文抛出去会把人引向错误方向
    （比如去查 ACL，而实际问题可能是传了个目录）。这里对已知模式给中文结论，
    原文一律附在末尾，运维仍能据此定位。
    """
    raw = str(exc)
    low = raw.lower()
    fallback = "任务处理失败，请联系管理员并附上下方技术细节"

    reason = fallback
    if "errno 13" in low or "permission denied" in low or "权限不足" in raw:
        reason = ("沙箱无权读取或写入该数据文件。请确认该文件已完整上传落盘，"
                  "且不是目录或压缩包内的临时路径")
    elif "errno 28" in low or "no space left" in low:
        reason = "服务器2工作区磁盘空间不足，请清理 incoming 目录中的失败上传与临时文件后重试"
    elif "no such file" in low or "not found" in low or "不存在" in raw:
        reason = "数据文件不存在或已被清理，请重新上传后再提交任务"
    elif "is a directory" in low:
        reason = "降精度需要单个数据文件，当前给出的是目录，请检查压缩包内容"
    elif "zpkg" in low or "not a database" in low:
        reason = "数据文件已损坏或不是有效的 GeoPackage，请重新导出后上传"
    elif "timed out" in low or "timeout" in low:
        reason = "处理超时，数据可能过大，请拆分后重试或联系管理员调整超时时间"
    elif "unsupported" in low or "only implemented for" in low:
        reason = "当前数据类型不支持所选的降精度方式，请改选匹配的方式"

    return f"{reason}。技术细节：{raw}"


def _resolve_precision_payload(source: Path, normalized: str) -> Path:
    """把「上传目录 / ZIP 解包目录 / 单文件」统一解析成降精度算子要的单个数据文件。

    降精度（DECIMALS / DOUGLAS_PEUCKER / RESAMPLE）在实现上都是「读一个文件、写一个文件」，
    因此这里必须交出真实文件，而不是目录。找不到唯一载荷时直接抛 PrecisionError，
    由调用方转成 422 把原因告诉调用方，绝不让目录流到算子内部变成 Errno 13。
    """
    from .precision import PrecisionError

    source = Path(source)
    if source.is_file():
        return source
    if not source.is_dir():
        raise PrecisionError(f"降精度载荷不存在：{source}")

    # 按数据类型给出「后缀优先顺序」。同一个包可能同时含 .shp 与 .gpkg（DLG 常见），
    # 这里必须与下游 _process_task 的取舍一致：它先找 .gpkg，找不到才用 .shp。
    # 另外 v1 任务接口把 DLG 别名成 SHP，所以 SHP 也必须接受 .gpkg，否则会把
    # 一个完好的 GeoPackage 包判成「找不到数据文件」。
    suffix_order = {
        "GPKG": [".gpkg"],
        "GEOPACKAGE": [".gpkg"],
        "SHP": [".gpkg", ".shp"],
        "DLG": [".gpkg", ".shp"],
        "GEOJSON": [".geojson", ".json"],
        "GEOTIFF": [".tif", ".tiff"],
        "GTIFF": [".tif", ".tiff"],
        "DEM": [".tif", ".tiff"],
        "DOM": [".tif", ".tiff"],
    }
    order = suffix_order.get(normalized.upper(), [".gpkg", ".shp", ".tif", ".tiff", ".geojson"])

    found_any: list[Path] = []
    for suffix in order:
        candidates = sorted(p for p in source.rglob("*")
                            if p.is_file() and p.suffix.lower() == suffix
                            and not p.name.startswith("."))
        if not candidates:
            continue
        found_any.extend(candidates)
        if len(candidates) == 1:
            return candidates[0]
        names = ", ".join(p.name for p in candidates[:5])
        raise PrecisionError(
            f"降精度载荷目录内有 {len(candidates)} 个 {suffix} 文件，无法确定处理哪个：{names}"
            "（请让压缩包内只保留一份数据）")

    raise PrecisionError(
        f"降精度载荷目录内找不到可处理的数据文件：{source}"
        f"（期望后缀 {order}；请确认压缩包内容完整）")


def _apply_precision(request, source: Path, normalized: str, output_root: Path) -> tuple[Path, dict | None]:
    """在嵌入水印之前先做降精度。返回 (用于后续处理的路径, 报告)。

    未要求降精度（mode=NONE）时原样返回源路径，报告为 None。
    参数与数据类型不匹配时抛 PrecisionError（由调用方转 422），绝不静默空转。
    """
    from .precision import apply_precision_plan, PrecisionError  # noqa: F401

    mode = (getattr(request, "precision_mode", "NONE") or "NONE").upper()
    if mode == "NONE":
        return source, None

    # 降精度算子只认「单个数据文件」，不认解包后的 payload 目录。
    # 上传 ZIP 时 source 是 .../payload 目录，里面才是真正的 .gpkg/.tif/.shp，
    # 直接传给 apply_precision_plan 会拿目录去做 copy2/open，报
    # PermissionError: [Errno 13] Permission denied: '...\payload'
    # （Windows 对目录做文件级读写就是这个错误码，很容易被误读成 ACL 问题）。
    # 这里先统一解析成真实载荷，后续边界检查与降精度都用它。
    source = _resolve_precision_payload(Path(source), normalized)

    # 降精度算子的分支由「真实载荷类型」决定，而不是调用方声明的 data_type。
    # DLG 在上游被别名成 SHP，但实际载荷常常是 .gpkg；若仍按 SHP 走，
    # process_shapefile_geometry 会报 "no .shp file found in source"。
    # 下游 _process_task 本身就是按内容分派的，这里保持一致。
    payload_suffix = source.suffix.lower()
    operator_type = normalized
    if payload_suffix == ".gpkg":
        operator_type = "GPKG"
    elif payload_suffix == ".shp":
        operator_type = "SHP"
    elif payload_suffix in {".tif", ".tiff"} and normalized in {"GEOTIFF", "GTIFF", "DEM", "DOM", "SHP", "DLG"}:
        operator_type = "GEOTIFF"
    elif payload_suffix in {".geojson", ".json"}:
        operator_type = "GEOJSON"

    # 载体安全边界检查：降精度在水印之前，降过头会让水印无处可落。
    # 到这里 `source` 已是真实载荷（ZIP 已解包、目录已定位），可以量出真实载体量。
    from . import precision_bounds
    if mode == "RESAMPLE":
        raster = None
        if Path(source).is_file() and Path(source).suffix.lower() in {".tif", ".tiff"}:
            raster = Path(source)
        elif Path(source).is_dir():
            found = sorted(p for p in Path(source).rglob("*")
                           if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
            raster = found[0] if found else None
        if raster is not None:
            dims = {}
            try:
                import rasterio
                with rasterio.open(raster) as src:
                    dims = {"width": src.width, "height": src.height}
            except Exception:
                dims = {}
            verdict = precision_bounds.check_resample(
                dims.get("width"), dims.get("height"),
                cross_tile=normalized == "DEM",
                scale=getattr(request, "resample_scale", None),
                target_width=getattr(request, "resample_width", None),
                target_height=getattr(request, "resample_height", None),
            )
            if not verdict["ok"]:
                from .precision import PrecisionError
                raise PrecisionError(verdict["reason"])
    elif mode == "DOUGLAS_PEUCKER":
        features = None
        try:
            if operator_type == "GPKG" and Path(source).is_file():
                features = _count_gpkg_features(Path(source))
            elif normalized in {"SHP", "DLG"} and Path(source).is_dir():
                import shapefile
                shapes = sorted(p for p in Path(source).rglob("*.shp") if p.is_file())
                if shapes:
                    with shapefile.Reader(str(shapes[0])) as reader:
                        features = len(reader)
            elif normalized == "GEOJSON":
                text = Path(source).read_text(encoding="utf-8") if Path(source).is_file() else \
                    next(p for p in Path(source).rglob("*.geojson") if p.is_file()).read_text(encoding="utf-8")
                features = len(json.loads(text).get("features") or [])
        except Exception:
            features = None
        if features is not None:
            bounds = precision_bounds.vector_bounds(features)
            if not bounds.get("feasible", True):
                from .precision import PrecisionError
                raise PrecisionError(bounds["reason"])

    suffix = {"RESAMPLE": ".tif"}.get(mode, "")
    if mode in {"DECIMALS", "DOUGLAS_PEUCKER"}:
        # 矢量降精度：按真实载荷类型决定产物，保证与算子分支一致。
        if operator_type == "GPKG":
            target = output_root / "precision_vector.gpkg"
        elif operator_type == "SHP":
            target = output_root / "precision_vector"
        else:
            target = output_root / f"precision_{mode.lower()}.geojson"
    else:
        target = output_root / f"precision_{mode.lower()}{suffix}"
    output_root.mkdir(parents=True, exist_ok=True)
    job = {
        "source": str(source), "destination": str(target), "mode": mode,
        "decimals": getattr(request, "precision_decimals", None),
        "tolerance": getattr(request, "douglas_peucker_tolerance", None),
        "scale": getattr(request, "resample_scale", None),
        "width": getattr(request, "resample_width", None),
        "height": getattr(request, "resample_height", None),
        "method": getattr(request, "resample_method", "bilinear"),
        "data_type": operator_type,
    }
    # 与嵌入水印一致：优先在沙箱账户 gds_sbx 内执行降精度；沙箱未启用时退回本进程。
    fallback_note = ""
    if sandbox_compute.enabled():
        job_path = output_root / "precision_job.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        try:
            report = sandbox_compute.run("precision_plan", str(job_path))
            return target, (report or {})
        except Exception as exc:  # noqa: BLE001
            # 沙箱子进程起不来（Windows 0xC0000142 DLL 初始化失败：受限账户读不到
            # venv 所在目录）时不能让整条任务失败——降精度本身不涉及不可信代码执行，
            # 退回本进程直算，并把退化原因写进报告，便于事后追查。
            if "3221225794" not in str(exc) and "-1073741502" not in str(exc):
                raise
            fallback_note = f"sandbox_unavailable: {str(exc)[:160]}"
    report = apply_precision_plan(
        source, target, mode=mode, decimals=job["decimals"], tolerance=job["tolerance"],
        scale=job["scale"], width=job["width"], height=job["height"],
        method=job["method"], data_type=normalized,
    )
    if fallback_note and isinstance(report, dict):
        report["degraded"] = fallback_note
    return target, report


def _prune_scratch_dirs(root: Path, keep_hours: float = 2.0, keep_min: int = 8) -> None:
    """清理溯源判定的临时工作目录。

    每次 check-any / check-geometric 都会在 root 下建一个 uuid 目录，里面是还原后的
    栅格副本（单次实测约 290 MB）。之前从不删除：43 次判定就占满 11 GB，
    直接把 V: 盘挤到 23.8 GB 可用，导致一个 28.2 GB 的 OSGB 任务因空间不足失败。

    保留最近 keep_min 个 + keep_hours 内的，其余删除；删除失败不影响主流程
    （工作目录残留只会占空间，不能让判定请求因此 500）。
    """
    try:
        if not root.is_dir():
            return
        entries = []
        for item in root.iterdir():
            if not item.is_dir():
                continue
            try:
                entries.append((item.stat().st_mtime, item))
            except OSError:
                continue
        entries.sort(reverse=True)
        cutoff = time.time() - keep_hours * 3600
        for idx, (mtime, item) in enumerate(entries):
            if idx < keep_min or mtime >= cutoff:
                continue
            shutil.rmtree(item, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def _process_task(request: TaskRequest, source: Path, service: ProcessingService, preflight: list[dict] | None = None):
    normalized = request.data_type.upper()
    # 别名表必须与 /api/v1/tasks 入口保持一致。入口已把 DLG/THEMATIC_VECTOR 归一到 SHP、
    # DOM 归一到 GEOTIFF、MESH 归一到 OSGB；这里再兜一次，避免任何直接调用 _process_task
    # 的路径（测试、内部调用、未来的新入口）漏掉别名而落到 not implemented。
    normalized = {
        "DOM": "GEOTIFF", "DLG": "DLG", "MESH": "OSGB",
        "THEMATIC_VECTOR": "SHP",
    }.get(normalized, normalized)

    # 降精度必须发生在水印嵌入之前：先降精度再嵌水印，水印才落在最终交付的像素/坐标上。
    precision_report = None
    precision_mode = (getattr(request, "precision_mode", "NONE") or "NONE").upper()
    if precision_mode != "NONE":
        staging = Path(source).parent / f".gds_precision_{secrets.token_hex(8)}"
        source, precision_report = _apply_precision(request, Path(source), normalized, staging)

    # 关水印只是不嵌水印，降精度已经在上一步做完；这里只把结果入库并写清单。
    if not getattr(request, "watermark_enabled", True):
        return service.process_unwatermarked(
            request.project_id, source, request.data_type, request.classification,
            getattr(request, "anchor_granularity", "UPLOAD"), precision_report=precision_report)

    if normalized in {"GEOJSON"}:
        return service.process(
            request.project_id,
            "GeoJSON",
            source,
            user_id=request.user_id,
            timestamp="task",
            decimals=request.precision_decimals,
            classification=request.classification,
            processing_profile=request.processing_profile,
            precision_report=precision_report,
        )
    # GPKG 为一等数据类型：前端可以直接按数据库类型提交，不必伪装成 DLG/SHP。
    # 载荷既可能是单个 .gpkg，也可能是解包后含 .gpkg 的目录。
    if normalized in {"GPKG", "GEOPACKAGE"}:
        gpkg_files = sorted(path for path in Path(source).rglob("*.gpkg") if path.is_file()) \
            if Path(source).is_dir() else ([Path(source)] if Path(source).suffix.lower() == ".gpkg" else [])
        if not gpkg_files:
            # S1 按文件扩展名映射格式，ZIP 承载的 SHP 会被声明成 GPKG（BE-DLG）。载荷里没有
            # .gpkg 但确实带 .shp 时按实际内容走 SHP 处理器，不让整条分发因声明不符而失败。
            shp_files = sorted(path for path in Path(source).rglob("*.shp") if path.is_file()) \
                if Path(source).is_dir() else []
            if shp_files:
                return service.process(
                    request.project_id, "SHP", source, user_id=request.user_id, timestamp="task",
                    watermark_text=request.watermark_text, classification=request.classification,
                    precision_report=precision_report,
                )
            raise ValueError(f"GPKG payload must contain a .gpkg file, looked in {source}")
        if len(gpkg_files) != 1:
            names = ", ".join(p.name for p in gpkg_files[:5])
            raise ValueError(f"GPKG payload must contain exactly one .gpkg file, found {len(gpkg_files)}: {names}")
        return service.process(
            request.project_id, "GPKG", gpkg_files[0], user_id=request.user_id, timestamp="task",
            watermark_text=request.watermark_text, classification=request.classification,
            precision_report=precision_report,
        )
    if normalized in {"SHP", "DLG"}:
        gpkg_files = sorted(path for path in Path(source).rglob("*.gpkg") if path.is_file()) if Path(source).is_dir() else ([Path(source)] if Path(source).suffix.lower() == ".gpkg" else [])
        if gpkg_files:
            if len(gpkg_files) != 1:
                raise ValueError("DLG GeoPackage payload must contain exactly one .gpkg file")
            return service.process(
                request.project_id, "GPKG", gpkg_files[0], user_id=request.user_id, timestamp="task",
                watermark_text=request.watermark_text, classification=request.classification,
                precision_report=precision_report,
            )
        return service.process(
            request.project_id, "SHP", source, user_id=request.user_id, timestamp="task",
            watermark_text=request.watermark_text, classification=request.classification,
            precision_report=precision_report,
        )
    if normalized in {"GEOTIFF", "GTIFF"}:
        raster_files = sorted(path for path in Path(source).rglob("*") if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}) if Path(source).is_dir() else [Path(source)]
        if len(raster_files) != 1:
            raise ValueError(f"GeoTIFF/DOM payload must contain exactly one .tif/.tiff file, found {len(raster_files)}")
        watermark = _parse_watermark_int(request.watermark_text)
        return service.process(request.project_id, "GeoTIFF", raster_files[0], watermark=watermark, classification=request.classification, preflight=preflight, precision_report=precision_report,
                               watermark_mode=getattr(request, "watermark_mode", "FORENSIC"),
                               visible_watermark_text=getattr(request, "visible_watermark_text", None))
    if normalized == "DEM":
        return service.process(request.project_id, "DEM", source, watermark=_parse_watermark_int(request.watermark_text), classification=request.classification, precision_report=precision_report)
    if normalized == "POINT_CLOUD":
        return service.process(request.project_id, "POINT_CLOUD", source, watermark_text=request.watermark_text, classification=request.classification, precision_report=precision_report)
    if normalized in {"TEXTURES", "OSGB_TEXTURES", "OSGB"}:
        watermark = _parse_watermark_int(request.watermark_text)
        profile = request.processing_profile
        if normalized == "OSGB" and profile == "PERFORMANCE":
            profile = "STRICT_PER_MEMBER"
        return service.process(request.project_id, "OSGB" if normalized == "OSGB" else "TEXTURES", source, watermark=watermark, classification=request.classification, processing_profile=profile, precision_report=precision_report)
    raise UnsupportedDataTypeError(f"task data type not implemented: {request.data_type}")
