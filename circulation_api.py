"""《记录2》大类 1–3 的 HTTP 接口层（账号 / 项目 / 三区资产 / 申请审批 / 授权归还 / 提醒告警 / 版本）。

认证边界（前端接入必须按这个来）
--------------------------------
* 管理员接口：``X-Admin-Token``（复用平台管理员令牌），覆盖建项目、发资产、审批、告警处置。
* 用户接口：``X-User-Token``，令牌由 ``POST /api/v1/platform/users/{id}/token`` 一次性签发，
  服务端只存 SHA-256。归属校验用令牌身份，不接受前端自报 ``user_id``。
* 机器上报（账号申请、告警上报）：``X-Service-Token``，只给 BFF / 沙箱监控 / 客户端 agent。
* 浏览器不直接持有上述任何令牌：由同源 BFF 代理注入（与既有 /api/v1 约定一致）。

所有路径前缀 ``/api/v1/platform``。
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable

import shutil

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .circulation import (
    AREAS, ASSET_ACTIVE, ROLE_ADMIN, ROLE_USER, ALARM_TYPES,
    CirculationError, CirculationStore, bbox_of_geometry,
    AREA_DISTRIBUTION, AREA_RESOURCE, AREA_WORK,
)

PREFIX = "/api/v1/platform"


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    display_name: str = Field(default="", max_length=128)
    role: str = Field(default=ROLE_USER, description="ADMIN 或 USER")
    note: str = Field(default="", max_length=512)


class UserApply(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    display_name: str = Field(default="", max_length=128)
    note: str = Field(default="", max_length=512)


class ProjectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=128)
    description: str = Field(default="", max_length=1024)


class JoinApply(BaseModel):
    user_id: str = Field(default="", max_length=64, description="留空表示本人；管理员可代他人申请")
    note: str = Field(default="", max_length=512)


class Decision(BaseModel):
    note: str = Field(default="", max_length=1024)


class AssetCreate(BaseModel):
    project_id: str = Field(min_length=1, max_length=64)
    area: str = Field(description="RESOURCE 资源区 / WORK 作业区 / DISTRIBUTION 分发区")
    owner_id: str = Field(default="", max_length=64, description="作业区必填；资源区通常留空")
    name: str = Field(min_length=1, max_length=256)
    data_type: str = Field(default="", max_length=32)
    classification: str = Field(default="", max_length=64)
    result_id: str = Field(default="", max_length=128)
    sandbox_path: str = Field(default="", max_length=2048)
    year: str = Field(default="", max_length=8)
    size_bytes: int = Field(default=0, ge=0)
    watermark_text: str = Field(default="", max_length=512)
    distributed_to: list[str] = Field(default_factory=list)
    valid_until: str = Field(default="", max_length=64)


class Redistribute(BaseModel):
    user_ids: list[str] = Field(min_length=1)


class RequestCreate(BaseModel):
    project_id: str = Field(min_length=1, max_length=64)
    asset_ids: list[str] = Field(min_length=1)
    need_text: str = Field(min_length=2, max_length=2000, description="需求描述，必填")
    data_type: str = Field(default="", max_length=32)
    year: str = Field(default="", max_length=8)
    requesters: list[str] = Field(default_factory=list, description="共同使用人（含申请人本人可不填）")
    borrow_days: int = Field(default=0, ge=0, le=3650)
    spatial_extent: dict | None = Field(default=None, description="GeoJSON Geometry/Feature，可选")
    mode: str = Field(default="DOWNLOAD", description="DOWNLOAD 下载到沙箱 / BORROW 借出平台")


class RequestApprove(BaseModel):
    due_days: int = Field(default=0, ge=0, le=3650, description="使用期限（天）；0=按申请填写或默认30天")
    distribution_asset_id: str = Field(default="", max_length=64, description="处理后分发时必填")
    processing_required: bool = Field(default=False, description="true 表示走「处理后分发」路径")
    note: str = Field(default="", max_length=1024)


class ReturnBody(BaseModel):
    note: str = Field(default="", max_length=1024)
    delete_sandbox_copy: bool = Field(default=True, description="是否删除该授权在本平台沙箱的副本")


class ExtensionApply(BaseModel):
    extra_days: int = Field(ge=1, le=365)
    reason: str = Field(default="", max_length=1024)


class AlarmReport(BaseModel):
    alarm_type: str = Field(min_length=3, max_length=64)
    actor: str = Field(default="", max_length=128)
    project_id: str = Field(default="", max_length=64)
    asset_id: str = Field(default="", max_length=64)
    target: str = Field(default="", max_length=2048)
    detail: dict = Field(default_factory=dict)
    source: str = Field(default="", max_length=128)


class VersionCreate(BaseModel):
    upload_id: str = Field(default="", max_length=128)
    sandbox_path: str = Field(default="", max_length=2048)
    note: str = Field(default="", max_length=1024)


class AlarmAck(BaseModel):
    note: str = Field(default="", max_length=1024)


def build_circulation_router(
    store: CirculationStore,
    *,
    admin_identity: Callable[[str | None], dict[str, Any]],
    require_service: Callable[[str | None], None],
    audit_record: Callable[..., None],
    export_root: Path | None = None,
    asset_source: Callable[[dict[str, Any]], Path | None] | None = None,
    result_status: Callable[[str], str | None] | None = None,
) -> APIRouter:
    router = APIRouter(prefix=PREFIX, tags=["platform-circulation"])
    root = Path(export_root) if export_root else store.export_root

    def _fail(exc: Exception) -> HTTPException:
        if isinstance(exc, KeyError):
            return HTTPException(status_code=404, detail=f"资源不存在：{exc.args[0] if exc.args else ''}")
        if isinstance(exc, CirculationError):
            # 归属类拒绝按 403，其余业务规则按 409。
            forbidden = any(word in str(exc) for word in ("他人", "越界", "不可见", "不在该项目"))
            return HTTPException(status_code=403 if forbidden else 409, detail=str(exc))
        raise exc

    def _admin(admin_token: str | None) -> dict[str, Any]:
        return admin_identity(admin_token)

    def _user(user_token: str | None, admin_token: str | None) -> dict[str, Any]:
        account = store.authenticate(user_token)
        if account is not None:
            return account
        if user_token:
            raise HTTPException(status_code=401, detail="用户令牌无效或已撤销")
        # 允许管理员令牌代运维（管理员可以看用户视角，但用户不能冒充管理员）。
        try:
            identity = admin_identity(admin_token)
        except HTTPException:
            raise HTTPException(status_code=401, detail="需要用户令牌 X-User-Token 或管理员令牌")
        return {"user_id": "", "username": identity.get("username", ""),
                "display_name": identity.get("username", ""), "role": identity.get("role", ROLE_ADMIN),
                "is_admin": True}

    def _both(user_token: str | None, admin_token: str | None) -> dict[str, Any]:
        return _user(user_token, admin_token)

    def _require_member_or_admin(project_id: str, account: dict[str, Any]) -> None:
        if account.get("is_admin") or account.get("role") == ROLE_ADMIN:
            return
        if not store.is_member(project_id, account["user_id"]):
            raise HTTPException(status_code=403, detail="你不是该项目成员")

    def _audit(event: str, **details: Any) -> None:
        try:
            audit_record(event, **details)
        except Exception:
            pass

    def _source_of(asset: dict[str, Any]) -> Path | None:
        """资产对应的真实载体：优先 result_id 的产出，其次沙箱路径。"""
        if asset.get("sandbox_path"):
            candidate = Path(asset["sandbox_path"])
            if candidate.exists():
                return candidate
        if asset_source and asset.get("result_id"):
            try:
                return asset_source(asset)
            except Exception:
                return None
        return None

    def _grant_guard(grant_id: str, account: dict[str, Any], action: str) -> tuple[dict, dict, Path]:
        """授权下载的统一闸门：身份、状态、载体三者都过才给文件。"""
        grant = store.get_grant(grant_id)
        if grant["user_id"] != account.get("user_id"):
            store.report_alarm("CROSS_SANDBOX_ACCESS", actor=account.get("username", ""),
                               project_id=grant["project_id"], asset_id=grant["asset_id"],
                               target=grant_id, detail={"action": action})
            raise HTTPException(status_code=403, detail="不能访问他人的授权")
        asset = store.get_asset(grant["asset_id"]) if grant["asset_id"] else {}
        source = _source_of(asset) if asset else None
        if grant["status"] != "ACTIVE" or grant.get("overdue"):
            store.report_alarm("EXPORT_DENIED", actor=account.get("username", ""),
                               project_id=grant["project_id"], asset_id=grant["asset_id"],
                               target=grant_id,
                               detail={"action": action, "status": grant["status"],
                                       "overdue": bool(grant.get("overdue")),
                                       "due_at": grant["due_at"]})
            raise HTTPException(status_code=403,
                                detail=f"授权已{'超期' if grant.get('overdue') else grant['status']}，不能下载；"
                                       "请先申请延期或重新提交申请")
        if source is None:
            raise HTTPException(status_code=404, detail="资产载体不存在或已被清理，请联系管理员")
        return grant, asset, source

    def _list_source_files(source: Path) -> list[dict[str, Any]]:
        if source.is_file():
            return [{"name": source.name, "size": source.stat().st_size}]
        rows = []
        for item in sorted(source.rglob("*")):
            if item.is_file():
                rows.append({"name": item.relative_to(source).as_posix(),
                             "size": item.stat().st_size})
            if len(rows) >= 5000:
                break
        return rows

    # ── 身份 / 自申请 ──────────────────────────────────────────────────
    @router.get("/me")
    def me(
        user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
        admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    ) -> dict[str, Any]:
        account = _user(user_token, admin_token)
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        projects = store.list_projects() if is_admin else store.list_projects(user_id=account["user_id"])
        return {
            "user": {"user_id": account.get("user_id", ""), "username": account.get("username", ""),
                     "display_name": account.get("display_name", ""),
                     "role": ROLE_ADMIN if is_admin else ROLE_USER},
            "ui_profile": "ADMIN" if is_admin else "USER",
            "projects": projects,
            "areas": sorted(AREAS),
            "capabilities": {
                "manage_projects": is_admin,
                "publish_assets": is_admin,
                "approve_requests": is_admin,
                "review_versions": is_admin,
                "view_all_alarms": is_admin,
                "submit_requests": not is_admin,
                "return_grants": not is_admin,
                "apply_extension": not is_admin,
                "report_versions": not is_admin,
            },
            "alarm_types": ALARM_TYPES,
        }

    @router.post("/users/apply", status_code=201)
    def apply_account(
        body: UserApply,
        service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None,
    ) -> dict[str, Any]:
        """用户申请开户（第一步）：BFF 校验登录后以服务令牌提交，账号为 PENDING 待管理员批准。"""
        require_service(service_token)
        try:
            user = store.create_user(username=body.username, display_name=body.display_name,
                                     note=body.note, status="PENDING", actor=body.username)
        except CirculationError as exc:
            raise _fail(exc) from exc
        _audit("PLATFORM_ACCOUNT_APPLIED", actor=body.username, user_id=user["user_id"])
        return {**user, "next": "等待管理员在 /api/v1/platform/users 中批准（status=PENDING）"}

    # ── 用户管理（管理员） ─────────────────────────────────────────────
    @router.post("/users", status_code=201)
    def create_user(body: UserCreate,
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            user = store.create_user(username=body.username, display_name=body.display_name,
                                     role=body.role, note=body.note,
                                     actor=actor.get("username", ""))
        except CirculationError as exc:
            raise _fail(exc) from exc
        _audit("PLATFORM_USER_CREATED", actor=actor.get("username"), user_id=user["user_id"])
        return user

    @router.get("/users")
    def list_users(role: str | None = None,
                   admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        _admin(admin_token)
        rows = store.list_users(role=role)
        return {"count": len(rows), "users": rows}

    @router.post("/users/{user_id}/approve")
    def approve_user(user_id: str,
                     admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            user = store.get_user(user_id)
            if user["status"] != "PENDING":
                raise CirculationError(f"账号当前状态是 {user['status']}，不是待审批")
            user = store.set_user_status(user_id, "ACTIVE", actor=actor.get("username", ""))
        except CirculationError as exc:
            raise _fail(exc) from exc
        return user

    @router.post("/users/{user_id}/disable")
    def disable_user(user_id: str,
                     admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        return store.set_user_status(user_id, "DISABLED", actor=actor.get("username", ""))

    @router.post("/users/{user_id}/enable")
    def enable_user(user_id: str,
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        return store.set_user_status(user_id, "ACTIVE", actor=actor.get("username", ""))

    @router.post("/users/{user_id}/token")
    def issue_token(user_id: str,
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.issue_user_token(user_id, actor=actor.get("username", ""))
        except CirculationError as exc:
            raise _fail(exc) from exc

    @router.post("/users/{user_id}/revoke-tokens")
    def revoke_tokens(user_id: str,
                      admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        return {"user_id": user_id, "revoked": store.revoke_user_tokens(user_id, actor=actor.get("username", ""))}

    # ── 项目管理 ───────────────────────────────────────────────────────
    @router.post("/projects", status_code=201)
    def create_project(body: ProjectCreate,
                       admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.create_project(name=body.name, description=body.description,
                                        created_by=actor.get("username", ""))
        except CirculationError as exc:
            raise _fail(exc) from exc

    @router.get("/projects")
    def list_projects(
        user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
        admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    ) -> dict:
        account = _both(user_token, admin_token)
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        rows = store.list_projects() if is_admin else store.list_projects(user_id=account["user_id"])
        return {"count": len(rows), "projects": rows}

    @router.post("/projects/{project_id}/join-requests", status_code=201)
    def apply_join(project_id: str, body: JoinApply,
                   user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                   admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        target = body.user_id or account.get("user_id") or ""
        if not target:
            raise HTTPException(status_code=422, detail="缺少 user_id")
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        if target != account.get("user_id") and not is_admin:
            raise HTTPException(status_code=403, detail="不能替他人申请加入项目")
        try:
            return store.join(project_id, target, note=body.note)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    @router.get("/projects/{project_id}/members")
    def list_members(project_id: str, status: str | None = None,
                     admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        _admin(admin_token)
        rows = store.list_members(project_id, status=status)
        return {"count": len(rows), "members": rows}

    @router.post("/projects/{project_id}/members/{user_id}/approve")
    def approve_member(project_id: str, user_id: str, body: Decision,
                       admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.decide_join(project_id, user_id, approve=True,
                                     decided_by=actor.get("username", ""), note=body.note)
        except CirculationError as exc:
            raise _fail(exc) from exc

    @router.post("/projects/{project_id}/members/{user_id}/reject")
    def reject_member(project_id: str, user_id: str, body: Decision,
                      admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.decide_join(project_id, user_id, approve=False,
                                     decided_by=actor.get("username", ""), note=body.note)
        except CirculationError as exc:
            raise _fail(exc) from exc

    # ── 三区资产 ───────────────────────────────────────────────────────
    @router.post("/assets", status_code=201)
    def publish_asset(body: AssetCreate,
                      admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        owner = body.owner_id
        if body.area == AREA_RESOURCE and not owner:
            owner = "RESOURCE"
        # 「沙箱里必须都是有平台水印的数据」：
        # 绑定的成果若已登记，必须已审批；成果未登记（外部/历史编号）时必须显式给出水印号。
        if body.result_id and result_status is not None:
            status = result_status(body.result_id)
            if status is not None and status not in {"APPROVED", "PARTIALLY_APPROVED"}:
                raise HTTPException(
                    status_code=409,
                    detail=(f"成果 {body.result_id} 当前状态是 {status}，未审批成果不能进沙箱；"
                            "请先调用 /api/v1/results/{result_id}/approve"))
            if status is None and not body.watermark_text:
                raise HTTPException(
                    status_code=409,
                    detail=(f"成果 {body.result_id} 在本平台没有登记记录，且未提供 watermark_text；"
                            "无法证明带平台水印，拒绝入沙箱"))
        try:
            asset = store.create_asset(
                project_id=body.project_id, area=body.area, owner_id=owner, name=body.name,
                data_type=body.data_type, classification=body.classification,
                result_id=body.result_id, sandbox_path=body.sandbox_path, year=body.year,
                size_bytes=body.size_bytes, watermark_text=body.watermark_text,
                created_by=actor.get("username", ""), distributed_to=body.distributed_to,
                valid_until=body.valid_until)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc
        _audit("PLATFORM_ASSET_PUBLISHED", actor=actor.get("username"),
               asset_id=asset["asset_id"], area=asset["area"], project_id=asset["project_id"])
        return asset

    @router.get("/assets")
    def list_assets(
        project_id: str | None = None,
        area: str | None = None,
        data_type: str | None = None,
        year: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        keyword: str | None = None,
        scope: str = Query(default="mine", description="mine=按角色可见范围；all=管理员看全部"),
        user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
        admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    ) -> dict:
        account = _both(user_token, admin_token)
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        if area and area not in AREAS:
            raise HTTPException(status_code=422, detail=f"area 只能是 {sorted(AREAS)}")
        if is_admin and scope == "all":
            rows = store.list_assets_admin(project_id=project_id, area=area)
        else:
            if not account.get("user_id"):
                raise HTTPException(status_code=422, detail="管理员查看用户视角需带 X-User-Token")
            rows = store.visible_assets(project_id=project_id, user_id=account["user_id"], area=area,
                                        data_type=data_type, year=year, date_from=date_from,
                                        date_to=date_to, keyword=keyword)
        return {"count": len(rows), "assets": rows}

    @router.get("/assets/{asset_id}")
    def get_asset(asset_id: str,
                  user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                  admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        try:
            asset = store.get_asset(asset_id)
        except KeyError as exc:
            raise _fail(exc) from exc
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        if not is_admin:
            if not store.is_member(asset["project_id"], account["user_id"]):
                raise HTTPException(status_code=403, detail="你不是该项目成员")
            if asset["area"] == AREA_WORK and asset["owner_id"] != account["user_id"]:
                raise HTTPException(status_code=403, detail="不能查看他人的作业区资产")
            if asset["area"] == AREA_DISTRIBUTION and asset["distributed_to"] \
                    and account["user_id"] not in asset["distributed_to"]:
                raise HTTPException(status_code=403, detail="该分发资产未发放给你")
        return asset

    @router.post("/assets/{asset_id}/redistribute")
    def redistribute(asset_id: str, body: Redistribute,
                     admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.redistribute(asset_id, body.user_ids, actor=actor.get("username", ""))
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    @router.delete("/assets/{asset_id}")
    def recycle_asset(asset_id: str,
                      admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            asset = store.recycle_asset(asset_id, actor=actor.get("username", ""))
        except KeyError as exc:
            raise _fail(exc) from exc
        _audit("PLATFORM_ASSET_RECYCLED", actor=actor.get("username"), asset_id=asset_id)
        return {"asset": asset, "alarm": "RECYCLE_BIN_DELETE 已生成（文档要求删除到回收站也要告警）"}

    @router.post("/assets/{asset_id}/restore")
    def restore_asset(asset_id: str,
                      admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.restore_asset(asset_id, actor=actor.get("username", ""))
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    @router.get("/recycle-bin")
    def recycle_bin(project_id: str | None = None,
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        _admin(admin_token)
        rows = store.list_recycle_bin(project_id=project_id)
        return {"count": len(rows), "assets": rows}

    # ── 申请 / 审批 ────────────────────────────────────────────────────
    @router.post("/requests", status_code=201)
    def submit_request(body: RequestCreate,
                       user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                       admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        if not account.get("user_id"):
            raise HTTPException(status_code=422, detail="提交申请需要用户令牌")
        try:
            req = store.submit_request(
                project_id=body.project_id, requester_id=account["user_id"],
                asset_ids=body.asset_ids, need_text=body.need_text, data_type=body.data_type,
                year=body.year, requesters=body.requesters, borrow_days=body.borrow_days,
                spatial_extent=body.spatial_extent, mode=body.mode)
        except (KeyError, CirculationError, ValueError) as exc:
            raise _fail(exc) from exc
        _audit("PLATFORM_ACCESS_REQUESTED", actor=account.get("username"),
               request_id=req["request_id"], project_id=body.project_id, mode=body.mode)
        return req

    @router.get("/requests")
    def list_requests(project_id: str | None = None, status: str | None = None,
                      mine: bool = False,
                      user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                      admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        requester = account["user_id"] if (mine or not is_admin) else None
        rows = store.list_requests(project_id=project_id, requester_id=requester, status=status)
        return {"count": len(rows), "requests": rows}

    @router.get("/requests/{request_id}")
    def get_request(request_id: str,
                    user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        try:
            req = store.get_request(request_id)
        except KeyError as exc:
            raise _fail(exc) from exc
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        if not is_admin and account["user_id"] not in (req["requesters"] or []) \
                and account["user_id"] != req["requester_id"]:
            raise HTTPException(status_code=403, detail="不能查看他人的申请")
        grants = store.list_grants(request_id=request_id)
        return {**req, "grants": grants}

    @router.post("/requests/{request_id}/approve")
    def approve_request(request_id: str, body: RequestApprove,
                        admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            req = store.decide_request(
                request_id, approve=True, decided_by=actor.get("username", ""), note=body.note,
                due_days=body.due_days, distribution_asset_id=body.distribution_asset_id,
                processing_required=body.processing_required)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc
        _audit("PLATFORM_REQUEST_APPROVED", actor=actor.get("username"), request_id=request_id,
               grants=req["grant_ids"])
        return {**req, "grants": store.list_grants(request_id=request_id)}

    @router.post("/requests/{request_id}/reject")
    def reject_request(request_id: str, body: Decision,
                       admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.decide_request(request_id, approve=False,
                                        decided_by=actor.get("username", ""), note=body.note)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    # ── 授权 / 归还 / 延期 ─────────────────────────────────────────────
    @router.get("/grants")
    def list_grants(project_id: str | None = None, status: str | None = None, mine: bool = False,
                    user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        rows = store.list_grants(user_id=None if (is_admin and not mine) else account["user_id"],
                                 project_id=project_id, status=status)
        return {"count": len(rows), "grants": rows,
                "due_soon": store.due_reminders(user_id=None if is_admin else account["user_id"])}

    @router.get("/grants/{grant_id}")
    def get_grant(grant_id: str,
                  user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                  admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        try:
            grant = store.get_grant(grant_id)
        except KeyError as exc:
            raise _fail(exc) from exc
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        if not is_admin and grant["user_id"] != account["user_id"]:
            raise HTTPException(status_code=403, detail="不能查看他人的授权")
        return {**grant, "asset": store.get_asset(grant["asset_id"]) if grant["asset_id"] else None}

    @router.post("/grants/{grant_id}/return")
    def return_grant(grant_id: str, body: ReturnBody,
                     user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                     admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        if is_admin and not account.get("user_id"):
            try:
                owner = store.get_grant(grant_id)["user_id"]
            except KeyError as exc:
                raise _fail(exc) from exc
            account = {**account, "user_id": owner}
        try:
            result = store.return_grant(grant_id, user_id=account["user_id"], note=body.note,
                                        delete_sandbox_copy=body.delete_sandbox_copy)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc
        _audit("PLATFORM_GRANT_RETURNED", actor=account.get("username"), grant_id=grant_id,
               deleted=len(result["deleted_files"]))
        return result

    @router.post("/grants/{grant_id}/extension", status_code=201)
    def apply_extension(grant_id: str, body: ExtensionApply,
                        user_token: Annotated[str | None, Header(alias="X-User-Token")] = None) -> dict:
        account = _user(user_token, None)
        if not account.get("user_id"):
            raise HTTPException(status_code=422, detail="申请延期需要用户令牌")
        try:
            return store.request_extension(grant_id, user_id=account["user_id"],
                                           extra_days=body.extra_days, reason=body.reason)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    @router.post("/requests/{request_id}/extension/approve")
    def approve_extension(request_id: str, body: Decision,
                          admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.decide_extension(request_id, approve=True,
                                          decided_by=actor.get("username", ""), note=body.note)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    @router.post("/requests/{request_id}/extension/reject")
    def reject_extension(request_id: str, body: Decision,
                         admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.decide_extension(request_id, approve=False,
                                          decided_by=actor.get("username", ""), note=body.note)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    @router.get("/reminders")
    def reminders(within_days: int = Query(default=3, ge=0, le=90), mark_sent: bool = False,
                  user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                  admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        rows = store.due_reminders(within_days=within_days,
                                   user_id=None if is_admin and not account.get("user_id") else account.get("user_id", ""),
                                   mark_sent=mark_sent)
        return {"count": len(rows), "within_days": within_days, "reminders": rows}

    @router.get("/exports/dir")
    def export_dir(grant_id: str | None = None,
                   user_token: Annotated[str | None, Header(alias="X-User-Token")] = None) -> dict:
        """每个用户一个独立导出根（文档第 8 条）；带 grant_id 返回该授权的子目录与占用。"""
        account = _user(user_token, None)
        if not account.get("user_id"):
            raise HTTPException(status_code=422, detail="需要用户令牌")
        user_root = (root / account["user_id"]).resolve()
        if root.resolve() not in user_root.parents:
            raise HTTPException(status_code=403, detail="导出目录越界")
        user_root.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"user_id": account["user_id"], "root": str(user_root)}
        if grant_id:
            grant = store.get_grant(grant_id)
            if grant["user_id"] != account["user_id"]:
                raise HTTPException(status_code=403, detail="不能查看他人的导出目录")
            target = Path(grant["export_dir"])
            files = [p for p in target.rglob("*") if p.is_file()] if target.exists() else []
            payload.update({"grant_id": grant_id, "dir": str(target),
                            "files": len(files),
                            "bytes": sum(f.stat().st_size for f in files),
                            "days_left": grant["days_left"], "due_at": grant["due_at"]})
        return payload

    # ── 下载（用户拿到授权后的取数路径） ──────────────────────────────
    @router.get("/grants/{grant_id}/files")
    def grant_files(grant_id: str,
                    user_token: Annotated[str | None, Header(alias="X-User-Token")] = None) -> dict:
        """列出这份授权可下载的文件（已审批成果的产出）。超期/已归还直接 403 并告警。"""
        account = _user(user_token, None)
        if not account.get("user_id"):
            raise HTTPException(status_code=422, detail="需要用户令牌")
        grant, asset, source = _grant_guard(grant_id, account, "list")
        files = _list_source_files(source)
        return {"grant_id": grant_id, "asset_id": grant["asset_id"], "name": asset.get("name", ""),
                "data_type": asset.get("data_type", ""),
                "classification": asset.get("classification", ""),
                "due_at": grant["due_at"], "days_left": grant["days_left"],
                "count": len(files), "files": files,
                "download": f"/api/v1/platform/grants/{grant_id}/files/{{filename}}"}

    @router.get("/grants/{grant_id}/files/{filename:path}")
    def grant_file(grant_id: str, filename: str,
                   user_token: Annotated[str | None, Header(alias="X-User-Token")] = None) -> FileResponse:
        account = _user(user_token, None)
        if not account.get("user_id"):
            raise HTTPException(status_code=422, detail="需要用户令牌")
        grant, asset, source = _grant_guard(grant_id, account, "download")
        base = source if source.is_dir() else source.parent
        candidate = (base / filename).resolve()
        if base.resolve() not in candidate.parents and candidate != source.resolve():
            raise HTTPException(status_code=403, detail="文件名越界，已拒绝")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"文件不存在：{filename}")
        _audit("PLATFORM_GRANT_FILE_DOWNLOADED", actor=account.get("username"),
               grant_id=grant_id, asset_id=asset.get("asset_id"), filename=filename)
        store.record_event("GRANT_FILE_DOWNLOADED", ref_id=grant_id,
                           actor=account.get("username", ""), filename=filename,
                           bytes=candidate.stat().st_size)
        return FileResponse(candidate, filename=candidate.name)

    @router.post("/grants/{grant_id}/materialize")
    def materialize_grant(grant_id: str,
                          user_token: Annotated[str | None, Header(alias="X-User-Token")] = None) -> dict:
        """把授权数据落到该用户自己的沙箱导出目录（文档：导出=到用户沙箱）。

        只在该用户名下的 `exports/users/<user_id>/<grant_id>/` 里写文件；同名同大小跳过，
        剩余空间不足时拒绝而不是写半份。大文件请用 `/files/{filename}` 流式下载，
        本接口适合"先落盘再由沙箱内软件打开"的场景。
        """
        account = _user(user_token, None)
        if not account.get("user_id"):
            raise HTTPException(status_code=422, detail="需要用户令牌")
        grant, asset, source = _grant_guard(grant_id, account, "materialize")
        target = Path(grant["export_dir"])
        root = (store.export_root / account["user_id"]).resolve()
        if root not in target.resolve().parents and target.resolve() != root:
            raise HTTPException(status_code=403, detail="导出目录越界，已拒绝")
        target.mkdir(parents=True, exist_ok=True)
        needed = sum(f["size"] for f in _list_source_files(source))
        free = shutil.disk_usage(target).free
        if needed > free:
            raise HTTPException(status_code=507,
                                detail=f"沙箱剩余空间不足：需要 {needed} 字节，可用 {free} 字节")
        copied, skipped, total = [], [], 0
        base = source if source.is_dir() else source.parent
        for item in ([source] if source.is_file() else sorted(source.rglob("*"))):
            if not item.is_file():
                continue
            rel = item.relative_to(base)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == item.stat().st_size:
                skipped.append(rel.as_posix())
                continue
            shutil.copy2(item, dest)
            copied.append(rel.as_posix())
            total += item.stat().st_size
        store.record_event("GRANT_MATERIALIZED", ref_id=grant_id,
                           actor=account.get("username", ""),
                           copied=len(copied), skipped=len(skipped), bytes=total)
        return {"grant_id": grant_id, "export_dir": str(target), "copied": copied,
                "skipped": skipped, "bytes": total, "days_left": grant["days_left"],
                "due_at": grant["due_at"]}

    # ── 告警 ───────────────────────────────────────────────────────────
    @router.post("/alarms/report", status_code=201)
    def report_alarm(body: AlarmReport,
                     service_token: Annotated[str | None, Header(alias="X-Service-Token")] = None,
                     admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        """告警上报口：沙箱监控 / 客户端 agent / ElastAlert 规则命中后调用。

        记录2 要求「具备平台水印数据出沙箱无条件报警且及时」：本接口不做静默/采样，
        每次上报都落一条 OPEN 告警并写事件链；真正的秒级发现由上报方（文件系统
        监控/ElastAlert）负责，平台侧不以采样或延迟合并替代。
        """
        if service_token is not None:
            require_service(service_token)
        else:
            _admin(admin_token)
        try:
            alarm = store.report_alarm(body.alarm_type, actor=body.actor, project_id=body.project_id,
                                       asset_id=body.asset_id, target=body.target,
                                       detail=body.detail, source=body.source)
        except CirculationError as exc:
            raise _fail(exc) from exc
        _audit("PLATFORM_ALARM_REPORTED", actor=body.actor or "agent",
               alarm_type=body.alarm_type, target=body.target)
        return alarm

    @router.get("/alarms")
    def list_alarms(status: str | None = None, alarm_type: str | None = None,
                    severity: str | None = None, project_id: str | None = None,
                    limit: int = Query(default=200, ge=1, le=1000),
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        _admin(admin_token)
        rows = store.list_alarms(status=status, alarm_type=alarm_type, severity=severity,
                                 project_id=project_id, limit=limit)
        summary: dict[str, int] = {}
        for row in rows:
            summary[row["severity"]] = summary.get(row["severity"], 0) + 1
        return {"count": len(rows), "open": sum(1 for r in rows if r["status"] == "OPEN"),
                "summary": summary, "alarms": rows}

    @router.post("/alarms/{alarm_id}/ack")
    def ack_alarm(alarm_id: str, body: AlarmAck,
                  admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.acknowledge_alarm(alarm_id, actor=actor.get("username", ""), note=body.note)
        except KeyError as exc:
            raise _fail(exc) from exc

    # ── 版本（大类 3） ─────────────────────────────────────────────────
    @router.post("/assets/{asset_id}/versions", status_code=201)
    def add_version(asset_id: str, body: VersionCreate,
                    user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        try:
            asset = store.get_asset(asset_id)
        except KeyError as exc:
            raise _fail(exc) from exc
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        if not is_admin and asset["owner_id"] not in {account.get("user_id"), "RESOURCE"}:
            raise HTTPException(status_code=403, detail="不能给他人的资产提交新版本")
        try:
            return store.add_version(asset_id, upload_id=body.upload_id,
                                     sandbox_path=body.sandbox_path, note=body.note,
                                     created_by=account.get("username", ""))
        except CirculationError as exc:
            raise _fail(exc) from exc

    @router.get("/assets/{asset_id}/versions")
    def list_versions(asset_id: str,
                      user_token: Annotated[str | None, Header(alias="X-User-Token")] = None,
                      admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        account = _both(user_token, admin_token)
        try:
            asset = store.get_asset(asset_id)
        except KeyError as exc:
            raise _fail(exc) from exc
        is_admin = bool(account.get("is_admin") or account.get("role") == ROLE_ADMIN)
        if not is_admin:
            if not store.is_member(asset["project_id"], account.get("user_id", "")):
                raise HTTPException(status_code=403, detail="你不是该项目成员")
            if asset["area"] == AREA_WORK and asset["owner_id"] != account.get("user_id"):
                raise HTTPException(status_code=403, detail="不能查看他人的版本记录")
            if asset["area"] == AREA_DISTRIBUTION and asset["distributed_to"] \
                    and account.get("user_id") not in asset["distributed_to"]:
                raise HTTPException(status_code=403, detail="该分发资产未发放给你")
        rows = store.list_versions(asset_id)
        return {"count": len(rows), "versions": rows}

    @router.post("/versions/{version_id}/approve")
    def approve_version(version_id: str, body: Decision,
                        admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.decide_version(version_id, approve=True,
                                        reviewed_by=actor.get("username", ""), note=body.note)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    @router.post("/versions/{version_id}/reject")
    def reject_version(version_id: str, body: Decision,
                       admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        actor = _admin(admin_token)
        try:
            return store.decide_version(version_id, approve=False,
                                        reviewed_by=actor.get("username", ""), note=body.note)
        except (KeyError, CirculationError) as exc:
            raise _fail(exc) from exc

    # ── 事件链 ─────────────────────────────────────────────────────────
    @router.get("/events")
    def list_events(ref_id: str | None = None, limit: int = Query(default=200, ge=1, le=2000),
                    admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        _admin(admin_token)
        return {"chain": store.verify_events(), "events": store.list_events(ref_id=ref_id, limit=limit)}

    @router.get("/events/verify")
    def verify_events(admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None) -> dict:
        _admin(admin_token)
        return store.verify_events()

    return router
