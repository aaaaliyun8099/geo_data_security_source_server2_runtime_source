"""流转业务内核：用户 / 项目 / 三区数据资产 / 申请审批 / 借阅授权 / 归还 / 到期提醒 / 告警 / 版本。

文档《记录2》大类 1–3 的后端落点
--------------------------------
1 导入：用户申请入项目 → 管理员建项目 → 管理员上传入沙箱（走既有 ingest 流程）→
  资产按「资源区 / 作业区 / 分发区」三区组织并按用户隔离 → 用户提交下载/借阅申请
  （需求描述必选、类型、年份、需求人员、借阅时长、空间范围线）→ 管理员两条审批路径
  （直接发放 / 处理后放入分发区再发放）→ 授权带使用期限与剩余天数 → 到期提醒。
2 归还：归还登记 → 删除该用户名下沙箱副本 → 状态闭环；导出（到用户沙箱）与借出
  （带出平台，走既有 data_loans 水印核验）在字段上分开表达。
3 更新：用户在沙箱编辑后回传新版本 → 版本链登记与审核，原资产不动。

诚实边界（代码即声明）
----------------------
* 本模块只做**台账与裁决**：出沙箱告警的真实性来自调用方上报（沙箱监控 / 客户端 agent /
  ElastAlert 规则），本模块负责落库、去重、分级与可追溯，不假装自己在盯文件系统。
* 沙箱副本删除只允许删除 `exports/users/<user_id>/` 之下的路径；越界一律 403 并记告警。
* 每个资产必须有平台水印：发布时要么绑定已审批成果（result_id），要么显式给出水印号，
  两者都没有直接拒绝——「沙箱里有无水印数据」是文档明令禁止的状态。
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .timefmt import now_local

# ── 枚举 ────────────────────────────────────────────────────────────────
ROLE_ADMIN = "ADMIN"
ROLE_USER = "USER"
ROLES = {ROLE_ADMIN, ROLE_USER}

AREA_RESOURCE = "RESOURCE"        # 管理员上传的基础数据
AREA_WORK = "WORK"                # 用户自己的作业区
AREA_DISTRIBUTION = "DISTRIBUTION"  # 管理员处理后分发给用户
AREAS = {AREA_RESOURCE, AREA_WORK, AREA_DISTRIBUTION}

ASSET_ACTIVE = "ACTIVE"
ASSET_RECYCLED = "RECYCLED"

REQ_SUBMITTED = "SUBMITTED"
REQ_APPROVED = "APPROVED"
REQ_REJECTED = "REJECTED"
REQ_CANCELLED = "CANCELLED"
REQ_EXTEND_PENDING = "EXTENSION_PENDING"
# 延期单的终态与普通申请分开：否则列表里一条被拒的延期看起来像"数据申请被拒"。
REQ_EXTEND_APPROVED = "EXTENSION_APPROVED"
REQ_EXTEND_REJECTED = "EXTENSION_REJECTED"

GRANT_ACTIVE = "ACTIVE"
GRANT_RETURNED = "RETURNED"
GRANT_EXPIRED = "EXPIRED"
GRANT_REVOKED = "REVOKED"

ALARM_OPEN = "OPEN"
ALARM_ACKED = "ACKED"

# 告警类型 → (默认等级, 说明)。未登记的类型按 MEDIUM 处理并保留原文，避免上报被吞。
ALARM_TYPES: dict[str, tuple[str, str]] = {
    "OUT_OF_SANDBOX": ("CRITICAL", "带平台水印的数据离开沙箱（无条件告警）"),
    "RECYCLE_BIN_DELETE": ("HIGH", "数据被移入回收站"),
    "PEER_TRANSFER": ("CRITICAL", "用户间试图传输数据"),
    "CROSS_SANDBOX_ACCESS": ("CRITICAL", "试图进入他人沙箱"),
    "EXPORT_DENIED": ("HIGH", "越权或超期导出被拒绝"),
    "WATERMARK_MISSING": ("CRITICAL", "沙箱内出现无水印数据"),
    "OVERDUE": ("MEDIUM", "借阅超期"),
    "REMINDER": ("LOW", "到期提醒"),
}

VERSION_PENDING = "PENDING_REVIEW"
VERSION_APPROVED = "APPROVED"
VERSION_REJECTED = "REJECTED"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _parse_dt(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def _days_left(due_at: str, *, at: datetime | None = None) -> tuple[int, bool]:
    """剩余天数（向上取整，负数表示已超期天数）。期限当天返回 0 且未超期。"""
    due = _parse_dt(due_at)
    if due is None:
        return 0, False
    now = at or datetime.now(due.tzinfo) if due.tzinfo else (at or datetime.now())
    if due.tzinfo and now.tzinfo is None:
        now = now.replace(tzinfo=due.tzinfo)
    if now <= due:
        return max(0, -(-int((due - now).total_seconds()) // 86400)), False
    overdue = (now - due).total_seconds()
    return -max(1, int(-(-overdue // 86400))), True


def bbox_of_geometry(geometry: Any) -> list[float]:
    """从 GeoJSON Geometry/Feature/FeatureCollection 里算出 [minx,miny,maxx,maxy]。

    不引第三方几何库：申请单只需要一个可存储、可展示、可做粗筛的外包框。
    坐标非法（非数值 / 空坐标 / 越界）直接抛 ValueError，由接口层转 422。
    """
    if not isinstance(geometry, dict):
        raise ValueError("空间范围必须是 GeoJSON 对象")
    kind = geometry.get("type")
    if kind == "Feature":
        return bbox_of_geometry(geometry.get("geometry"))
    if kind == "FeatureCollection":
        boxes = [bbox_of_geometry(f) for f in (geometry.get("features") or [])]
        boxes = [b for b in boxes if b]
        if not boxes:
            raise ValueError("空间范围为空")
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]
    if kind == "GeometryCollection":
        boxes = [bbox_of_geometry(g) for g in (geometry.get("geometries") or [])]
        boxes = [b for b in boxes if b]
        if not boxes:
            raise ValueError("空间范围为空")
        return [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]

    xs: list[float] = []
    ys: list[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                      for v in node[:2]):
                xs.append(float(node[0]))
                ys.append(float(node[1]))
                if len(node) > 2:
                    for extra in node[2:]:
                        walk(extra)
                return
            for child in node:
                walk(child)

    coords = geometry.get("coordinates")
    if coords is None:
        raise ValueError(f"不支持的 GeoJSON 类型或缺少坐标：{kind}")
    walk(coords)
    if not xs or not ys:
        raise ValueError("空间范围为空")
    if not (-180 <= min(xs) and max(xs) <= 180 and -90 <= min(ys) and max(ys) <= 90):
        raise ValueError("空间范围坐标超出经纬度范围（当前只接受 WGS84 经纬度）")
    return [min(xs), min(ys), max(xs), max(ys)]


class CirculationError(ValueError):
    """业务规则拒绝。接口层按 409/403 原样返回给前端。"""


class CirculationStore:
    """流转台账（独立 SQLite 文件，与任务库、借出库解耦）。"""

    def __init__(self, path: Path | str, *, export_root: Path | str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.export_root = Path(export_root) if export_root else self.path.parent / "exports" / "users"
        self.export_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    # ── 建表 ────────────────────────────────────────────────────────────
    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS platform_users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS platform_user_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_put_user ON platform_user_tokens(user_id);

                CREATE TABLE IF NOT EXISTS projects (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS project_members (
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'MEMBER',
                    status TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    decided_at TEXT,
                    decided_by TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (project_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    area TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    data_type TEXT NOT NULL DEFAULT '',
                    classification TEXT NOT NULL DEFAULT '',
                    result_id TEXT NOT NULL DEFAULT '',
                    sandbox_path TEXT NOT NULL DEFAULT '',
                    year TEXT NOT NULL DEFAULT '',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    watermark_text TEXT NOT NULL DEFAULT '',
                    distributed_to TEXT NOT NULL DEFAULT '[]',
                    valid_until TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    recycled_at TEXT,
                    recycled_by TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_assets_project_area ON assets(project_id, area, status);
                CREATE INDEX IF NOT EXISTS idx_assets_owner ON assets(owner_id);

                CREATE TABLE IF NOT EXISTS asset_versions (
                    version_id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    version_no INTEGER NOT NULL,
                    upload_id TEXT NOT NULL DEFAULT '',
                    sandbox_path TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reviewed_by TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT,
                    review_note TEXT NOT NULL DEFAULT '',
                    UNIQUE(asset_id, version_no)
                );

                CREATE TABLE IF NOT EXISTS access_requests (
                    request_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    requester_id TEXT NOT NULL,
                    asset_ids TEXT NOT NULL DEFAULT '[]',
                    need_text TEXT NOT NULL,
                    data_type TEXT NOT NULL DEFAULT '',
                    year TEXT NOT NULL DEFAULT '',
                    requesters TEXT NOT NULL DEFAULT '[]',
                    borrow_days INTEGER NOT NULL DEFAULT 0,
                    spatial_extent TEXT NOT NULL DEFAULT '',
                    extent_bbox TEXT NOT NULL DEFAULT '[]',
                    mode TEXT NOT NULL DEFAULT 'DOWNLOAD',
                    status TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    decided_at TEXT,
                    decided_by TEXT NOT NULL DEFAULT '',
                    decision_note TEXT NOT NULL DEFAULT '',
                    distribution_asset_id TEXT NOT NULL DEFAULT '',
                    due_at TEXT NOT NULL DEFAULT '',
                    parent_request_id TEXT NOT NULL DEFAULT '',
                    parent_grant_id TEXT NOT NULL DEFAULT '',
                    grant_ids TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_req_project ON access_requests(project_id, status);
                CREATE INDEX IF NOT EXISTS idx_req_requester ON access_requests(requester_id);

                CREATE TABLE IF NOT EXISTS grants (
                    grant_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    granted_at TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    export_dir TEXT NOT NULL DEFAULT '',
                    returned_at TEXT,
                    return_note TEXT NOT NULL DEFAULT '',
                    reminder_sent_at TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_grants_user ON grants(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_grants_due ON grants(due_at);

                CREATE TABLE IF NOT EXISTS alarms (
                    alarm_id TEXT PRIMARY KEY,
                    alarm_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    project_id TEXT NOT NULL DEFAULT '',
                    asset_id TEXT NOT NULL DEFAULT '',
                    actor TEXT NOT NULL DEFAULT '',
                    target TEXT NOT NULL DEFAULT '',
                    detail TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    created_at TEXT NOT NULL,
                    acked_by TEXT NOT NULL DEFAULT '',
                    acked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_alarms_status ON alarms(status, created_at);

                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    ref_id TEXT NOT NULL DEFAULT '',
                    actor TEXT NOT NULL DEFAULT '',
                    payload TEXT NOT NULL DEFAULT '{}',
                    prev_hash TEXT NOT NULL DEFAULT '',
                    record_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )

    # ── 事件哈希链 ──────────────────────────────────────────────────────
    def _last_hash(self) -> str:
        row = self._connection.execute(
            "SELECT record_hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return (row["record_hash"] if row else "") or ""

    def record_event(self, event_type: str, *, ref_id: str = "", actor: str = "",
                     **payload: Any) -> dict[str, Any]:
        with self._lock, self._connection:
            prev = self._last_hash()
            stamp = now_local(millis=True)
            body = json.dumps({"e": event_type, "ref": ref_id, "actor": actor,
                               "payload": payload, "prev": prev, "at": stamp},
                              ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            cur = self._connection.execute(
                "INSERT INTO events(event_type, ref_id, actor, payload, prev_hash, record_hash, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (event_type, ref_id, actor, json.dumps(payload, ensure_ascii=False, sort_keys=True),
                 prev, digest, stamp))
        return {"seq": cur.lastrowid, "event_type": event_type, "ref_id": ref_id,
                "actor": actor, "payload": payload, "record_hash": digest, "created_at": stamp}

    def verify_events(self) -> dict[str, Any]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM events ORDER BY seq").fetchall()
        prev, broken = "", []
        for row in rows:
            body = json.dumps({"e": row["event_type"], "ref": row["ref_id"], "actor": row["actor"],
                               "payload": json.loads(row["payload"] or "{}"),
                               "prev": prev, "at": row["created_at"]},
                              ensure_ascii=False, sort_keys=True)
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if digest != row["record_hash"] or row["prev_hash"] != prev:
                broken.append(row["seq"])
            prev = row["record_hash"]
        return {"valid": not broken, "records": len(rows), "broken_seq": broken,
                "head": prev}

    def list_events(self, *, ref_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        clause, params = "", []
        if ref_id:
            clause, params = "WHERE ref_id=?", [ref_id]
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM events {clause} ORDER BY seq DESC LIMIT ?", (*params, limit)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload"] or "{}")
            out.append(item)
        return out

    # ── 用户 ────────────────────────────────────────────────────────────
    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_user(self, *, username: str, display_name: str = "", role: str = ROLE_USER,
                    note: str = "", actor: str = "", status: str = "ACTIVE") -> dict[str, Any]:
        username = (username or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\-]{2,64}", username):
            raise CirculationError("用户名只能包含字母、数字、下划线、点、横线，长度 2–64")
        if role not in ROLES:
            raise CirculationError(f"未知角色：{role}")
        if status not in {"ACTIVE", "PENDING", "DISABLED"}:
            raise CirculationError(f"未知账号状态：{status}")
        user_id = _new_id("usr")
        stamp = now_local()
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    "INSERT INTO platform_users(user_id,username,display_name,role,status,note,created_at,updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (user_id, username, display_name or username, role, status, note, stamp, stamp))
        except sqlite3.IntegrityError as exc:
            raise CirculationError(f"用户名已存在：{username}") from exc
        self.record_event("PLATFORM_USER_CREATED", ref_id=user_id, actor=actor,
                          username=username, role=role)
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM platform_users WHERE user_id=?", (user_id,)).fetchone()
        if row is None:
            raise KeyError(user_id)
        return dict(row)

    def find_user(self, username: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM platform_users WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None

    def list_users(self, *, role: str | None = None) -> list[dict[str, Any]]:
        clause, params = "", []
        if role:
            clause, params = "WHERE role=?", [role]
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM platform_users {clause} ORDER BY created_at", params).fetchall()
        return [dict(r) for r in rows]

    def set_user_status(self, user_id: str, status: str, *, actor: str = "") -> dict[str, Any]:
        if status not in {"ACTIVE", "DISABLED"}:
            raise CirculationError("status 只能是 ACTIVE / DISABLED")
        self.get_user(user_id)
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE platform_users SET status=?, updated_at=? WHERE user_id=?",
                (status, now_local(), user_id))
            if status == "DISABLED":
                self._connection.execute(
                    "UPDATE platform_user_tokens SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL",
                    (now_local(), user_id))
        self.record_event("PLATFORM_USER_STATUS", ref_id=user_id, actor=actor, status=status)
        return self.get_user(user_id)

    def issue_user_token(self, user_id: str, *, actor: str = "") -> dict[str, Any]:
        user = self.get_user(user_id)
        if user["status"] != "ACTIVE":
            raise CirculationError("账号已停用，不能签发令牌")
        token = "gsu_" + secrets.token_urlsafe(32)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO platform_user_tokens(token_hash,user_id,created_at) VALUES (?,?,?)",
                (self._hash_token(token), user_id, now_local()))
        self.record_event("PLATFORM_USER_TOKEN_ISSUED", ref_id=user_id, actor=actor)
        # 明文只在这里出现一次，不落库、不写事件。
        return {"user_id": user_id, "username": user["username"], "token": token,
                "note": "令牌只在本次响应中返回一次，请在前端服务端安全存储"}

    def authenticate(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT t.user_id, t.revoked_at, u.* FROM platform_user_tokens t"
                " JOIN platform_users u ON u.user_id=t.user_id WHERE t.token_hash=?",
                (self._hash_token(token),)).fetchone()
        if row is None or row["revoked_at"]:
            return None
        if row["status"] != "ACTIVE":
            return None
        return dict(row)

    def revoke_user_tokens(self, user_id: str, *, actor: str = "") -> int:
        with self._lock, self._connection:
            cur = self._connection.execute(
                "UPDATE platform_user_tokens SET revoked_at=? WHERE user_id=? AND revoked_at IS NULL",
                (now_local(), user_id))
        self.record_event("PLATFORM_USER_TOKENS_REVOKED", ref_id=user_id, actor=actor,
                          count=cur.rowcount)
        return cur.rowcount

    # ── 项目与成员 ──────────────────────────────────────────────────────
    def create_project(self, *, name: str, description: str = "", created_by: str = "") -> dict[str, Any]:
        name = (name or "").strip()
        if len(name) < 2:
            raise CirculationError("项目名称至少 2 个字符")
        project_id = _new_id("prj")
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO projects(project_id,name,description,created_by,status,created_at)"
                " VALUES (?,?,?,?,'ACTIVE',?)",
                (project_id, name, description, created_by, stamp))
        self.record_event("PROJECT_CREATED", ref_id=project_id, actor=created_by, name=name)
        return self.get_project(project_id)

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(project_id)
        return dict(row)

    def list_projects(self, *, user_id: str | None = None) -> list[dict[str, Any]]:
        if user_id is None:
            with self._lock:
                rows = self._connection.execute("SELECT * FROM projects ORDER BY created_at").fetchall()
            return [dict(r) for r in rows]
        with self._lock:
            rows = self._connection.execute(
                "SELECT p.* FROM projects p JOIN project_members m ON m.project_id=p.project_id"
                " WHERE m.user_id=? AND m.status='ACTIVE' ORDER BY p.created_at", (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def member(self, project_id: str, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM project_members WHERE project_id=? AND user_id=?",
                (project_id, user_id)).fetchone()
        return dict(row) if row else None

    def is_member(self, project_id: str, user_id: str) -> bool:
        row = self.member(project_id, user_id)
        return bool(row and row["status"] == "ACTIVE")

    def join(self, project_id: str, user_id: str, *, note: str = "") -> dict[str, Any]:
        self.get_project(project_id)
        existing = self.member(project_id, user_id)
        if existing and existing["status"] == "ACTIVE":
            return existing
        if existing and existing["status"] == "REJECTED":
            raise CirculationError("该项目已拒绝过你的申请，请联系管理员")
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO project_members(project_id,user_id,role,status,applied_at,note)"
                " VALUES (?,?,'MEMBER','PENDING',?,?)"
                " ON CONFLICT(project_id,user_id) DO UPDATE SET status='PENDING',"
                " applied_at=excluded.applied_at, note=excluded.note",
                (project_id, user_id, stamp, note))
        self.record_event("PROJECT_JOIN_APPLIED", ref_id=project_id, actor=user_id, note=note)
        return self.member(project_id, user_id) or {}

    def decide_join(self, project_id: str, user_id: str, *, approve: bool,
                    decided_by: str = "", note: str = "") -> dict[str, Any]:
        row = self.member(project_id, user_id)
        if row is None or row["status"] != "PENDING":
            raise CirculationError("没有待审批的入项目申请")
        status = "ACTIVE" if approve else "REJECTED"
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE project_members SET status=?, decided_at=?, decided_by=?, note=?"
                " WHERE project_id=? AND user_id=?",
                (status, stamp, decided_by, note, project_id, user_id))
        self.record_event("PROJECT_JOIN_DECIDED", ref_id=project_id, actor=decided_by,
                          user_id=user_id, status=status, note=note)
        return self.member(project_id, user_id) or {}

    def list_members(self, project_id: str, *, status: str | None = None) -> list[dict[str, Any]]:
        clause, params = "", [project_id]
        if status:
            clause, params = " AND status=?", [project_id, status]
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM project_members WHERE project_id=?{clause} ORDER BY applied_at",
                params).fetchall()
        return [dict(r) for r in rows]

    # ── 资产 ────────────────────────────────────────────────────────────
    def create_asset(self, *, project_id: str, area: str, owner_id: str, name: str,
                     data_type: str = "", classification: str = "", result_id: str = "",
                     sandbox_path: str = "", year: str = "", size_bytes: int = 0,
                     watermark_text: str = "", created_by: str = "",
                     distributed_to: Iterable[str] | None = None,
                     valid_until: str = "", require_watermark: bool = True) -> dict[str, Any]:
        if area not in AREAS:
            raise CirculationError(f"未知数据区：{area}")
        self.get_project(project_id)
        if area == AREA_WORK and not owner_id:
            raise CirculationError("作业区资产必须指定归属用户")
        if not (result_id or sandbox_path):
            raise CirculationError("资产必须绑定成果（result_id）或沙箱路径")
        if require_watermark:
            if not (result_id or watermark_text):
                raise CirculationError(
                    "沙箱内不允许存在无水印数据：发布资产必须绑定已审批成果或显式给出水印号")
            if result_id and not watermark_text:
                watermark_text = ""  # 允许由调用方随后补，但绑定成果视为可信来源
        asset_id = _new_id("ast")
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO assets(asset_id,project_id,area,owner_id,name,data_type,classification,"
                "result_id,sandbox_path,year,size_bytes,watermark_text,distributed_to,valid_until,"
                "status,created_by,created_at,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'ACTIVE',?,?,?)",
                (asset_id, project_id, area, owner_id, name, data_type, classification, result_id,
                 sandbox_path, year, int(size_bytes or 0), watermark_text,
                 json.dumps(sorted(set(distributed_to or [])), ensure_ascii=False),
                 valid_until, created_by, stamp, stamp))
        self.record_event("ASSET_PUBLISHED", ref_id=asset_id, actor=created_by, project_id=project_id,
                          area=area, owner_id=owner_id, name=name, result_id=result_id,
                          watermark_text=watermark_text)
        return self.get_asset(asset_id)

    @staticmethod
    def _asset_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["distributed_to"] = json.loads(item.get("distributed_to") or "[]")
        return item

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM assets WHERE asset_id=?", (asset_id,)).fetchone()
        if row is None:
            raise KeyError(asset_id)
        return self._asset_row(row)

    def visible_assets(self, *, project_id: str | None, user_id: str,
                       area: str | None = None, status: str = ASSET_ACTIVE,
                       data_type: str | None = None, year: str | None = None,
                       date_from: str | None = None, date_to: str | None = None,
                       keyword: str | None = None) -> list[dict[str, Any]]:
        """用户可见资产：资源区=项目成员可见；作业区=只看自己的；分发区=发给了自己。

        用户间隔离在这里落地：不是前端隐藏，而是查询条件本身不会返回别人的作业区资产。
        """
        clauses = ["a.status=?", "a.project_id IN (SELECT project_id FROM project_members"
                                " WHERE user_id=? AND status='ACTIVE')"]
        params: list[Any] = [status, user_id]
        if project_id:
            clauses.append("a.project_id=?")
            params.append(project_id)
        if area:
            clauses.append("a.area=?")
            params.append(area)
        if data_type:
            clauses.append("a.data_type=?")
            params.append(data_type)
        if year:
            clauses.append("a.year=?")
            params.append(year)
        if date_from:
            clauses.append("substr(a.created_at,1,10)>=?")
            params.append(date_from)
        if date_to:
            clauses.append("substr(a.created_at,1,10)<=?")
            params.append(date_to)
        if keyword:
            clauses.append("(a.name LIKE ? OR a.classification LIKE ?)")
            params.extend([f"%{keyword}%", f"%{keyword}%"])
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT a.* FROM assets a WHERE {where} ORDER BY a.created_at DESC", params).fetchall()
        out = []
        for row in rows:
            item = self._asset_row(row)
            if item["area"] == AREA_WORK and item["owner_id"] != user_id:
                continue
            if item["area"] == AREA_DISTRIBUTION:
                targets = item["distributed_to"]
                if targets and user_id not in targets:
                    continue
            out.append(item)
        return out

    def list_assets_admin(self, *, project_id: str | None = None, area: str | None = None,
                          status: str = ASSET_ACTIVE) -> list[dict[str, Any]]:
        clauses, params = ["status=?"], [status]
        if project_id:
            clauses.append("project_id=?")
            params.append(project_id)
        if area:
            clauses.append("area=?")
            params.append(area)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM assets WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
                params).fetchall()
        return [self._asset_row(r) for r in rows]

    def redistribute(self, asset_id: str, user_ids: Iterable[str], *, actor: str = "") -> dict[str, Any]:
        asset = self.get_asset(asset_id)
        if asset["area"] != AREA_DISTRIBUTION:
            raise CirculationError("只有分发区资产可以追加发放对象")
        targets = sorted(set(asset["distributed_to"]) | {u for u in user_ids if u})
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE assets SET distributed_to=?, updated_at=? WHERE asset_id=?",
                (json.dumps(targets, ensure_ascii=False), now_local(), asset_id))
        self.record_event("ASSET_REDISTRIBUTED", ref_id=asset_id, actor=actor, users=targets)
        return self.get_asset(asset_id)

    def recycle_asset(self, asset_id: str, *, actor: str = "") -> dict[str, Any]:
        asset = self.get_asset(asset_id)
        if asset["status"] == ASSET_RECYCLED:
            return asset
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE assets SET status=?, recycled_at=?, recycled_by=?, updated_at=? WHERE asset_id=?",
                (ASSET_RECYCLED, stamp, actor, stamp, asset_id))
        self.record_event("ASSET_RECYCLED", ref_id=asset_id, actor=actor)
        # 文档要求：删除到回收站也要告警。
        self.report_alarm("RECYCLE_BIN_DELETE", actor=actor, asset_id=asset_id,
                          project_id=asset["project_id"], target=asset["name"],
                          detail={"area": asset["area"], "owner_id": asset["owner_id"]})
        return self.get_asset(asset_id)

    def restore_asset(self, asset_id: str, *, actor: str = "") -> dict[str, Any]:
        asset = self.get_asset(asset_id)
        if asset["status"] != ASSET_RECYCLED:
            raise CirculationError("资产不在回收站")
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE assets SET status=?, recycled_at=NULL, recycled_by='', updated_at=? WHERE asset_id=?",
                (ASSET_ACTIVE, now_local(), asset_id))
        self.record_event("ASSET_RESTORED", ref_id=asset_id, actor=actor)
        return self.get_asset(asset_id)

    def list_recycle_bin(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        return self.list_assets_admin(project_id=project_id, status=ASSET_RECYCLED)

    # ── 申请 ────────────────────────────────────────────────────────────
    @staticmethod
    def _request_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for field in ("asset_ids", "requesters", "extent_bbox", "grant_ids"):
            try:
                item[field] = json.loads(item.get(field) or "[]")
            except (TypeError, ValueError):
                item[field] = []
        try:
            item["spatial_extent"] = json.loads(item["spatial_extent"]) if item.get("spatial_extent") else None
        except (TypeError, ValueError):
            item["spatial_extent"] = None
        if item.get("due_at") and item.get("status") in {REQ_APPROVED}:
            left, overdue = _days_left(item["due_at"])
            item["days_left"], item["overdue"] = left, overdue
        return item

    def submit_request(self, *, project_id: str, requester_id: str, asset_ids: list[str],
                       need_text: str, data_type: str = "", year: str = "",
                       requesters: Iterable[str] | None = None, borrow_days: int = 0,
                       spatial_extent: Any = None, mode: str = "DOWNLOAD") -> dict[str, Any]:
        """提交下载/借阅申请。需求描述为必选项（文档第 4 条）。"""
        if not self.is_member(project_id, requester_id):
            raise CirculationError("你不在该项目中，不能提交申请")
        need_text = (need_text or "").strip()
        if len(need_text) < 2:
            raise CirculationError("需求描述为必填项")
        if mode not in {"DOWNLOAD", "BORROW"}:
            raise CirculationError("mode 只能是 DOWNLOAD（下载到沙箱）或 BORROW（借出平台）")
        if not asset_ids:
            raise CirculationError("至少要选择一项数据资产")
        if borrow_days < 0 or borrow_days > 3650:
            raise CirculationError("借阅时长必须在 0–3650 天之间")
        if mode == "BORROW" and borrow_days <= 0:
            raise CirculationError("借出模式必须填写借阅时长（天）")
        bbox: list[float] = []
        if spatial_extent is not None:
            bbox = bbox_of_geometry(spatial_extent)
        people = sorted(set(requesters or []) | {requester_id})
        for pid in people:
            if not self.is_member(project_id, pid):
                raise CirculationError(f"共同申请人 {pid} 不是项目成员，不能加入申请")
        # 资产必须在该用户可见范围内，防止拿别人的作业区资产下单。
        visible = {a["asset_id"] for a in self.visible_assets(project_id=project_id, user_id=requester_id)}
        missing = [a for a in asset_ids if a not in visible]
        if missing:
            raise CirculationError(f"这些资产对你不可见或不存在：{', '.join(missing[:5])}")
        request_id = _new_id("rq")
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO access_requests(request_id,project_id,requester_id,asset_ids,need_text,"
                "data_type,year,requesters,borrow_days,spatial_extent,extent_bbox,mode,status,submitted_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, project_id, requester_id, json.dumps(asset_ids),
                 need_text, data_type, year, json.dumps(people), borrow_days,
                 json.dumps(spatial_extent, ensure_ascii=False) if spatial_extent else "",
                 json.dumps(bbox), mode, REQ_SUBMITTED, stamp))
        self.record_event("ACCESS_REQUEST_SUBMITTED", ref_id=request_id, actor=requester_id,
                          project_id=project_id, assets=asset_ids, mode=mode,
                          borrow_days=borrow_days, bbox=bbox, need=need_text)
        return self.get_request(request_id)

    def get_request(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM access_requests WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._request_row(row)

    def list_requests(self, *, project_id: str | None = None, requester_id: str | None = None,
                      status: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        for field, value in (("project_id", project_id), ("requester_id", requester_id),
                             ("status", status)):
            if value:
                clauses.append(f"{field}=?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM access_requests {where} ORDER BY submitted_at DESC", params).fetchall()
        return [self._request_row(r) for r in rows]

    def decide_request(self, request_id: str, *, approve: bool, decided_by: str,
                       note: str = "", due_days: int = 0, distribution_asset_id: str = "",
                       processing_required: bool = False) -> dict[str, Any]:
        """管理员审批。两条路径（文档第 5 条）：

        * 不需要处理 → 直接发放（为申请人与共同申请人各开授权、各建沙箱导出目录）。
        * 需要处理 → 管理员先处理并发布到分发区，再以此接口带 ``distribution_asset_id``
          发放；没带分发资产时申请保持 SUBMITTED 不动（避免"批了但没数据"）。
        """
        req = self.get_request(request_id)
        if req["status"] not in {REQ_SUBMITTED}:
            raise CirculationError(f"申请当前状态是 {req['status']}，不能再审批")
        if not approve:
            with self._lock, self._connection:
                self._connection.execute(
                    "UPDATE access_requests SET status=?, decided_at=?, decided_by=?, decision_note=?"
                    " WHERE request_id=?",
                    (REQ_REJECTED, now_local(), decided_by, note, request_id))
            self.record_event("ACCESS_REQUEST_REJECTED", ref_id=request_id, actor=decided_by, note=note)
            return self.get_request(request_id)

        if processing_required and not distribution_asset_id:
            raise CirculationError("需要处理后分发：请先在分发区发布成果，再带上 distribution_asset_id 审批")

        stamp = now_local()
        due_at = ""
        if due_days:
            due_at = _iso(datetime.now() + timedelta(days=due_days))
        elif req["borrow_days"]:
            due_at = _iso(datetime.now() + timedelta(days=int(req["borrow_days"])))
        else:
            due_at = _iso(datetime.now() + timedelta(days=30))

        targets = req["requesters"] or [req["requester_id"]]
        asset_ids = [distribution_asset_id] if distribution_asset_id else req["asset_ids"]
        grant_ids: list[str] = []
        for user_id in targets:
            for asset_id in asset_ids:
                asset = self.get_asset(asset_id)
                if asset["status"] != ASSET_ACTIVE:
                    raise CirculationError(f"资产 {asset_id} 不在可用状态")
                grant_id = _new_id("grn")
                export_dir = self._grant_dir(user_id, grant_id)
                with self._lock, self._connection:
                    self._connection.execute(
                        "INSERT INTO grants(grant_id,request_id,project_id,asset_id,user_id,granted_at,"
                        "due_at,status,export_dir,created_at) VALUES (?,?,?,?,?,?,?,'ACTIVE',?,?)",
                        (grant_id, request_id, req["project_id"], asset_id, user_id,
                         stamp, due_at, str(export_dir), stamp))
                grant_ids.append(grant_id)
        if distribution_asset_id:
            self.redistribute(distribution_asset_id, targets, actor=decided_by)
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE access_requests SET status=?, decided_at=?, decided_by=?, decision_note=?,"
                " distribution_asset_id=?, due_at=?, grant_ids=? WHERE request_id=?",
                (REQ_APPROVED, stamp, decided_by, note, distribution_asset_id, due_at,
                 json.dumps(grant_ids), request_id))
        self.record_event("ACCESS_REQUEST_APPROVED", ref_id=request_id, actor=decided_by,
                          grants=grant_ids, due_at=due_at, distribution_asset_id=distribution_asset_id,
                          note=note)
        return self.get_request(request_id)

    # ── 授权 / 沙箱导出目录 ─────────────────────────────────────────────
    def _grant_dir(self, user_id: str, grant_id: str) -> Path:
        """每个用户一个独立导出根，每个授权一个子目录（文档第 8 条）。"""
        root = (self.export_root / user_id).resolve()
        target = (root / grant_id).resolve()
        if root not in target.parents:
            raise CirculationError("导出目录越界，已拒绝")
        target.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _grant_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        left, overdue = _days_left(item["due_at"]) if item["status"] == GRANT_ACTIVE else (0, False)
        item["days_left"], item["overdue"] = left, overdue
        return item

    def get_grant(self, grant_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM grants WHERE grant_id=?", (grant_id,)).fetchone()
        if row is None:
            raise KeyError(grant_id)
        return self._grant_row(row)

    def list_grants(self, *, user_id: str | None = None, project_id: str | None = None,
                    request_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        clauses, params = [], []
        for field, value in (("user_id", user_id), ("project_id", project_id),
                             ("request_id", request_id), ("status", status)):
            if value:
                clauses.append(f"{field}=?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM grants {where} ORDER BY granted_at DESC", params).fetchall()
        return [self._grant_row(r) for r in rows]

    def refresh_expired(self) -> list[dict[str, Any]]:
        newly = []
        for grant in self.list_grants(status=GRANT_ACTIVE):
            _, overdue = _days_left(grant["due_at"])
            if not overdue:
                continue
            with self._lock, self._connection:
                self._connection.execute(
                    "UPDATE grants SET status=? WHERE grant_id=?", (GRANT_EXPIRED, grant["grant_id"]))
            self.record_event("GRANT_EXPIRED", ref_id=grant["grant_id"], actor="system",
                              user_id=grant["user_id"], due_at=grant["due_at"])
            self.report_alarm("OVERDUE", actor=grant["user_id"], project_id=grant["project_id"],
                              asset_id=grant["asset_id"], detail={"grant_id": grant["grant_id"],
                                                                   "due_at": grant["due_at"]})
            newly.append(self.get_grant(grant["grant_id"]))
        return newly

    def return_grant(self, grant_id: str, *, user_id: str, note: str = "",
                     delete_sandbox_copy: bool = True) -> dict[str, Any]:
        """归还：登记 + 删除该用户名下的沙箱副本（只在授权导出目录内动手）。"""
        grant = self.get_grant(grant_id)
        if grant["user_id"] != user_id:
            raise CirculationError("不能归还他人的授权")
        if grant["status"] == GRANT_RETURNED:
            raise CirculationError("该授权已归还")
        deleted, kept = [], []
        if delete_sandbox_copy:
            target = Path(grant["export_dir"] or "")
            root = (self.export_root / user_id).resolve()
            if target.exists():
                resolved = target.resolve()
                if root not in resolved.parents and resolved != root:
                    raise CirculationError("沙箱副本路径越界，已拒绝删除")
                for item in sorted(resolved.rglob("*")):
                    if item.is_file():
                        item.unlink(missing_ok=True)
                        deleted.append(str(item))
                for item in sorted(resolved.rglob("*"), reverse=True):
                    if item.is_dir():
                        item.rmdir()
                resolved.rmdir()
            else:
                kept.append("沙箱副本不存在，可能已被清理")
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE grants SET status=?, returned_at=?, return_note=? WHERE grant_id=?",
                (GRANT_RETURNED, stamp, note, grant_id))
        self.record_event("DATA_RETURNED", ref_id=grant_id, actor=user_id,
                          asset_id=grant["asset_id"], deleted=len(deleted), note=note)
        return {**self.get_grant(grant_id), "deleted_files": deleted, "notes": kept}

    def request_extension(self, grant_id: str, *, user_id: str, extra_days: int,
                          reason: str = "") -> dict[str, Any]:
        grant = self.get_grant(grant_id)
        if grant["user_id"] != user_id:
            raise CirculationError("不能为他人的授权申请延期")
        if grant["status"] != GRANT_ACTIVE:
            raise CirculationError("只有生效中的授权可以申请延期")
        if extra_days < 1 or extra_days > 365:
            raise CirculationError("延期天数必须在 1–365 天")
        req = self.get_request(grant["request_id"])
        request_id = _new_id("rq")
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO access_requests(request_id,project_id,requester_id,asset_ids,need_text,"
                "data_type,year,requesters,borrow_days,spatial_extent,extent_bbox,mode,status,"
                "submitted_at,parent_request_id,parent_grant_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (request_id, req["project_id"], user_id, json.dumps([grant["asset_id"]]),
                 f"延期申请：{reason or '继续使用'}", req["data_type"], req["year"],
                 json.dumps([user_id]), extra_days,
                 json.dumps(req["spatial_extent"], ensure_ascii=False) if req["spatial_extent"] else "",
                 json.dumps(req["extent_bbox"]), req["mode"], REQ_EXTEND_PENDING, stamp,
                 grant["request_id"], grant_id))
        self.record_event("ACCESS_EXTENSION_REQUESTED", ref_id=request_id, actor=user_id,
                          grant_id=grant_id, extra_days=extra_days)
        return self.get_request(request_id)

    def decide_extension(self, request_id: str, *, approve: bool, decided_by: str,
                         note: str = "") -> dict[str, Any]:
        req = self.get_request(request_id)
        if req["status"] != REQ_EXTEND_PENDING:
            raise CirculationError("这不是待审批的延期申请")
        if not req["parent_grant_id"]:
            raise CirculationError("延期申请缺少关联授权")
        grant = self.get_grant(req["parent_grant_id"])
        if not approve:
            with self._lock, self._connection:
                self._connection.execute(
                    "UPDATE access_requests SET status=?, decided_at=?, decision_note=?, decided_by=?"
                    " WHERE request_id=?",
                    (REQ_EXTEND_REJECTED, now_local(), note, decided_by, request_id))
            self.record_event("ACCESS_EXTENSION_REJECTED", ref_id=request_id, actor=decided_by)
            return self.get_request(request_id)
        extra = int(req["borrow_days"] or 0)
        base = _parse_dt(grant["due_at"]) or datetime.now()
        new_due = _iso(base + timedelta(days=extra))
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE grants SET due_at=?, status=? WHERE grant_id=?", (new_due, GRANT_ACTIVE, grant["grant_id"]))
            self._connection.execute(
                "UPDATE access_requests SET status=?, decided_at=?, decision_note=?, decided_by=?,"
                " due_at=? WHERE request_id=?",
                (REQ_EXTEND_APPROVED, now_local(), note, decided_by, new_due, request_id))
        self.record_event("ACCESS_EXTENSION_APPROVED", ref_id=request_id, actor=decided_by,
                          grant_id=grant["grant_id"], extra_days=extra, due_at=new_due)
        return self.get_request(request_id)

    # ── 到期提醒 / 告警 ─────────────────────────────────────────────────
    def due_reminders(self, *, within_days: int = 3, user_id: str | None = None,
                      mark_sent: bool = False) -> list[dict[str, Any]]:
        """快到期（含已超期）的授权。文档第 6 条：快到期要给用户发提醒。

        这里生成"提醒记录"，是否真的送到用户由前端/BFF 的站内信或邮件承担；
        ``mark_sent`` 会落地 reminder_sent_at，便于核查提醒是否重复。
        """
        out = []
        for grant in self.list_grants(user_id=user_id, status=GRANT_ACTIVE):
            left, overdue = _days_left(grant["due_at"])
            if not overdue and left > within_days:
                continue
            item = {**grant, "days_left": left, "overdue": overdue,
                    "message": (f"授权 {grant['grant_id']} 已超期 {-left} 天，请尽快归还"
                                if overdue else f"授权 {grant['grant_id']} 还有 {left} 天到期")}
            if mark_sent:
                with self._lock, self._connection:
                    self._connection.execute(
                        "UPDATE grants SET reminder_sent_at=? WHERE grant_id=?",
                        (now_local(), grant["grant_id"]))
                self.record_event("GRANT_REMINDER", ref_id=grant["grant_id"], actor="system",
                                  days_left=left, overdue=overdue)
            out.append(item)
        out.sort(key=lambda r: r["due_at"])
        return out

    def report_alarm(self, alarm_type: str, *, actor: str = "", project_id: str = "",
                     asset_id: str = "", target: str = "", detail: Any = None,
                     source: str = "") -> dict[str, Any]:
        severity, description = ALARM_TYPES.get(alarm_type, ("MEDIUM", "未登记类型的告警"))
        payload = {"description": description, "source": source or "platform",
                   "detail": detail if isinstance(detail, dict) else {"value": detail}}
        alarm_id = _new_id("alm")
        stamp = now_local()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO alarms(alarm_id,alarm_type,severity,project_id,asset_id,actor,target,"
                "detail,status,created_at) VALUES (?,?,?,?,?,?,?,?,'OPEN',?)",
                (alarm_id, alarm_type, severity, project_id, asset_id, actor, target,
                 json.dumps(payload, ensure_ascii=False), stamp))
        self.record_event("ALARM_RAISED", ref_id=alarm_id, actor=actor or "system",
                          alarm_type=alarm_type, severity=severity, target=target,
                          project_id=project_id, asset_id=asset_id)
        return self.get_alarm(alarm_id)

    def get_alarm(self, alarm_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute("SELECT * FROM alarms WHERE alarm_id=?", (alarm_id,)).fetchone()
        if row is None:
            raise KeyError(alarm_id)
        item = dict(row)
        item["detail"] = json.loads(item["detail"] or "{}")
        return item

    def list_alarms(self, *, status: str | None = None, alarm_type: str | None = None,
                    severity: str | None = None, project_id: str | None = None,
                    limit: int = 200) -> list[dict[str, Any]]:
        clauses, params = [], []
        for field, value in (("status", status), ("alarm_type", alarm_type),
                             ("severity", severity), ("project_id", project_id)):
            if value:
                clauses.append(f"{field}=?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM alarms {where} ORDER BY created_at DESC LIMIT ?",
                (*params, limit)).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["detail"] = json.loads(item["detail"] or "{}")
            out.append(item)
        return out

    def acknowledge_alarm(self, alarm_id: str, *, actor: str, note: str = "") -> dict[str, Any]:
        self.get_alarm(alarm_id)
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE alarms SET status=?, acked_by=?, acked_at=? WHERE alarm_id=?",
                (ALARM_ACKED, actor, now_local(), alarm_id))
        self.record_event("ALARM_ACKED", ref_id=alarm_id, actor=actor, note=note)
        return self.get_alarm(alarm_id)

    # ── 版本（大类 3：沙箱编辑后回传） ──────────────────────────────────
    def add_version(self, asset_id: str, *, upload_id: str = "", sandbox_path: str = "",
                    note: str = "", created_by: str = "") -> dict[str, Any]:
        self.get_asset(asset_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(version_no),0) AS n FROM asset_versions WHERE asset_id=?",
                (asset_id,)).fetchone()
        version_no = int(row["n"]) + 1
        version_id = _new_id("ver")
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO asset_versions(version_id,asset_id,version_no,upload_id,sandbox_path,note,"
                "status,created_by,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (version_id, asset_id, version_no, upload_id, sandbox_path, note,
                 VERSION_PENDING, created_by, now_local()))
        self.record_event("ASSET_VERSION_SUBMITTED", ref_id=version_id, actor=created_by,
                          asset_id=asset_id, version_no=version_no, upload_id=upload_id)
        return self.get_version(version_id)

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM asset_versions WHERE version_id=?", (version_id,)).fetchone()
        if row is None:
            raise KeyError(version_id)
        return dict(row)

    def list_versions(self, asset_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM asset_versions WHERE asset_id=? ORDER BY version_no DESC",
                (asset_id,)).fetchall()
        return [dict(r) for r in rows]

    def decide_version(self, version_id: str, *, approve: bool, reviewed_by: str,
                       note: str = "") -> dict[str, Any]:
        version = self.get_version(version_id)
        if version["status"] != VERSION_PENDING:
            raise CirculationError("该版本已审核过")
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE asset_versions SET status=?, reviewed_by=?, reviewed_at=?, review_note=?"
                " WHERE version_id=?",
                (VERSION_APPROVED if approve else VERSION_REJECTED, reviewed_by, now_local(),
                 note, version_id))
        self.record_event("ASSET_VERSION_REVIEWED", ref_id=version_id, actor=reviewed_by,
                          approve=approve, asset_id=version["asset_id"], note=note)
        return self.get_version(version_id)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
