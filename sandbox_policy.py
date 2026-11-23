"""沙箱容量策略存储：管理员可配置的策略落盘，且与实时磁盘状态分开。

设计要点
--------
1. **策略 vs 事实分离**：`quota_bytes` 是管理员给沙箱划的上限；`used_bytes` 永远是当场
   `shutil.disk_usage()` 读出来的实测值。两者不可互相推导，汇报时必须分开说。
2. **不阻塞主链路**：配额只在新上传开始和写入前被检查（fail fast），不在每个 chunk 上
   做全卷扫描。历史数据不动。
3. **可回滚**：每次修改都写一条审计事件并把旧策略留在 `history` 里，出问题能看出改了什么。
4. **卷不可用时降级**：V: 未挂载时读取策略不抛异常，返回 `mounted=False`，让管理页仍能打开，
   否则卷故障时连清理界面都进不去。
"""
from __future__ import annotations

import json
import os
import sqlite3
import shutil
from pathlib import Path
from typing import Any

# 延迟导入避免循环依赖（sandbox_pool 不依赖 policy，policy 需要 pool 统计卷数）
def _count_pool_volumes(state_dir: Path, workspace: Path) -> int:
    try:
        from .sandbox_pool import SandboxPoolStore
        pool = SandboxPoolStore(state_dir / "sandbox_pool.sqlite3", workspace)
        try:
            return len(pool.volumes())
        finally:
            pool.close()
    except Exception:
        return 1

DEFAULT_QUOTA_RATIO = 0.90  # 默认给沙箱用到卷容量的 90%，留 10% 给 SQLite WAL / 审计 / 临时文件
MIN_QUOTA_BYTES = 1 << 30   # 1 GiB：低于这个值平台自身的 state/audit 都放不下
TABLES_AREA = ("incoming", "projects", "exports")


class SandboxPolicyStore:
    """沙箱容量策略：持久化到 workspace 的独立 SQLite，避免和任务库同锁竞争。"""

    def __init__(self, database_path: Path | str, workspace: Path | str):
        self.path = Path(database_path)
        self.workspace = Path(workspace)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sandbox_policy (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sandbox_policy_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                changed_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                changed_json TEXT NOT NULL,
                reason TEXT
            );
            """
        )
        self.db.commit()

    # ---- policy -------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM sandbox_policy WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except (TypeError, ValueError):
            return default

    def _set(self, key: str, value: Any) -> None:
        self.db.execute(
            "INSERT INTO sandbox_policy(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )

    def policy(self) -> dict[str, Any]:
        return {
            "quota_bytes": self._get("quota_bytes"),
            "updated_at": self._get("updated_at"),
            "updated_by": self._get("updated_by"),
            "history": [dict(r) for r in self.db.execute(
                "SELECT id,changed_at,actor,reason,changed_json FROM sandbox_policy_history "
                "ORDER BY id DESC LIMIT 20")],
        }

    def update_policy(self, *, quota_bytes: int | None, actor: str, reason: str = "",
                      now: str | None = None) -> dict[str, Any]:
        before = {k: v for k, v in self.policy().items() if k != "history"}
        if quota_bytes is not None:
            volume = self.volume()
            if volume["mounted"] and quota_bytes > volume["capacity_bytes"]:
                raise ValueError(
                    f"配额 {quota_bytes} 超过卷容量 {volume['capacity_bytes']}；"
                    "如需更大空间必须先扩容加密卷")
            if quota_bytes < MIN_QUOTA_BYTES:
                raise ValueError(f"配额不能低于 {MIN_QUOTA_BYTES} 字节（1 GiB）")
            self._set("quota_bytes", int(quota_bytes))
        stamp = now or ""
        self._set("updated_at", stamp)
        self._set("updated_by", actor)
        after = {k: v for k, v in self.policy().items() if k != "history"}
        diff = {k: {"from": before.get(k), "to": after.get(k)}
                for k in after if before.get(k) != after.get(k)}
        self.db.execute(
            "INSERT INTO sandbox_policy_history(changed_at,actor,changed_json,reason) VALUES(?,?,?,?)",
            (stamp, actor, json.dumps(diff or {"noop": True}, ensure_ascii=False), reason),
        )
        self.db.commit()
        return self.policy()

    # ---- facts --------------------------------------------------------
    def volume(self) -> dict[str, Any]:
        """卷的实时状态。卷不可用时返回 mounted=False 而不是抛异常。"""
        root = self.workspace
        if not root.exists():
            return {"mounted": False, "path": str(root), "capacity_bytes": 0,
                    "used_bytes": 0, "free_bytes": 0, "percent_used": None}
        try:
            usage = shutil.disk_usage(str(root))
        except OSError as exc:
            return {"mounted": False, "path": str(root), "capacity_bytes": 0,
                    "used_bytes": 0, "free_bytes": 0, "percent_used": None,
                    "error": f"{type(exc).__name__}: {exc}"}
        return {
            "mounted": True,
            "path": str(root),
            "capacity_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "percent_used": round(usage.used / usage.total, 4) if usage.total else None,
        }

    def area_usage(self) -> dict[str, Any]:
        """各数据区文件数与字节数。逐文件 stat，只用于管理页展示，不做写路径依赖。"""
        out: dict[str, Any] = {}
        for name in TABLES_AREA + ("sandbox", "state", "audit"):
            base = self.workspace / name
            files = 0
            size = 0
            if base.exists():
                for dirpath, _dirnames, filenames in os.walk(base):
                    for fn in filenames:
                        files += 1
                        try:
                            size += os.path.getsize(os.path.join(dirpath, fn))
                        except OSError:
                            pass
            out[name] = {"bytes": size, "files": files}
        return out

    def status(self) -> dict[str, Any]:
        policy = self.policy()
        volume = self.volume()
        quota = policy.get("quota_bytes")
        if quota is None and volume["mounted"]:
            quota = int(volume["capacity_bytes"] * DEFAULT_QUOTA_RATIO)
            quota_source = "default_ratio"
        else:
            quota_source = "configured" if quota is not None else "unknown"
        sandbox_used = sum(self.area_usage()[a]["bytes"] for a in TABLES_AREA)
        return {
            "volume": volume,
            "quota_bytes": quota,
            "quota_source": quota_source,
            "sandbox_used_bytes": sandbox_used,
            "quota_used_ratio": round(sandbox_used / quota, 4) if quota else None,
            "quota_remaining_bytes": max(0, quota - sandbox_used) if quota else None,
            "over_quota": bool(quota and sandbox_used > quota),
            "policy": {k: v for k, v in policy.items() if k != "history"},
            "updated_at": policy.get("updated_at"),
            "updated_by": policy.get("updated_by"),
        }

    def check_quota(self, incoming_bytes: int = 0, *, volume_root: Path | str | None = None) -> tuple[bool, str]:
        """写入前的配额闸门。返回 (允许, 原因)。

        ``volume_root`` 给出时按**那个卷**的用量判断（多卷部署的正确语义：
        配额是"每个卷最多用多少"，不是"整个平台加起来最多用多少"）。
        省略时退回平台总量口径，兼容单卷老部署。
        """
        if volume_root is None:
            volume = self.volume()
            if not volume["mounted"]:
                return False, "sandbox volume is not mounted"
            quota = self.policy().get("quota_bytes")
            if quota is None:
                quota = int(volume["capacity_bytes"] * DEFAULT_QUOTA_RATIO)
            used = sum(self.area_usage()[a]["bytes"] for a in TABLES_AREA)
        else:
            root = Path(volume_root)
            if not root.exists():
                return False, f"sandbox volume not available: {root}"
            try:
                capacity = shutil.disk_usage(str(root)).total
            except OSError as exc:
                return False, f"sandbox volume unreadable: {exc}"
            quota = self._volume_quota(root)
            if quota is None:
                quota = int(capacity * DEFAULT_QUOTA_RATIO)
            # 用**本平台数据区**的占用，不用整盘已用量：多沙箱可能落在同一物理盘上，
            # 整盘用量会把别处的文件算进来，导致配额一设就判满。
            used = self._area_usage_bytes(root)
        if used + incoming_bytes > quota:
            return False, (f"沙箱配额不足：已用 {used} 字节 + 本次 {incoming_bytes} 字节 "
                           f"超过配额 {quota} 字节，请管理员扩容或清理")
        return True, ""

    def _volume_quota(self, root: Path) -> int | None:
        key = f"quota_bytes::{root}"
        configured = self._get(key)
        if configured is not None:
            return int(configured)
        # 老部署只有一条全局 quota_bytes；单卷时沿用它，多卷时不能拿它去限制每个卷。
        return self._get("quota_bytes") if self._count_volumes() <= 1 else None

    def _area_usage_bytes(self, root: Path) -> int:
        """只统计本平台数据区的占用（root 下的 incoming/projects/... 等）。"""
        total = 0
        for area in TABLES_AREA:
            base = root / area
            if not base.exists():
                continue
            for dirpath, _dirnames, filenames in os.walk(base):
                for fn in filenames:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, fn))
                    except OSError:
                        pass
        return total

    def _count_volumes(self) -> int:
        return _count_pool_volumes(self.path.parent, self.workspace)

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass
