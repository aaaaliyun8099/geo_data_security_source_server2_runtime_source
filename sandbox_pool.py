"""沙箱卷池：多个沙箱卷的注册、探测、按策略选卷。

为什么要多卷而不是一个大卷
--------------------------
VeraCrypt **扩容**只能 GUI 操作（`VeraCryptExpander.exe` 无命令行参数），且必须卸载卷停机。
而**新增卷**全程 CLI：`Format.exe /create` 建容器 → `VeraCrypt.exe /letter W` 挂载，
业务不中断。所以"容量不够"的正确解法是往池里加一个卷，而不是把现有卷撑大。

数据布局
--------
每个卷都有一份平台目录骨架（`<卷根>/geo_data_security/{incoming,projects,exports,...}`）。
`state/`（哈希链、队列库、管理员凭据）**只存在于主卷**，不分散——否则链会分裂。
业务数据（上传件/中间件/成果）按选卷策略落在任意在线卷上；数据库里存的是**绝对路径**，
所以一个卷被摘掉不会让已有记录指向错误位置（只是那条数据暂时不可读，恢复挂载即恢复）。

选卷策略（`select_volume`）
---------------------------
按下列顺序过滤，第一个满足的就是目标：
1. 卷必须 `enabled` 且 `ONLINE`（挂载且在容）
2. 剩余可用 ≥ `请求字节 + reserve_bytes`（默认 2 GiB 余量）
3. 按 `priority` 降序（数字大优先）；相同优先级取**剩余空间最大**的
4. 全部不满足 → 抛 `NoCapacityError`，调用方返回 507 并给出每个卷的实际余量

刻意不做的事
------------
- **不做自动清理**：没有定时执行器，就不提供"到点自动删"的假开关。
- **不做跨卷迁移**：数据落在哪个卷就留在哪个卷；要挪需显式调用（未实现即不承诺）。
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Any

# 每个卷上的平台目录骨架。state/audit 只在主卷创建，其余卷只放业务数据。
DATA_AREAS = ("incoming", "projects", "exports", "sandbox_inbox", "provenance_geom",
              "sandbox", "results", "acceptance", "runtime")
PLATFORM_DIRNAME = "geo_data_security"
MIN_FREE_AFTER_WRITE = int(os.environ.get("GDS_SANDBOX_MIN_FREE_BYTES", 2 * 1024 ** 3))

VERACRYPT = r"C:\Program Files\VeraCrypt\VeraCrypt.exe"
VERACRYPT_FORMAT = r"C:\Program Files\VeraCrypt\VeraCrypt Format.exe"


class NoCapacityError(RuntimeError):
    """所有在线卷都放不下这次请求。携带每个卷的实测余量供接口原样返回。"""

    def __init__(self, message: str, detail: list[dict[str, Any]]):
        super().__init__(message)
        self.detail = detail


class SandboxVolume:
    """一个沙箱卷。

    ``root`` 是该卷上平台目录的根（例如 `V:\\geo_data_security`）。
    ``container`` 非空表示这是一个 VeraCrypt 容器文件，可被本平台挂载/卸载；
    为空表示卷已由外部挂好（例如 BitLocker、或管理员手工挂的盘），平台只读写不管理挂载。
    """

    def __init__(self, data: dict[str, Any]):
        self.id: str = data["id"]
        self.root: Path = Path(data["root"])
        self.container: str = data.get("container") or ""
        self.letter: str = data.get("letter") or ""
        self.priority: int = int(data.get("priority") or 0)
        self.enabled: bool = bool(data.get("enabled", True))
        # 数据库列名是 is_primary；构造时也接受 primary，两种写法都要认，
        # 否则从 SQLite 读出来的主卷会被当成普通卷（主卷保护全部失效）。
        self.primary: bool = bool(data.get("is_primary", data.get("primary", False)))
        self.label: str = data.get("label") or ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "root": str(self.root), "container": self.container,
                "letter": self.letter, "priority": self.priority, "enabled": self.enabled,
                "primary": self.primary, "label": self.label}


class SandboxPoolStore:
    """卷池持久化 + 容量探测 + 选卷。SQLite 落在**主卷** state/ 下。"""

    def __init__(self, database_path: Path | str, primary_root: Path | str):
        self.path = Path(database_path)
        self.primary_root = Path(primary_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._create_schema()
        self._ensure_primary()

    # ---- volume password ---------------------------------------------
    @property
    def secret_path(self) -> Path:
        """卷口令文件，与主卷 volume.secret 同目录同权限口径。

        为什么要落盘：建卷时用的口令如果只存在于请求里，卷一旦卸载就再也挂不回来
        （实测踩过：unmount 成功、mount 报"缺少卷口令"）。口令属于部署机密，
        存在主卷 state/ 下，不写数据库、不进审计、不回显接口。
        """
        return self.path.parent / "sandbox_volume.secret"

    def volume_password(self) -> str:
        if self.secret_path.exists():
            value = self.secret_path.read_text(encoding="utf-8").strip()
            if value:
                return value
        return os.environ.get("GDS_SANDBOX_VOLUME_PASSWORD", "")

    def remember_password(self, password: str) -> None:
        """建卷成功后记住口令，保证以后能重新挂载。"""
        if not password:
            return
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        self.secret_path.write_text(password, encoding="utf-8")
        try:
            os.chmod(self.secret_path, 0o600)
        except OSError:
            pass

    def _create_schema(self) -> None:
        with self._lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sandbox_volumes (
                    id TEXT PRIMARY KEY,
                    root TEXT NOT NULL,
                    container TEXT NOT NULL DEFAULT '',
                    letter TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    is_primary INTEGER NOT NULL DEFAULT 0,
                    label TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL DEFAULT ''
                );
                -- 建卷是分钟级的阻塞操作（VeraCrypt Format.exe 是 GUI 程序），
                -- 必须异步执行并把进度持久化，否则 HTTP 请求会超时、刷新页面就丢状态。
                CREATE TABLE IF NOT EXISTS sandbox_volume_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL DEFAULT '',
                    size_gb INTEGER NOT NULL DEFAULT 0,
                    letter TEXT NOT NULL DEFAULT '',
                    label TEXT NOT NULL DEFAULT '',
                    priority INTEGER NOT NULL DEFAULT 0,
                    volume_id TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT ''
                );
                """
            )
            self.db.commit()

    def _ensure_primary(self) -> None:
        """首次启动时把当前主卷登记进池，保证老部署升级后行为不变。"""
        with self._lock:
            row = self.db.execute("SELECT COUNT(*) FROM sandbox_volumes").fetchone()
            if row and int(row[0]) > 0:
                return
            letter = self.primary_root.drive.rstrip(":") if self.primary_root.drive else ""
            self.db.execute(
                "INSERT INTO sandbox_volumes(id,root,container,letter,priority,enabled,is_primary,label,created_at,created_by)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                # 主卷默认优先级 50（与新建卷相同），不是最高 ——
                # 否则 priority 高的大主卷会永远赢，新加的卷接不到任何数据，
                # "加卷扩容"就名存实亡。同级时按剩余空间比，空间大的自然多接。
                ("vol_primary", str(self.primary_root), "", letter, 50, 1, 1,
                 "主卷（state/链所在）", "", "system"),
            )
            self.db.commit()

    # ---- async volume creation ---------------------------------------
    def start_create_job(self, *, size_gb: int, letter: str, label: str, priority: int,
                         password: str, actor: str, now: str) -> str:
        """登记一个建卷任务并立刻返回。真正的建卷在后台线程跑。"""
        job_id = "vjob_" + os.urandom(6).hex()
        with self._lock:
            self.db.execute(
                "INSERT INTO sandbox_volume_jobs(job_id,status,stage,size_gb,letter,label,priority,volume_id,error,started_at,finished_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, "RUNNING", "queued", size_gb, letter, label, priority, "", "", now, ""))
            self.db.commit()
        worker = threading.Thread(target=self._run_create_job,
                                  args=(job_id, size_gb, letter, label, priority, password, actor),
                                  name=f"vdcreate-{job_id}", daemon=True)
        worker.start()
        return job_id

    def _run_create_job(self, job_id: str, size_gb: int, letter: str, label: str,
                        priority: int, password: str, actor: str) -> None:
        import time as _t
        from .timefmt import now_local as _now
        container_dir = Path(os.environ.get("GDS_VERACRYPT_DIR", r"D:\GeoDataSecurity\veracrypt"))
        container_path = container_dir / f"sandbox_{size_gb}GB_{letter}_{os.urandom(3).hex()}.hc"
        try:
            self._set_job(job_id, stage="formatting")
            create_container(str(container_path), size_gb, password, letter)
            # 建卷成功立刻记住口令，否则卷卸载后就挂不回来了
            self.remember_password(password)
            self._set_job(job_id, stage="registering")
            vol = self.add(root=f"{letter}:\\{PLATFORM_DIRNAME}", container=str(container_path),
                           letter=letter, priority=priority, label=label,
                           actor=actor, now=_now())
            self._set_job(job_id, status="DONE", stage="done", volume_id=vol.id,
                          finished_at=_now())
        except Exception as exc:  # noqa: BLE001 —— 后台线程不能把异常抛给任何人
            detail = f"{type(exc).__name__}: {exc}"
            # 失败时把可能已挂载的盘符卸掉，不留孤儿
            try:
                unmount_letter(letter)
            except Exception:
                pass
            try:
                if container_path.exists():
                    container_path.unlink()
            except Exception:
                pass
            self._set_job(job_id, status="FAILED", stage="failed", error=detail[:900],
                          finished_at=_now())

    def _set_job(self, job_id: str, **fields: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self.db.execute(f"UPDATE sandbox_volume_jobs SET {cols} WHERE job_id=?",
                            (*fields.values(), job_id))
            self.db.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM sandbox_volume_jobs WHERE job_id=?",
                                  (job_id,)).fetchone()
        return dict(row) if row else None

    def jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM sandbox_volume_jobs ORDER BY started_at DESC LIMIT ?",
                (int(limit),)).fetchall()
        return [dict(r) for r in rows]

    def ensure_all_mounted(self) -> list[dict[str, Any]]:
        """把池里所有「容器存在但没挂上」的卷挂回来。启动时调用。

        为什么必须有：`startup/gds_boot.ps1` 只挂主卷 V:（state/ 在它上面）。
        加了数据卷之后，机器一重启，V: 回来了、E:/F: 全没了 —— 卷池里它们变 OFFLINE，
        选卷时被过滤掉，所有上传静默挤回主卷，多出来的容量等于没加。
        （这个缺口在加第一个 270 GB 卷之后才发现，属于必须自动化的运维动作。）
        """
        password = self.volume_password()
        if not password:
            return [{"skipped": "no volume password"}]
        import time as _t
        results: list[dict[str, Any]] = []
        for vol in self.volumes():
            if not vol.container or not vol.letter:
                continue
            state = self.volume_state(vol)
            if state["state"] in {"ONLINE", "MISSING"}:
                continue
            ok = mount_container(vol.container, password, vol.letter)
            if ok:
                try:
                    self._ensure_skeleton(vol)
                except OSError as exc:
                    results.append({"volume_id": vol.id, "letter": vol.letter,
                                    "mounted": True, "skeleton_error": str(exc)})
                    continue
            results.append({"volume_id": vol.id, "letter": vol.letter, "mounted": ok})
        return results

    # ---- registry ----------------------------------------------------
    def volumes(self) -> list[SandboxVolume]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM sandbox_volumes ORDER BY is_primary DESC, priority DESC, id").fetchall()
        return [SandboxVolume(dict(r)) for r in rows]

    def get(self, volume_id: str) -> SandboxVolume | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM sandbox_volumes WHERE id=?", (volume_id,)).fetchone()
        return SandboxVolume(dict(row)) if row else None

    def primary(self) -> SandboxVolume:
        for vol in self.volumes():
            if vol.primary:
                return vol
        raise RuntimeError("卷池里没有主卷 —— state/ 无法定位")

    def add(self, *, root: str, container: str = "", letter: str = "", priority: int = 0,
            label: str = "", actor: str = "", now: str = "") -> SandboxVolume:
        root_path = Path(root)
        if not root_path.is_absolute():
            raise ValueError("卷根必须是绝对路径")
        # 先建骨架再登记：目录建不出来（卷没挂上 / 不可写）就不该进池，
        # 否则会留下一个"能选到但写不进"的坏卷。
        vol_stub = SandboxVolume({"id": "pending", "root": str(root_path),
                                  "container": container, "letter": letter,
                                  "priority": priority, "enabled": True,
                                  "primary": False, "label": label})
        self._ensure_skeleton(vol_stub)
        with self._lock:
            existing = self.db.execute("SELECT id FROM sandbox_volumes WHERE root=?", (str(root_path),)).fetchone()
            if existing:
                raise ValueError(f"该路径已在池中: {existing['id']}")
            volume_id = "vol_" + os.urandom(6).hex()
            self.db.execute(
                "INSERT INTO sandbox_volumes(id,root,container,letter,priority,enabled,is_primary,label,created_at,created_by)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (volume_id, str(root_path), container, letter, int(priority), 1, 0,
                 label, now, actor),
            )
            self.db.commit()
        return self.get(volume_id)  # type: ignore[return-value]

    def remove(self, volume_id: str) -> None:
        vol = self.get(volume_id)
        if vol is None:
            raise KeyError(volume_id)
        if vol.primary:
            raise ValueError("主卷不能从池中移除：state/ 与哈希链在上面")
        with self._lock:
            self.db.execute("DELETE FROM sandbox_volumes WHERE id=?", (volume_id,))
            self.db.commit()

    def set_enabled(self, volume_id: str, enabled: bool) -> SandboxVolume:
        with self._lock:
            self.db.execute("UPDATE sandbox_volumes SET enabled=? WHERE id=?",
                            (1 if enabled else 0, volume_id))
            self.db.commit()
        vol = self.get(volume_id)
        if vol is None:
            raise KeyError(volume_id)
        return vol

    def set_priority(self, volume_id: str, priority: int) -> SandboxVolume:
        with self._lock:
            self.db.execute("UPDATE sandbox_volumes SET priority=? WHERE id=?",
                            (int(priority), volume_id))
            self.db.commit()
        vol = self.get(volume_id)
        if vol is None:
            raise KeyError(volume_id)
        return vol

    # ---- skeleton ----------------------------------------------------
    def _ensure_skeleton(self, vol: SandboxVolume) -> None:
        """在新卷上建平台目录骨架。state/audit 只在主卷，避免链分裂。

        必须在**挂载成功之后**调用，且失败要抛出来 —— 之前静默 `pass` 掉，
        结果卷登记进池了、盘也挂着，但 `W:\\geo_data_security` 根本不存在，
        选卷时通过、写文件时才炸（实测踩过）。
        """
        if not vol.root.parent.exists():
            raise RuntimeError(f"卷未挂载或不可访问: {vol.root.parent}")
        for area in DATA_AREAS:
            (vol.root / area).mkdir(parents=True, exist_ok=True)
        probe = vol.root / ".gds_skeleton_probe"
        probe.write_bytes(b"1")
        probe.unlink()

    # ---- probe -------------------------------------------------------
    def volume_state(self, vol: SandboxVolume) -> dict[str, Any]:
        """卷的实时状态：挂载与否、容量、剩余、是否可写。全部当场实测。"""
        info: dict[str, Any] = {**vol.to_dict(), "state": "OFFLINE", "mounted": False,
                                "capacity_bytes": 0, "used_bytes": 0, "free_bytes": 0,
                                "writable": False}
        # 容器文件不见了（被手工删掉 / 磁盘上被移走）→ 明确报出来，
        # 而不是显示成一个永远 OFFLINE 的僵尸卷让人猜。
        if vol.container and not Path(vol.container).exists():
            info["state"] = "MISSING"
            info["error"] = f"容器文件不存在: {vol.container}"
            return info
        if not vol.root.drive:
            info["state"] = "OFFLINE"
            info["error"] = "卷根没有盘符，无法探测容量"
            return info
        drive_root = Path(vol.root.drive + "\\")
        try:
            present = drive_root.exists()
        except OSError as exc:
            # 挂着坏文件系统的卷：明确报 READONLY/坏，而不是让接口 500
            info["state"] = "BAD_FS"
            info["error"] = f"{vol.root.drive} 无法访问（可能文件系统损坏）: {exc}"
            return info
        if not present:
            info["error"] = f"{vol.root.drive} 未挂载"
            return info
        info["mounted"] = True
        try:
            usage = shutil.disk_usage(str(drive_root))
        except OSError as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
            return info
        info.update({"capacity_bytes": usage.total, "used_bytes": usage.used,
                     "free_bytes": usage.free, "state": "ONLINE"})
        probe = vol.root if vol.root.exists() else drive_root
        try:
            marker = probe / ".gds_write_probe"
            marker.write_bytes(b"1")
            marker.unlink()
            info["writable"] = True
        except OSError as exc:
            info["error"] = f"不可写: {type(exc).__name__}: {exc}"
            info["state"] = "READONLY"
        return info

    def status(self) -> dict[str, Any]:
        vols = [self.volume_state(v) for v in self.volumes()]
        online = [v for v in vols if v["state"] == "ONLINE" and v["writable"]]
        return {
            "volumes": vols,
            "count": len(vols),
            "online_count": len(online),
            "total_capacity_bytes": sum(v["capacity_bytes"] for v in online),
            "total_free_bytes": sum(v["free_bytes"] for v in online),
            "primary": str(self.primary_root),
        }

    # ---- selection ---------------------------------------------------
    def select_volume(self, need_bytes: int, *, prefer: str | None = None) -> dict[str, Any]:
        """按策略选一个能放下 `need_bytes` 的在线卷。

        `prefer` 指定卷 id 时优先用它（调用方显式要求），但容量不足仍会拒绝——
        不能因为"点名要这个卷"就让写到一半失败。

        排序规则（重要）：
        1. 先按 `priority` 降序 —— 管理员可以用它做硬性偏置（比如"新数据一律进 W:"）。
        2. 同优先级内按**剩余空间降序** —— 这样空间大的卷会自然多接数据，
           不会出现"200GB 主卷 priority=100 把 3GB 新卷彻底饿死"的情况。
           如果只按 priority 排，加多少卷都没用，主卷永远赢。
        """
        candidates: list[dict[str, Any]] = []
        for vol in self.volumes():
            if not vol.enabled:
                continue
            state = self.volume_state(vol)
            candidates.append(state)
        usable = [v for v in candidates if v["state"] == "ONLINE" and v["writable"]]
        if prefer:
            picked = [v for v in usable if v["id"] == prefer]
            if not picked:
                raise NoCapacityError(f"指定卷 {prefer} 不可用", candidates)
            v = picked[0]
            if v["free_bytes"] < need_bytes + MIN_FREE_AFTER_WRITE:
                raise NoCapacityError(
                    f"指定卷 {prefer} 剩余不足：需 {need_bytes + MIN_FREE_AFTER_WRITE} 字节", candidates)
            return v
        fit = [v for v in usable if v["free_bytes"] >= need_bytes + MIN_FREE_AFTER_WRITE]
        if not fit:
            raise NoCapacityError(
                f"没有卷能放下 {need_bytes} 字节（已预留 {MIN_FREE_AFTER_WRITE} 字节余量）", candidates)
        fit.sort(key=lambda v: (v["priority"], v["free_bytes"]), reverse=True)
        return fit[0]

    def area_path(self, volume_id: str, *parts: str) -> Path:
        vol = self.get(volume_id)
        if vol is None:
            raise KeyError(volume_id)
        return vol.root.joinpath(*parts)

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VeraCrypt 容器管理（全部 CLI，无需 GUI —— 这是"加卷"可行的关键）
# ---------------------------------------------------------------------------

def _drive_present(letter: str) -> bool:
    """盘符是否被占用。**必须吞掉 OSError**：挂着一个坏文件系统的卷时，
    `Path('E:\\\\').exists()` 会抛 WinError 1005（"此卷不包含可识别的文件系统"），
    而不是返回 False —— 那会让整个建卷接口 500。"""
    try:
        return Path(f"{letter}:\\").exists()
    except OSError:
        return True  # 存在但读不了 → 当作已占用，别往上分配


def free_drive_letters() -> list[str]:
    used = set()
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if _drive_present(letter):
            used.add(letter)
    return [c for c in "DEFGHIJKLMNOPQRSTUVWXYZ" if c not in used]


def create_container(container_path: str, size_gb: int, password: str,
                     letter: str, *, encryption: str = "aes",
                     hash_alg: str = "sha512", timeout_s: int = 7200) -> dict[str, Any]:
    """建 VeraCrypt 容器并挂载。**全 CLI**，不需要点 GUI。

    实测参数格式（1.26.29）：
      - `/create <路径>` —— 路径直接跟在后面，不能写成 `/volume=`
      - `/size 500G`     —— 空格分开，不能写成 `/size=500G`
      - 口令必须 ≥ 20 字符，否则弹弱口令确认框，`/silent` 绕不过

    **关键坑**：`VeraCrypt Format.exe` 是无静默模式的 GUI 程序。容器写完后它不退出，
    而是弹一个"卷已创建"窗口等人点确定，`subprocess.run(...)` 会一直阻塞到超时。

    完成判据只能是**能挂上**：
      - 不能等进程退出（它永不退出）；
      - 不能看文件大小达标（VeraCrypt 几秒就把文件扩展到目标大小，NTFS 格式化仍在跑，
        此时杀进程会留下**挂不上**的坏容器，已实测复现）。
    所以这里周期性地试挂载，成功即完成。
    """
    if size_gb < 1:
        raise ValueError("容器至少 1 GB")
    if len(password) < 20:
        raise ValueError("VeraCrypt 口令必须 ≥ 20 字符（短口令会弹出确认框，无 CLI 绕过）")
    # 测试护栏：GDS_VERACRYPT_DISABLE=1 时直接拒绝，绝不在宿主机上真建卷。
    # （早期测试只把目录指到不存在的路径，但 create_container 会自己 mkdir，
    #  结果测试真的建出了容器、还挂上了 E:/F:/G: —— 所以要有显式开关。）
    if os.environ.get("GDS_VERACRYPT_DISABLE") == "1":
        raise RuntimeError("VeraCrypt disabled by GDS_VERACRYPT_DISABLE=1 (test guard)")
    target = Path(container_path)
    if target.exists():
        raise ValueError(f"容器文件已存在: {container_path}")
    target.parent.mkdir(parents=True, exist_ok=True)

    expected = size_gb * 1024 ** 3
    create_cmd = [VERACRYPT_FORMAT, "/create", str(target), "/size", f"{size_gb}G",
                  "/password", password, "/hash", hash_alg, "/encryption", encryption,
                  "/filesystem", "NTFS", "/quick", "/force"]
    proc = subprocess.Popen(create_cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, text=True)
    import time as _t
    deadline = _t.time() + timeout_s
    mounted = False
    try:
        # 完成信号：**能挂上**。不能靠"文件大小达标"——VeraCrypt 会先把文件扩展到
        # 目标大小（几秒），NTFS 格式化仍在跑；此时杀进程会留下挂不上的坏容器（已实测复现）。
        # Format.exe 是 GUI 程序，写完后弹窗不退出，所以也不能等它退出。
        # 唯一可靠判据是周期性地试挂载：成功即格式化完成。
        probe_letter = letter
        while _t.time() < deadline:
            _t.sleep(5)
            if Path(f"{probe_letter}:\\").exists():
                mounted = True
                break
            if not target.exists() or target.stat().st_size < expected:
                continue  # 还在扩展文件，别急着试挂
            subprocess.run([VERACRYPT, "/volume", str(target), "/letter", probe_letter,
                            "/password", password, "/quit", "/silent"],
                           capture_output=True, text=True, timeout=180,
                           errors="backslashreplace")
            _t.sleep(2)
            if Path(f"{probe_letter}:\\").exists():
                mounted = True
                break
        if not mounted:
            raise RuntimeError(f"建卷超时（{timeout_s}s）：容器始终无法挂载")
    finally:
        # Format.exe 建完会挂 GUI 窗口不退出，必须收掉，否则进程越积越多。
        # 此时容器已可挂载（上面已确认），杀掉不会损坏。
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError("容器创建失败：文件不存在或为空")

    mount_cmd = [VERACRYPT, "/volume", str(target), "/letter", letter,
                 "/password", password, "/quit", "/silent"]
    subprocess.run(mount_cmd, capture_output=True, text=True, timeout=180, errors="backslashreplace")
    for _ in range(90):
        if Path(f"{letter}:\\").exists():
            break
        _t.sleep(1)
    if not Path(f"{letter}:\\").exists():
        raise RuntimeError(f"容器已创建但挂载 {letter}: 失败")
    return {"container": str(target), "letter": letter, "size_bytes": target.stat().st_size}


def mount_container(container_path: str, password: str, letter: str, timeout_s: int = 180) -> bool:
    if not Path(container_path).exists():
        raise ValueError(f"容器不存在: {container_path}")
    subprocess.run([VERACRYPT, "/volume", container_path, "/letter", letter,
                    "/password", password, "/quit", "/silent"],
                   capture_output=True, text=True, timeout=timeout_s, errors="backslashreplace")
    import time as _t
    for _ in range(60):
        if Path(f"{letter}:\\").exists():
            return True
        _t.sleep(1)
    return False


def unmount_letter(letter: str, timeout_s: int = 120) -> bool:
    subprocess.run([VERACRYPT, "/unmount", letter, "/quit", "/silent"],
                   capture_output=True, text=True, timeout=timeout_s, errors="backslashreplace")
    import time as _t
    for _ in range(30):
        if not Path(f"{letter}:\\").exists():
            return True
        _t.sleep(1)
    return False
