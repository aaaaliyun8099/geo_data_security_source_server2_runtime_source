"""SQLite-backed task and artifact state for server2.

The database stores workflow metadata only. Large source/result files remain in the
configured workspace and are referenced by absolute paths in the artifact record.
SQLite is the local/default backend; a future deployment can replace this boundary
with PostgreSQL without changing the API layer.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .processing import ProcessingArtifact
from .timefmt import local_time, row_dict  # noqa: F401  (re-exported for existing callers)


class TaskStore:
    """Durable task/artifact repository with atomic status updates."""

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    result_id TEXT NOT NULL DEFAULT '',
                    error TEXT,
                    classification TEXT NOT NULL DEFAULT '',
                    watermark_enabled INTEGER NOT NULL DEFAULT 1,
                    anchor_granularity TEXT NOT NULL DEFAULT 'UPLOAD',
                    stage TEXT NOT NULL DEFAULT 'QUEUED',
                    progress_percent INTEGER NOT NULL DEFAULT 0,
                    stage_detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    output TEXT NOT NULL,
                    files_processed INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
                CREATE INDEX IF NOT EXISTS idx_artifacts_project_user ON artifacts(project_id, user_id);
                CREATE TABLE IF NOT EXISTS external_file_ids (
                    external_id TEXT PRIMARY KEY,
                    artifact_id TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'server1',
                    note TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_external_artifact ON external_file_ids(artifact_id);
                CREATE TABLE IF NOT EXISTS upload_sessions (
                    upload_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, user_id TEXT NOT NULL,
                    data_type TEXT NOT NULL, classification TEXT NOT NULL, filename TEXT NOT NULL,
                    expected_bytes INTEGER NOT NULL, expected_sha256 TEXT NOT NULL, chunk_size INTEGER NOT NULL,
                    received_bytes INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, source_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_upload_project ON upload_sessions(project_id, user_id);
                CREATE TABLE IF NOT EXISTS upload_chunks (
                    upload_id TEXT NOT NULL,
                    chunk_no INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    PRIMARY KEY (upload_id, chunk_no),
                    FOREIGN KEY (upload_id) REFERENCES upload_sessions(upload_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS export_jobs (
                    export_id TEXT PRIMARY KEY, result_id TEXT NOT NULL, project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL, status TEXT NOT NULL, archive_path TEXT NOT NULL DEFAULT '',
                    credential_value TEXT NOT NULL DEFAULT '', trace_record_hash TEXT NOT NULL DEFAULT '',
                    anchor_json TEXT NOT NULL DEFAULT '', error TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_export_result ON export_jobs(result_id);
                """
            )
            columns = {row[1] for row in self._connection.execute("PRAGMA table_info(tasks)")}
            if "classification" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN classification TEXT NOT NULL DEFAULT ''")
            if "watermark_enabled" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN watermark_enabled INTEGER NOT NULL DEFAULT 1")
            if "anchor_granularity" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN anchor_granularity TEXT NOT NULL DEFAULT 'UPLOAD'")
            if "source_integrity" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN source_integrity TEXT")
            if "warnings" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN warnings TEXT NOT NULL DEFAULT '[]'")
            if "stage" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN stage TEXT NOT NULL DEFAULT 'QUEUED'")
            if "progress_percent" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN progress_percent INTEGER NOT NULL DEFAULT 0")
            if "stage_detail" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN stage_detail TEXT NOT NULL DEFAULT ''")
            # 一键重提需要原始入参：没有 upload_id 和数据类型就无法在重启中断后重放同一个任务。
            if "upload_id" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN upload_id TEXT NOT NULL DEFAULT ''")
            if "data_type" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN data_type TEXT NOT NULL DEFAULT ''")
            if "resume_payload" not in columns:
                self._connection.execute("ALTER TABLE tasks ADD COLUMN resume_payload TEXT NOT NULL DEFAULT ''")
            columns = {row[1] for row in self._connection.execute("PRAGMA table_info(export_jobs)")}
            for column, definition in {
                "credential_value": "TEXT NOT NULL DEFAULT ''",
                "trace_record_hash": "TEXT NOT NULL DEFAULT ''",
                "anchor_json": "TEXT NOT NULL DEFAULT ''",
            }.items():
                if column not in columns:
                    self._connection.execute(f"ALTER TABLE export_jobs ADD COLUMN {column} {definition}")

    @staticmethod
    def artifact_to_dict(artifact: ProcessingArtifact) -> dict[str, Any]:
        return asdict(artifact)

    def save_task(self, task: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO tasks(task_id,status,project_id,user_id,result_id,error,classification,watermark_enabled,anchor_granularity,stage,progress_percent,stage_detail,upload_id,data_type,resume_payload)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(task_id) DO UPDATE SET status=excluded.status,
                     project_id=excluded.project_id,user_id=excluded.user_id,
                     result_id=excluded.result_id,error=excluded.error,classification=excluded.classification,
                     watermark_enabled=excluded.watermark_enabled,anchor_granularity=excluded.anchor_granularity,
                     stage=excluded.stage,progress_percent=excluded.progress_percent,stage_detail=excluded.stage_detail,
                     upload_id=excluded.upload_id,data_type=excluded.data_type,resume_payload=excluded.resume_payload,
                     updated_at=CURRENT_TIMESTAMP""",
                (task["task_id"], task["status"], task.get("project_id", ""),
                 task.get("user_id", ""), task.get("result_id", ""), task.get("error"), task.get("classification", ""),
                 int(task.get("watermark_enabled", True)), task.get("anchor_granularity", "UPLOAD"),
                 task.get("stage", "QUEUED"), int(task.get("progress_percent", 0)), task.get("stage_detail", ""),
                 task.get("upload_id", ""), task.get("data_type", ""), task.get("resume_payload", "")),
            )

    def set_task_progress(self, task_id: str, *, stage: str, percent: int, detail: str = "") -> None:
        """记录任务阶段进度，供前端工作台做进度提示与监测展示。

        阶段与百分比由处理管线在关键节点上报；同一 task 的更新是幂等的 last-write-wins。
        """
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE tasks SET stage=?, progress_percent=?, stage_detail=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE task_id=?",
                (stage, max(0, min(100, int(percent))), detail, task_id),
            )

    def list_tasks(self, *, project_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        """按项目/状态列出任务（列表形式），供工作台监测面板轮询。"""
        query = "SELECT * FROM tasks"
        clauses, params = [], []
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC"
        with self._lock:
            cursor = self._connection.execute(query, params)
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def save_artifact(self, artifact: ProcessingArtifact, *, project_id: str, user_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO artifacts(artifact_id,data_type,source,output,files_processed,
                   status,project_id,user_id)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(artifact_id) DO UPDATE SET status=excluded.status,
                     output=excluded.output,files_processed=excluded.files_processed,
                     updated_at=CURRENT_TIMESTAMP""",
                (artifact.artifact_id, artifact.data_type, artifact.source, artifact.output,
                 artifact.files_processed, artifact.status, project_id, user_id),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return row_dict(row) if row else None

    def update_task_status(self, task_id: str, status: str, *, result_id: str = "", error: str | None = None) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE tasks SET status=?, result_id=?, error=?, updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                (status, result_id, error, task_id),
            )

    def fail_stale_tasks(self, reason: str) -> list[dict[str, Any]]:
        """把遗留的 QUEUED / RUNNING 任务标记为 INTERRUPTED，返回被改动的任务信息。

        任务执行依赖进程内的 ThreadPoolExecutor，进程退出后线程即消失，
        但数据库行仍是非终态。若不在启动时收尾，前端会永久显示卡在某个阶段的进度条，
        用户无从判断是「还在跑」还是「已经死了」。

        这里刻意**不**标成 FAILED：中断不是任务本身的失败，用户的数据和意图都还在，
        标 FAILED 会让前端把「可重提」误当成「白做了」。改为 INTERRUPTED，
        并把原始入参（resume_payload）留在行里，前端可调 resume 接口原样重放。
        """
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT task_id, stage, progress_percent, upload_id, data_type, resume_payload "
                "FROM tasks WHERE status IN ('QUEUED', 'RUNNING')"
            ).fetchall()
            if not rows:
                return []
            changed: list[dict[str, Any]] = []
            for task_id, stage, percent, upload_id, data_type, resume_payload in rows:
                detail = reason
                if stage:
                    detail = f"{reason}（中断于阶段 {stage}，进度 {percent}%）"
                self._connection.execute(
                    "UPDATE tasks SET status='INTERRUPTED', error=?, updated_at=CURRENT_TIMESTAMP "
                    "WHERE task_id=?",
                    (detail, task_id),
                )
                changed.append({
                    "task_id": task_id,
                    "stage": stage,
                    "progress_percent": percent,
                    "upload_id": upload_id or "",
                    "data_type": data_type or "",
                    "resumable": bool((upload_id or "").strip() and (resume_payload or "").strip()),
                })
            return changed

    def set_task_integrity(self, task_id: str, source_integrity: str | None, warnings: Any = None) -> None:
        """Record the source-integrity verdict (and its warning) on the task row.

        The verdict is decided by the pre-flight scan, which runs before the long watermark job, so
        the frontend can show \"source incomplete, partial result expected\" while the task is still in
        flight instead of only after the manifest exists.
        """
        payload = json.dumps(list(warnings or []), ensure_ascii=False)
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE tasks SET source_integrity=?, warnings=?, updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                (source_integrity, payload, task_id),
            )

    def link_external_file_id(self, external_id: str, artifact_id: str,
                              source: str = "server1", note: str | None = None) -> dict[str, Any]:
        """登记「外部系统编号 → 本系统 artifact」的映射。

        服务器1 有自己的 fileId（如 file_1792552013127），本系统不认识。
        允许对方把自己那侧的编号登记过来，之后就能直接用该编号调用沙箱接口。
        """
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO external_file_ids (external_id, artifact_id, source, note)"
                " VALUES (?,?,?,?)", (external_id, artifact_id, source, note))
        return {"external_id": external_id, "artifact_id": artifact_id,
                "source": source, "note": note}

    def resolve_external_file_id(self, external_id: str) -> str | None:
        """把外部编号解析回本系统 artifact_id；未登记返回 None。"""
        with self._lock:
            row = self._connection.execute(
                "SELECT artifact_id FROM external_file_ids WHERE external_id=?",
                (external_id,)).fetchone()
        return row[0] if row else None

    def list_external_file_ids(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT external_id, artifact_id, source, note, created_at"
                " FROM external_file_ids ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [row_dict(r) for r in rows]

    def get_artifact_record(self, artifact_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        if not row:
            return None
        result = row_dict(row)
        result["artifact"] = ProcessingArtifact(
            result["artifact_id"], result["data_type"], result["source"], result["output"],
            result["files_processed"], result["status"],
        )
        return result

    def set_artifact_status(self, artifact_id: str, status: str) -> ProcessingArtifact | None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE artifacts SET status=?, updated_at=CURRENT_TIMESTAMP WHERE artifact_id=?",
                (status, artifact_id),
            )
        record = self.get_artifact_record(artifact_id)
        return record["artifact"] if record else None

    def load_tasks(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM tasks").fetchall()
        return {row["task_id"]: row_dict(row) for row in rows}

    def load_artifacts(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM artifacts").fetchall()
        result = {}
        for row in rows:
            item = row_dict(row)
            item["artifact"] = ProcessingArtifact(
                item["artifact_id"], item["data_type"], item["source"], item["output"],
                item["files_processed"], item["status"],
            )
            result[item["artifact_id"]] = item
        return result

    def save_upload(self, upload: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute("INSERT INTO upload_sessions(upload_id,project_id,user_id,data_type,classification,filename,expected_bytes,expected_sha256,chunk_size,received_bytes,status,source_path) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (upload["upload_id"], upload["project_id"], upload["user_id"], upload["data_type"], upload["classification"], upload["filename"], upload["expected_bytes"], upload["expected_sha256"], upload["chunk_size"], upload.get("received_bytes", 0), upload.get("status", "CREATED"), upload.get("source_path", "")))

    def get_upload(self, upload_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM upload_sessions WHERE upload_id=?", (upload_id,)).fetchone()
        return row_dict(row) if row else None

    def update_upload(self, upload_id: str, **fields: Any) -> None:
        fields = {k: v for k, v in fields.items() if k in {"received_bytes", "status", "source_path"}}
        if not fields: return
        assignment = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._connection:
            self._connection.execute(f"UPDATE upload_sessions SET {assignment}, updated_at=CURRENT_TIMESTAMP WHERE upload_id=?", (*fields.values(), upload_id))

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def record_upload_chunk(self, upload_id: str, chunk_no: int, size_bytes: int,
                            expected_bytes: int | None = None) -> int:
        """记一片并返回**累计已收字节**，插入/汇总/回写会话走同一个事务。

        旧实现每片要做两次提交：`record_upload_chunk` 一次、`update_upload` 再一次。
        SQLite 在 WAL 下每次提交都要 fsync，而库在 VeraCrypt 加密卷上，一次提交就是
        一次加密卷落盘。分片多的时候（64KB 切 1GB = 16384 片）这两次 fsync 就是纯浪费。
        合并后每片只剩一次提交；同时把 Python 侧的 O(n) 求和改成 SQL 侧 SUM。
        """
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO upload_chunks(upload_id, chunk_no, size_bytes) VALUES(?,?,?) "
                "ON CONFLICT(upload_id, chunk_no) DO UPDATE SET size_bytes=excluded.size_bytes",
                (upload_id, chunk_no, size_bytes),
            )
            row = self._connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM upload_chunks WHERE upload_id=?",
                (upload_id,),
            ).fetchone()
            received = int(row[0]) if row else 0
            if expected_bytes is not None:
                received = min(int(expected_bytes), received)
            self._connection.execute(
                "UPDATE upload_sessions SET received_bytes=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE upload_id=?", (received, upload_id),
            )
        return received

    def upload_chunk_numbers(self, upload_id: str) -> list[int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT chunk_no FROM upload_chunks WHERE upload_id=? ORDER BY chunk_no", (upload_id,)
            ).fetchall()
        return [int(row[0]) for row in rows]

    def upload_received_bytes(self, upload_id: str) -> int:
        """已收字节合计（SQL 侧 SUM，供只关心总数的调用方使用）。"""
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM upload_chunks WHERE upload_id=?", (upload_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def upload_chunk_sizes(self, upload_id: str) -> dict[int, int]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT chunk_no, size_bytes FROM upload_chunks WHERE upload_id=? ORDER BY chunk_no", (upload_id,)
            ).fetchall()
        return {int(row[0]): int(row[1]) for row in rows}

    def clear_upload_chunks(self, upload_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM upload_chunks WHERE upload_id=?", (upload_id,))

    def finalize_upload(self, upload_id: str, expected_bytes: int, chunk_size: int,
                        digest_hex: str, status: str = "IMPORTED") -> dict[str, Any] | None:
        """校验分片边界 + 落最终状态，走同一个事务。

        complete 阶段原来是 3 次以上独立提交（多次 FAILED 分支各提交一次、
        最后 IMPORTED 再一次），每次都要 fsync 加密卷。这里合成一次，
        并且把「分片齐全 + 大小正确」的判断也放进同一把锁里，避免并发 complete 互相踩。
        """
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT chunk_no, size_bytes FROM upload_chunks WHERE upload_id=? ORDER BY chunk_no",
                (upload_id,),
            ).fetchall()
            sizes = {int(r[0]): int(r[1]) for r in rows}
            expected_chunks = (expected_bytes + chunk_size - 1) // chunk_size
            if set(sizes) != set(range(expected_chunks)):
                return {"error": "upload has missing chunks", "status": "FAILED"}
            for chunk_no, size in sizes.items():
                want = min(chunk_size, expected_bytes - chunk_no * chunk_size)
                if size != want:
                    return {"error": "upload has invalid chunk size", "status": "FAILED"}
            self._connection.execute(
                "UPDATE upload_sessions SET status=?, received_bytes=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE upload_id=?", (status, expected_bytes, upload_id),
            )
        return None

    def mark_upload_failed(self, upload_id: str, status: str = "FAILED") -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE upload_sessions SET status=?, updated_at=CURRENT_TIMESTAMP WHERE upload_id=?",
                (status, upload_id),
            )

    def save_export_job(self, item: dict[str, Any]) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO export_jobs(export_id,result_id,project_id,user_id,status,archive_path,credential_value,trace_record_hash,anchor_json,error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (item["export_id"], item["result_id"], item["project_id"], item["user_id"], item["status"], item.get("archive_path", ""), item.get("credential_value", ""), item.get("trace_record_hash", ""), item.get("anchor_json", ""), item.get("error")),
            )

    def get_export_job(self, export_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM export_jobs WHERE export_id=?", (export_id,)).fetchone()
        return row_dict(row) if row else None

    def update_export_job(self, export_id: str, **fields: Any) -> None:
        allowed = {"status", "archive_path", "credential_value", "trace_record_hash", "anchor_json", "error"}
        fields = {k: v for k, v in fields.items() if k in allowed}
        if not fields:
            return
        assignment = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._connection:
            self._connection.execute(f"UPDATE export_jobs SET {assignment}, updated_at=CURRENT_TIMESTAMP WHERE export_id=?", (*fields.values(), export_id))

    def list_rows(self, table: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        allowed = {"tasks", "artifacts", "upload_sessions", "export_jobs"}
        if table not in allowed:
            raise ValueError("unsupported admin table")
        limit = max(1, min(int(limit), 1000)); offset = max(0, int(offset))
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        return [row_dict(row) for row in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {table: int(self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in ("tasks", "artifacts", "upload_sessions", "export_jobs")}

    def count_by_status(self, table: str, statuses: tuple[str, ...]) -> int:
        """统计某表处于指定状态的行数。清理前置检查用它判断"是否有活任务"。"""
        if table not in {"tasks", "artifacts", "upload_sessions", "export_jobs"}:
            raise ValueError("unsupported admin table")
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            row = self._connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE status IN ({placeholders})", tuple(statuses)).fetchone()
        return int(row[0] or 0)
