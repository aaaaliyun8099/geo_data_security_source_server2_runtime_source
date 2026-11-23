"""数据外借 / 归还 / 超期 与归还核验（provenance check-in）。

业务场景
--------
数据被"借出"到系统之外（拷给别人、离线带走），归还时管理员需要回答三个问题：

1. 这份数据**是不是本系统产出的**？（靠水印现场检出 + 结果库比对）
2. 它在本系统里**有没有真实的借出流转记录**？（借出单 + 链上事件）
3. 有没有**超过归还期限**？（期限字段 + 超期判定）

设计立场（与本项目既有约定一致）
--------------------------------
* **不伪造**：水印检不出、库里无记录时，返回 ``is_ours=false`` 并给出原因，绝不"看起来像就放过"。
* **可回滚**：本模块只读写自己的 ``data_loans`` 表与链事件，不动沙箱、不动水印计算逻辑。
* **链上留痕**：借出 / 归还 / 超期 / 归还核验都写链事件，可被 ``/internal/trace/verify`` 验证。
* **诚实分级**：核验结论分 ``RETURNED_OK`` / ``RETURNED_MISMATCH`` / ``RETURNED_NOT_OURS`` /
  ``RETURNED_OVERDUE`` 四档，超期与"不是我们的"同时成立时两者都报。

校验顺序（重要，避免误判）
--------------------------
先判**是不是本系统产出**（水印/记录），再判**期限**。理由：一份不是我们借出的数据
即使"逾期"也没有业务意义，先说清身份再说期限，避免把外来数据标成"超期归还"。
"""
from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .timefmt import now_local

# 借出单状态
STATUS_LENT = "LENT"            # 已借出，未到/已过期限，未归还
STATUS_RETURNED = "RETURNED"    # 已归还（含超期归还）
STATUS_OVERDUE = "OVERDUE"      # 已过期限且未归还
STATUS_CANCELLED = "CANCELLED"  # 作废

# 归还核验结论
VERDICT_OK = "RETURNED_OK"                  # 身份核对通过，未超期
VERDICT_OVERDUE = "RETURNED_OVERDUE"        # 身份核对通过，但已超期
VERDICT_MISMATCH = "RETURNED_MISMATCH"      # 是本系统数据，但不是这张借出单对应的那份
VERDICT_NOT_OURS = "RETURNED_NOT_OURS"      # 水印检不出且无记录 → 不是本系统数据
VERDICT_UNKNOWN = "RETURNED_UNVERIFIABLE"   # 水印检出但无法归位到具体结果


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class LoanStore:
    """借出单的持久化存储（独立 SQLite 文件，与任务库解耦，便于单独备份/回滚）。"""

    def __init__(self, path: Path | str):
        import sqlite3

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS data_loans (
                    loan_id TEXT PRIMARY KEY,
                    result_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    classification TEXT NOT NULL DEFAULT '',
                    data_type TEXT NOT NULL DEFAULT '',
                    borrower TEXT NOT NULL,
                    purpose TEXT NOT NULL DEFAULT '',
                    operator TEXT NOT NULL DEFAULT '',
                    lent_at TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    returned_at TEXT,
                    status TEXT NOT NULL,
                    expected_watermark TEXT NOT NULL DEFAULT '',
                    return_verdict TEXT NOT NULL DEFAULT '',
                    return_detail TEXT NOT NULL DEFAULT '[]',
                    carrier_hash TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_loans_result ON data_loans(result_id);
                CREATE INDEX IF NOT EXISTS idx_loans_status ON data_loans(status);
                CREATE INDEX IF NOT EXISTS idx_loans_borrower ON data_loans(borrower);
                """
            )

    # ── 内部工具 ────────────────────────────────────────────────────────
    @staticmethod
    def _row(row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["return_detail"] = json.loads(item.get("return_detail") or "[]")
        except (TypeError, ValueError):
            item["return_detail"] = []
        return item

    def _touch(self, loan_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock, self._connection:
            self._connection.execute(
                f"UPDATE data_loans SET {sets}, updated_at=? WHERE loan_id=?",
                (*fields.values(), now_local(), loan_id),
            )

    # ── 借出 ────────────────────────────────────────────────────────────
    def create(
        self,
        *,
        result_id: str,
        project_id: str,
        borrower: str,
        due_at: str,
        classification: str = "",
        data_type: str = "",
        purpose: str = "",
        operator: str = "",
        expected_watermark: str = "",
        carrier_hash: str = "",
    ) -> dict[str, Any]:
        loan_id = _new_id("loan")
        lent_at = now_local()
        # 同一结果不允许重复挂未归还的借出单：否则"有没有借出记录"这个问题就没有唯一答案。
        existing = self.open_loan_for_result(result_id)
        if existing:
            raise ValueError(
                f"result {result_id} already has an open loan {existing['loan_id']} "
                f"(due {existing['due_at']}); check it in before lending again"
            )
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT INTO data_loans
                   (loan_id, result_id, project_id, classification, data_type, borrower, purpose,
                    operator, lent_at, due_at, returned_at, status, expected_watermark, carrier_hash,
                    created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (loan_id, result_id, project_id, classification, data_type, borrower, purpose,
                 operator, lent_at, due_at, None, STATUS_LENT, expected_watermark, carrier_hash,
                 lent_at, lent_at),
            )
        return self.get(loan_id)

    # ── 查询 ────────────────────────────────────────────────────────────
    def get(self, loan_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM data_loans WHERE loan_id=?", (loan_id,)
            ).fetchone()
        if row is None:
            raise KeyError(loan_id)
        return self._row(row)

    def open_loan_for_result(self, result_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM data_loans WHERE result_id=? AND status IN (?,?) "
                "ORDER BY lent_at DESC LIMIT 1",
                (result_id, STATUS_LENT, STATUS_OVERDUE),
            ).fetchone()
        return self._row(row) if row else None

    def list(self, *, status: str | None = None, result_id: str | None = None,
             project_id: str | None = None, borrower: str | None = None,
             include_closed: bool = True) -> list[dict[str, Any]]:
        clauses, params = [], []
        if status:
            clauses.append("status=?"); params.append(status)
        if result_id:
            clauses.append("result_id=?"); params.append(result_id)
        if project_id:
            clauses.append("project_id=?"); params.append(project_id)
        if borrower:
            clauses.append("borrower=?"); params.append(borrower)
        if not include_closed and not status:
            clauses.append("status IN (?,?)")
            params.extend([STATUS_LENT, STATUS_OVERDUE])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM data_loans {where} ORDER BY lent_at DESC", tuple(params)
            ).fetchall()
        return [self._row(r) for r in rows]

    # ── 超期判定 ────────────────────────────────────────────────────────
    @staticmethod
    def _due_passed(due_at: str, *, at: datetime | None = None) -> tuple[bool, int]:
        """返回 (是否已过期, 逾期天数)。逾期天数用自然日，不足一天算 1 天。"""
        try:
            due = datetime.fromisoformat(due_at)
        except (TypeError, ValueError):
            return False, 0
        reference = at or datetime.now(due.tzinfo) if due.tzinfo else (at or datetime.now())
        if due.tzinfo and reference.tzinfo is None:
            reference = reference.replace(tzinfo=due.tzinfo)
        if reference <= due:
            return False, 0
        delta = reference - due
        return True, max(1, delta.days + (1 if delta.seconds or delta.microseconds else 0))

    def refresh_overdue(self) -> list[dict[str, Any]]:
        """把已过期且未归还的借出单标为 OVERDUE，返回本次新转为超期的单据。"""
        newly = []
        for loan in self.list(status=STATUS_LENT):
            passed, days = self._due_passed(loan["due_at"])
            if passed:
                self._touch(loan["loan_id"], status=STATUS_OVERDUE)
                item = self.get(loan["loan_id"])
                item["overdue_days"] = days
                newly.append(item)
        return newly

    # ── 归还核验 ────────────────────────────────────────────────────────
    def check_in(
        self,
        loan_id: str,
        *,
        expected_watermark: str | None,
        matched_watermark: str | None,
        identity_found: bool,
        result_id_of_found: str | None = None,
        carrier_hash: str = "",
        operator: str = "",
        detected: bool = False,
        note: str = "",
        probe_failed: str = "",
    ) -> dict[str, Any]:
        """落库一次归还核验。判定逻辑集中在这里，便于单测。

        参数
        ----
        identity_found        : 水印/结果库是否确认"这是本系统产出的数据"
        result_id_of_found    : 若是本系统数据，归位到的 result_id
        matched_watermark     : 实际检出的水印号
        expected_watermark    : 该借出单登记的水印号
        detected              : 载体上是否检出水印（用于区分"没有"与"检不出"）
        """
        loan = self.get(loan_id)
        detail: list[str] = []
        overdue, days = self._due_passed(loan["due_at"])

        if probe_failed:
            # 核验没跑通（载体路径没解包、格式不符、提取器报错）不等于"这不是我们的数据"。
            # 早先把两者混为一谈，界面上就出现"归还成功"和"非本系统成果"同时提示。
            verdict = VERDICT_UNKNOWN
            detail.append(f"本次未能完成水印核验，无法判定数据身份：{probe_failed}")
            detail.append("归还动作已登记，请修正载体后重新核验")
        elif not identity_found:
            verdict = VERDICT_NOT_OURS
            detail.append("载体上未检出水印，且结果库中无对应记录 → 判定为非本系统数据")
            if not detected:
                detail.append("水印检测本身未命中（可能是数据被替换、覆盖或非我们的成果）")
        elif loan["result_id"] and result_id_of_found and result_id_of_found != loan["result_id"]:
            verdict = VERDICT_MISMATCH
            detail.append(f"数据确为本系统产出，但归属 {result_id_of_found}，"
                          f"与本借出单登记的 {loan['result_id']} 不一致")
        elif not result_id_of_found:
            verdict = VERDICT_UNKNOWN
            detail.append("检出水印但无法归位到具体结果记录，需人工比对")
        elif expected_watermark and matched_watermark and \
                str(matched_watermark) != str(expected_watermark):
            verdict = VERDICT_MISMATCH
            detail.append(f"检出 watermark={matched_watermark}，"
                          f"与本借出单登记的 {expected_watermark} 不一致")
        elif overdue:
            verdict = VERDICT_OVERDUE
            detail.append(f"身份核对通过，但已逾期归还 {days} 天")
        else:
            verdict = VERDICT_OK
            detail.append("身份核对通过，且在归还期限内")

        if note:
            detail.append(note)

        returned_at = now_local()
        self._touch(
            loan_id,
            status=STATUS_RETURNED,
            returned_at=returned_at,
            return_verdict=verdict,
            return_detail=json.dumps(detail, ensure_ascii=False),
            carrier_hash=carrier_hash or loan.get("carrier_hash") or "",
        )
        return {
            "loan_id": loan_id,
            "result_id": loan["result_id"],
            "borrower": loan["borrower"],
            "verdict": verdict,
            "identity_confirmed": verdict in {VERDICT_OK, VERDICT_OVERDUE},
            "overdue": overdue,
            "overdue_days": days if overdue else 0,
            "due_at": loan["due_at"],
            "returned_at": returned_at,
            "detail": detail,
            "operator": operator,
        }

    def hash_carrier(self, path: Path | str) -> str:
        """流式哈希一个文件或目录，用于登记/比对载体。"""
        target = Path(path)
        digest = hashlib.sha256()
        if target.is_file():
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        if target.is_dir():
            for item in sorted(p for p in target.rglob("*") if p.is_file()):
                digest.update(item.name.encode("utf-8"))
                with item.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            return digest.hexdigest()
        return ""

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def summarize(loans: list[dict[str, Any]], *, now: datetime | None = None) -> dict[str, Any]:
    """工作台用汇总：在借 / 超期 / 已归还 计数与最近到期项。"""
    in_field, overdue, returned = [], [], []
    for loan in loans:
        status = loan.get("status")
        if status == STATUS_RETURNED:
            returned.append(loan)
        else:
            passed, days = LoanStore._due_passed(loan.get("due_at") or "", at=now)
            loan = {**loan, "overdue": passed, "overdue_days": days if passed else 0}
            (overdue if passed else in_field).append(loan)
    upcoming = sorted(in_field, key=lambda r: r.get("due_at") or "")[:5]
    return {
        "total": len(loans),
        "lent": len(in_field),
        "overdue": len(overdue),
        "returned": len(returned),
        "due_soon": upcoming,
        "overdue_items": overdue,
    }


def default_due(days: int = 30) -> str:
    """便捷函数：从现在起 `days` 天的归还期限（本地时区 ISO 字符串）。"""
    base = datetime.now()
    return (base + timedelta(days=days)).replace(microsecond=0).isoformat()
