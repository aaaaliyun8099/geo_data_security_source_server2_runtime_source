"""Durable application audit log used by server2 APIs."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path


class AuditLog:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, event: str, actor: str | None = None, **details) -> dict:
        entry = {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "event": event,
        }
        if actor is not None:
            entry["actor"] = actor
        entry.update(details)
        path = self.directory / f"audit-{entry['timestamp'][:10]}.jsonl"
        with self._lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    @staticmethod
    def retention_policy_days() -> int:
        return 180

    def retention_cutoff(self) -> datetime:
        return datetime.now().astimezone() - timedelta(days=self.retention_policy_days())

    def retention_evidence(self) -> dict:
        files = sorted(self.directory.glob("audit-*.jsonl"))
        cutoff = self.retention_cutoff().date().isoformat()
        dates = [path.stem.removeprefix("audit-") for path in files]
        observed_span_days = 0
        if dates:
            from datetime import date
            observed_span_days = max(0, (date.fromisoformat(max(dates)) - date.fromisoformat(min(dates))).days)
        return {
            "directory": str(self.directory),
            "policy_days": self.retention_policy_days(),
            "cutoff_date": cutoff,
            "files_present": len(files),
            "policy_configured": True,
            "observed_span_days": observed_span_days,
            "retention_verified": observed_span_days >= self.retention_policy_days(),
        }
