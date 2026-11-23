from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Iterator

from .crypto import canonical_json
from .timefmt import now_local


# Serializes appends to the anchor receipt file across ALL instances and background
# threads. Anchoring runs on a daemon thread per record, so several writers can hit the
# same file at once; without this the append interleaves and produces a torn line that
# breaks every later reader (measured 2026-11-12: one orphan "}" line + one blank line
# made GET /results/{id}/chain-proof return 500).
_ANCHOR_WRITE_LOCK = threading.Lock()


def _iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects from a JSONL file, skipping unusable lines.

    Evidence files are append-only and written by background threads, so a crash or a
    concurrent write can leave a partial line behind. A single bad line must never take
    down the whole query: the records before and after it are still valid evidence.
    """
    missing = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except (ValueError, TypeError):
            missing += 1
            continue
        if isinstance(row, dict):
            yield row
    if missing:
        print(f"warning: skipped {missing} unparsable line(s) in {path}", flush=True)


class LocalTraceChain:
    """Append-only Server2 evidence chain; external blockchain anchoring is separate."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def _hash(record: dict[str, Any]) -> str:
        return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()

    def _last_hash(self) -> str:
        """链尾哈希。缓存 + 从文件末尾反向读取，**不要**每次全表扫描。

        旧实现每次 append 都把整个链文件逐行读一遍（1100+ 条起步，且只增不减）。
        上传完成、任务提交都要 append，链越长越慢 —— 实测 complete 里 4.2 秒
        纯粹耗在这上面（链长 1116 条时，全读一遍约 4 秒，因为链在加密卷上）。
        修法：首次读时定下长度，之后用「文件大小是否变化」判断缓存是否还有效；
        链文件是纯追加的，前缀不会变，所以这个判断是安全的。
        """
        if not self.path.exists():
            return "0" * 64
        try:
            stat = self.path.stat()
        except OSError:
            return "0" * 64
        cached = getattr(self, "_last_hash_cache", None)
        if cached and cached[0] == (stat.st_size, stat.st_mtime_ns):
            return cached[1]
        # 反向读：从文件尾部取最后几 KB，最后一行就是链尾，O(1) 而不是 O(n)
        last = ""
        with self.path.open("rb") as stream:
            size = stat.st_size
            step = min(8192, size)
            pos = size - step
            tail = b""
            while True:
                stream.seek(pos)
                tail = stream.read(step) + tail
                lines = [ln for ln in tail.split(b"\n") if ln.strip()]
                # 至少要有一条完整的行，且这一行不是文件末尾的半行
                if len(lines) >= 2 or pos == 0:
                    if lines:
                        last = lines[-1].decode("utf-8")
                    break
                pos = max(0, pos - step)
                step *= 2
        if not last:
            return "0" * 64
        value = json.loads(last)["record_hash"]
        self._last_hash_cache = ((stat.st_size, stat.st_mtime_ns), value)
        return value

    def append(self, event_type: str, **details: Any) -> dict[str, Any]:
        with self._lock:
            # Timestamps are local time WITH an offset everywhere the operator can see them
            # (audit trail, API rows, this chain), so the same instant reads the same in a demo.
            # The value is part of the record hash, so old UTC records stay valid as written.
            stamp = now_local(millis=True)
            record = {
                "event_id": hashlib.sha256(f"{stamp}|{event_type}|{details}".encode()).hexdigest()[:32],
                "event_type": event_type,
                "timestamp": stamp,
                "prev_hash": self._last_hash(),
                **details,
            }
            record["record_hash"] = self._hash(record)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(canonical_json(record) + "\n")
            if __import__("os").environ.get("GDS_BESU_ANCHOR_ENABLED", "0").lower() in {"1", "true", "yes"}:
                # 上链是"等区块确认"的同步调用，实测 4 秒以上（网关 + Besu 出块）。
                # 挂在 API 的请求路径上会让每次上传完成都卡这么久，而链记录本身
                # 已经落盘（上面那行），锚定回执是可追溯的补充信息，晚几秒无副作用。
                # 所以放到后台线程：失败只记日志，不影响已落盘的链记录。
                threading.Thread(target=self._anchor_async, args=(record,), daemon=True).start()
            return record

    def _anchor_async(self, record: dict[str, Any]) -> None:
        """后台锚定一条已落盘的记录。失败不抛出，避免线程静默死掉时无人知晓。"""
        try:
            from .besu_anchor import anchor_record
            receipt = anchor_record(record)
            self._persist_anchor(record["record_hash"], record["event_type"], receipt)
        except Exception as exc:  # noqa: BLE001
            try:
                self._persist_anchor(record["record_hash"], record["event_type"],
                                     {"onchain": False, "status": 0, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass

    def _anchor_path(self) -> Path:
        return self.path.with_suffix(".anchors.jsonl")

    def _persist_anchor(self, record_hash: str, event_type: str, receipt: dict[str, Any]) -> None:
        """Store the on-chain receipt NEXT TO the chain, never inside it.

        The chain hash covers every field of a record, so folding the receipt (which can only be
        obtained after the record hash exists) back into the record would invalidate the chain.
        """
        row = {"record_hash": record_hash, "event_type": event_type,
               "timestamp": now_local(millis=True),
               "anchor": receipt}
        try:
            with _ANCHOR_WRITE_LOCK, self._anchor_path().open("a", encoding="utf-8") as stream:
                stream.write(canonical_json(row) + "\n")
        except OSError as error:  # anchoring already succeeded; receipt bookkeeping must not fail the task
            print(f"warning: could not persist anchor receipt for {record_hash}: {error}", flush=True)

    def anchor_for(self, record_hash: str | None) -> dict[str, Any] | None:
        """Latest on-chain receipt recorded for a chain record hash (None when not anchored)."""
        path = self._anchor_path()
        if not record_hash or not path.exists():
            return None
        wanted = str(record_hash).removeprefix("0x")
        found = None
        for line in _iter_json_lines(path):
            if str(line.get("record_hash", "")).removeprefix("0x") == wanted:
                found = line.get("anchor")
        return found

    def verify(self) -> dict[str, Any]:
        previous = "0" * 64
        checked = 0
        if not self.path.exists():
            return {"valid": True, "records": 0}
        # A torn line inside the hash chain is evidence of damage, not something to skip:
        # the link from that record onward cannot be proven. Report it as invalid with the
        # line number instead of raising, so callers still get a usable verdict.
        for number, raw in enumerate(self.path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except (ValueError, TypeError):
                return {"valid": False, "records": checked, "invalid_line": number,
                        "error": "unparsable record in chain"}
            if not isinstance(row, dict):
                return {"valid": False, "records": checked, "invalid_line": number,
                        "error": "non-object record in chain"}
            claimed = row.pop("record_hash", None)
            if row.get("prev_hash") != previous or claimed != self._hash(row):
                return {"valid": False, "records": checked, "invalid_event_id": row.get("event_id")}
            previous = claimed
            checked += 1
        return {"valid": True, "records": checked, "head_hash": previous}

    def search(self, **filters: Any) -> list[dict[str, Any]]:
        """Read-only lookup by auditable record fields."""
        rows = []
        if not self.path.exists():
            return rows
        for row in _iter_json_lines(self.path):
            if all(value in (None, "") or str(row.get(key)) == str(value) for key, value in filters.items()):
                rows.append(row)
        return rows
