"""One place that turns stored timestamps into the platform's operator-facing time.

SQLite's ``CURRENT_TIMESTAMP`` writes UTC, the audit trail writes local time, and the hash chain
used to write UTC - so the same instant showed up differently depending on which surface you
looked at. Every outward-facing timestamp now goes through :func:`local_time`: **local time with an
explicit offset** (matching the audit trail), so an API timestamp and its audit event agree.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

TIME_FIELDS = ("created_at", "updated_at", "last_used_at")


def local_time(value: Any) -> Any:
    """Present a stored UTC timestamp as local time with an explicit offset.

    Accepts SQLite's ``YYYY-MM-DD HH:MM:SS`` (treated as UTC), ISO strings and anything already
    carrying an offset. Non-timestamp values are returned unchanged, so this is safe to apply to
    every row field.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    text = value.strip()
    try:
        if "T" in text or text.endswith("Z") or "+" in text[10:]:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        else:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().isoformat(timespec="seconds")


def now_local(millis: bool = False) -> str:
    """Current local time as an ISO string with offset (the audit-trail format)."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds" if millis else "seconds")


def row_dict(row: Any) -> dict[str, Any]:
    """Row -> dict with every timestamp normalised to local time."""
    item = dict(row)
    for field in TIME_FIELDS:
        if field in item:
            item[field] = local_time(item[field])
    return item
