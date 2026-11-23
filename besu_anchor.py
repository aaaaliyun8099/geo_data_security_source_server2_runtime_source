from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from typing import Any

_EVENT_CODES = {"DATA_IMPORTED": 1, "WATERMARK_EMBEDDED": 2, "WATERMARK_VERIFIED": 3, "RESULT_APPROVED": 4, "RESULT_EXPORTED": 5, "WATERMARK_PARTIAL": 6, "WATERMARK_SKIPPED": 7, "DATA_UNIT_ANCHORED": 8}
_CLASS_CODES = {"DLG_LINEWORK": 1, "REMOTE_SENSING_RASTER": 2, "THEMATIC_VECTOR": 3, "OBLIQUE_MESH": 4, "ELEVATION_DEM": 5, "ORTHOIMAGE": 6, "POINT_CLOUD": 7}


def _hex32(value: str) -> str:
    text = str(value or "")
    return "0x" + text.removeprefix("0x")


def anchor_record(record: dict[str, Any]) -> dict[str, Any]:
    """Anchor a local trace record through the localhost Besu gateway."""
    url = os.environ.get("GDS_BESU_ANCHOR_URL", "http://127.0.0.1:18549/anchor")
    payload = {
        "record_hash": _hex32(record["record_hash"]),
        "prev_hash": _hex32(record.get("prev_hash", "0" * 64)),
        "artifact_hash": _hex32(record.get("artifact_id_hash") or hashlib.sha256(str(record.get("artifact_id", "")).encode()).hexdigest()),
        "output_hash": _hex32(record.get("output_hash", "0" * 64)),
        "classification": int(record.get("classification_code", _CLASS_CODES.get(str(record.get("classification", "")), 0))),
        "event_type": int(record.get("event_type_code", _EVENT_CODES.get(str(record.get("event_type", "")), 0))),
        "timestamp": int(record.get("timestamp_epoch", 0) or 0),
    }
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("status") != 1 or result.get("onchain") is not True:
        raise RuntimeError(f"Besu anchor was not confirmed: {result}")
    return result
