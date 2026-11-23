from __future__ import annotations

import json
from pathlib import Path

from .crypto import decrypt_text, encrypt_text
from .watermark_policy import marker_digest, reembed_allowed


def _walk_features(value: dict) -> list[dict]:
    if value.get("type") == "FeatureCollection":
        return [item for item in value.get("features", []) if item.get("type") == "Feature"]
    if value.get("type") == "Feature":
        return [value]
    raise ValueError("GeoJSON must be a Feature or FeatureCollection")


def embed_geojson_watermark(source: Path, destination: Path, text: str, key: bytes,
                            report: dict | None = None) -> int:
    document = json.loads(Path(source).read_text(encoding="utf-8"))
    features = _walk_features(document)
    previous: list[str] = []
    existing = False
    for feature in features:
        properties = feature.setdefault("properties", {})
        if "_wm" in properties:
            existing = True
            if not reembed_allowed():
                raise ValueError("source already contains reserved property _wm（当前为严格取证策略，禁止二次嵌入）")
            previous.append(properties["_wm"])
        properties["_wm"] = encrypt_text(text, key)
    Path(destination).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    if report is not None:
        report.update({"rewatermarked": existing,
                       "previous_marker_sha256": marker_digest(previous) if existing else None})
    return len(features)


def extract_geojson_watermark(source: Path, key: bytes) -> list[str]:
    document = json.loads(Path(source).read_text(encoding="utf-8"))
    values = []
    for feature in _walk_features(document):
        properties = feature.get("properties") or {}
        if "_wm" not in properties:
            raise ValueError("watermark property _wm not found")
        values.append(decrypt_text(properties["_wm"], key))
    return values
