"""Incomplete-source acceptance: turn a bounded block scan into a frontend-visible warning.

The platform deliberately accepts a source whose trailing blocks cannot be read: the readable region
is processed and watermarked, the unreadable windows are written as explicit empty areas, and the
artifact is delivered as a partial result under manual review. This module builds the single warning
record that the API (task object, submit response, ingest integrity check) and the manifest all
expose, so the frontend never has to infer "why is this result partial?" from raw counters.
"""
from __future__ import annotations

WARNING_CODE = "SOURCE_TRUNCATED"


def source_integrity_warning(scan: dict | None) -> dict | None:
    """A structured warning for an incomplete raster scan, or None when the source is intact.

    `scan` is a `raster_read_scan` record: blocks_total/blocks_read/failed_blocks plus the raster's
    own geometry. The byte estimate is derived from the missing-block ratio and is labelled as an
    estimate, because tile sizes are uniform but compression is not.
    """
    if not scan or scan.get("source_integrity") != "INCOMPLETE":
        return None
    total = int(scan.get("blocks_total") or 0)
    read = int(scan.get("blocks_read") or 0)
    missing = int(scan.get("blocks_missing") or max(0, total - read))
    failed = scan.get("failed_blocks") or []
    first = failed[0] if failed else {}
    ratio = (missing / total) if total else None
    size_bytes = int(scan.get("size_bytes") or 0)
    estimate = int(size_bytes * ratio) if (size_bytes and ratio is not None) else None
    if ratio is None:
        scale = "占比未知"
    elif estimate:
        scale = f"缺失约 {ratio:.2%}、约 {estimate} 字节（估算）"
    else:
        scale = f"缺失约 {ratio:.2%}"
    return {
        "code": WARNING_CODE,
        "level": "WARNING",
        "source_integrity": "INCOMPLETE",
        "blocks_total": total,
        "blocks_read": read,
        "blocks_missing": missing,
        "missing_ratio": round(ratio, 6) if ratio is not None else None,
        "missing_bytes_estimate": estimate,
        "first_failed_block": ({key: first.get(key) for key in ("ordinal", "row", "col")} if first else None),
        "message": (
            f"源文件不完整：{missing}/{total} 个数据块无法读取（{scale}）。"
            "平台仍会继续处理可读取区域并嵌入水印，缺失区域在成果中显式置空；"
            "该成果按「部分成果」交付并需人工复核，不能作为完整数据验收。"
        ),
    }


def unreadable_source_warning(reason: str) -> dict:
    """The source could not be opened at all - a different verdict from a partial read."""
    return {
        "code": "SOURCE_UNREADABLE",
        "level": "ERROR",
        "source_integrity": "UNREADABLE",
        "message": (
            f"源文件无法打开或整体不可读（{reason[:300]}）。"
            "请重新导出并重新上传；重复上传同一文件会得到同样结果。"
        ),
    }
