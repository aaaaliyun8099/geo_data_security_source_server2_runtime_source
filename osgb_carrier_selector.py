from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

_TILE = re.compile(r"^(Tile_[+-]\d+_[+-]\d+)")
_LOD = re.compile(r"_L(\d+)")


def spatial_tile(member: str) -> str:
    first = member.replace("\\", "/").split("/", 1)[0]
    match = _TILE.match(first)
    return match.group(1) if match else first


def lod(member: str) -> int:
    match = _LOD.search(member)
    return int(match.group(1)) if match else -1


def select_spatial_carriers(items: list[dict[str, Any]], min_side: int = 224) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[spatial_tile(str(item["member"]))].append(item)
    selected = []
    unavailable = []
    for tile, rows in sorted(grouped.items()):
        eligible = [row for row in rows if min(int(row.get("width", 0)), int(row.get("height", 0))) >= min_side]
        if not eligible:
            unavailable.append(tile)
            continue
        # Prefer the largest valid texture; break ties with higher LOD and stable path ordering.
        eligible.sort(key=lambda row: (int(row["width"]) * int(row["height"]), lod(str(row["member"])), str(row["member"])), reverse=True)
        selected.append(eligible[0])
    return {"selected": selected, "selected_count": len(selected),
            "spatial_tiles": len(grouped), "not_watermarkable_tiles": unavailable,
            "not_watermarkable_count": len(unavailable)}
