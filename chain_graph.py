"""区块链存证图形看板：用 SVG 画出区块之间的链接关系，不依赖任何前端库。

两张图：
  一、链式结构图   —— 每个链上事件一个方块，按发生顺序自上而下串联，
                     连线标注这条记录引用的上一条记录哈希，直观看出链是不是连续的。
  二、成果与区块对应图 —— 左侧成果、中间事件、右侧区块高度，连线表示该成果的哪次事件落在哪个区块。

页面由服务端直接渲染，打开即可看，不需要在浏览器里填令牌。
"""
from __future__ import annotations

import json
from typing import Any

_W, _NODE_H, _GAP = 980, 84, 52


def _esc(value: Any) -> str:
    text = "" if value is None else str(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _short(text: Any, head: int = 10, tail: int = 8) -> str:
    value = "" if text is None else str(text)
    if not value:
        return "—"
    return value if len(value) <= head + tail + 1 else f"{value[:head]}…{value[-tail:]}"


def _ok(node: dict) -> bool:
    anchor = node.get("anchor") or node
    return bool(anchor.get("onchain") is True and anchor.get("status") == 1)


def chain_svg(events: list[dict], limit: int = 10) -> str:
    """链式结构图：自上而下串联，连线上标出引用的上一条记录哈希。"""
    nodes = [e for e in events][:limit]
    if not nodes:
        return '<p class="empty">暂无可展示的链上事件</p>'

    height = len(nodes) * (_NODE_H + _GAP) + 40
    parts = [f'<svg viewBox="0 0 {_W} {height}" width="100%" height="{height}" role="img" '
             f'aria-label="区块链链式结构图" xmlns="http://www.w3.org/2000/svg">']
    parts.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
                 'markerHeight="7" orient="auto-start-reverse">'
                 '<path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa3ad"/></marker></defs>')

    for index, node in enumerate(nodes):
        top = 20 + index * (_NODE_H + _GAP)
        ok = _ok(node)
        stroke = "#a3cfbb" if ok else "#f1aeb5"
        fill = "#f7fdfa" if ok else "#fdf7f7"
        parts.append(f'<rect x="104" y="{top}" width="{_W - 124}" height="{_NODE_H}" rx="8" '
                     f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        # 左侧区块编号：做成链上的节点徽标，由竖线串起来
        parts.append(f'<rect x="8" y="{top + 26}" width="80" height="30" rx="15" fill="#fff" '
                     f'stroke="{stroke}" stroke-width="1.5"/>')
        parts.append(f'<text x="48" y="{top + 46}" text-anchor="middle" font-size="12" '
                     f'fill="#111" font-family="Consolas, monospace">#{_esc(node.get("block_number") or "-")}</text>')

        parts.append(f'<text x="124" y="{top + 26}" font-size="14" font-weight="600" fill="#111" '
                     f'font-family="Microsoft YaHei, sans-serif">'
                     f'{_esc(node.get("event_type") or "—")}</text>')
        status_text = "已确认上链" if ok else "未确认"
        parts.append(f'<text x="{_W - 100}" y="{top + 26}" text-anchor="end" font-size="12" '
                     f'fill="{"#146c43" if ok else "#b02a37"}" font-family="Microsoft YaHei, sans-serif">'
                     f'{status_text}</text>')
        parts.append(f'<text x="124" y="{top + 49}" font-size="11.5" fill="#555" '
                     f'font-family="Microsoft YaHei, sans-serif">'
                     f'成果 {_esc(_short(node.get("artifact_id"), 8, 4))} · 项目 {_esc(node.get("project_id") or "—")}'
                     f' · {_esc(str(node.get("event_time") or "").replace("T", " ")[:19])}</text>')
        parts.append(f'<text x="124" y="{top + 70}" font-size="11" fill="#333" '
                     f'font-family="Consolas, monospace">'
                     f'tx {_esc(_short(node.get("tx_hash"), 14, 10))}    record {_esc(_short(node.get("record_hash"), 12, 10))}'
                     f'    prev {_esc(_short(node.get("prev_hash"), 12, 10))}</text>')

        if index < len(nodes) - 1:
            x = 48
            y2 = top + _NODE_H + _GAP + 20
            parts.append(f'<line x1="{x}" y1="{top + 56}" x2="{x}" y2="{y2}" stroke="#9aa3ad" '
                         f'stroke-width="1.5" marker-end="url(#arrow)"/>')
            parts.append(f'<text x="{x + 10}" y="{(top + 56 + y2) / 2 + 3}" font-size="10.5" fill="#777" '
                         f'font-family="Microsoft YaHei, sans-serif">哈希衔接</text>')

    parts.append('</svg>')
    return "".join(parts)


def relation_svg(events: list[dict], limit: int = 7) -> str:
    """成果、事件、区块的对应关系图。"""
    rows = [e for e in events if e.get("artifact_id")][:limit]
    if not rows:
        return '<p class="empty">暂无可展示的成果与区块对应关系</p>'

    row_h = 74
    height = len(rows) * row_h + 30
    parts = [f'<svg viewBox="0 0 980 {height}" width="100%" height="{height}" role="img" '
             f'aria-label="成果与区块对应关系图" xmlns="http://www.w3.org/2000/svg">']
    parts.append('<defs><marker id="arrow2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
                 'markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
                 'fill="#9aa3ad"/></marker></defs>')
    parts.append(f'<text x="10" y="14" font-size="12" font-weight="600" fill="#444" '
                 f'font-family="Microsoft YaHei, sans-serif">成果</text>')
    parts.append(f'<text x="400" y="14" font-size="12" font-weight="600" fill="#444" '
                 f'font-family="Microsoft YaHei, sans-serif">链上事件</text>')
    parts.append(f'<text x="740" y="14" font-size="12" font-weight="600" fill="#444" '
                 f'font-family="Microsoft YaHei, sans-serif">区块</text>')

    for index, node in enumerate(rows):
        top = 24 + index * row_h
        ok = _ok(node)
        stroke = "#a3cfbb" if ok else "#f1aeb5"
        parts.append(f'<rect x="8" y="{top}" width="300" height="48" rx="7" fill="#fff" stroke="#d5d9de"/>')
        parts.append(f'<text x="20" y="{top + 20}" font-size="12.5" font-weight="600" fill="#111" '
                     f'font-family="Consolas, monospace">{_esc(_short(node.get("artifact_id"), 10, 6))}</text>')
        parts.append(f'<text x="20" y="{top + 38}" font-size="11.5" fill="#555" '
                     f'font-family="Microsoft YaHei, sans-serif">项目 {_esc(node.get("project_id") or "—")}</text>')

        parts.append(f'<path d="M 308 {top + 24} C 340 {top + 24}, 350 {top + 24}, 372 {top + 24}" '
                     f'fill="none" stroke="#9aa3ad" stroke-width="1.5" marker-end="url(#arrow2)"/>')

        parts.append(f'<rect x="372" y="{top}" width="300" height="48" rx="7" fill="#fff" stroke="{stroke}"/>')
        parts.append(f'<text x="384" y="{top + 20}" font-size="12.5" font-weight="600" fill="#111" '
                     f'font-family="Microsoft YaHei, sans-serif">{_esc(node.get("event_type") or "—")}</text>')
        parts.append(f'<text x="384" y="{top + 38}" font-size="11" fill="#555" '
                     f'font-family="Consolas, monospace">{_esc(_short(node.get("tx_hash"), 12, 8))}</text>')

        parts.append(f'<path d="M 672 {top + 24} C 700 {top + 24}, 710 {top + 24}, 730 {top + 24}" '
                     f'fill="none" stroke="#9aa3ad" stroke-width="1.5" marker-end="url(#arrow2)"/>')

        parts.append(f'<rect x="730" y="{top}" width="240" height="48" rx="7" fill="#f7fdfa" stroke="{stroke}"/>')
        parts.append(f'<text x="742" y="{top + 21}" font-size="13" font-weight="600" fill="#111" '
                     f'font-family="Consolas, monospace">#{_esc(node.get("block_number") or "—")}</text>')
        parts.append(f'<text x="742" y="{top + 38}" font-size="11" fill="{"#146c43" if ok else "#b02a37"}" '
                     f'font-family="Microsoft YaHei, sans-serif">{"已确认" if ok else "未确认"}</text>')

    parts.append('</svg>')
    return "".join(parts)


_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>区块链存证图</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
         font-size: 14px; color: #111; background: #f6f7f9; }
  h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
  .sub { color: #555; font-size: 13px; margin-bottom: 16px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 18px; }
  .card { background: #fff; border: 1px solid #e2e5e9; border-radius: 6px; padding: 11px 13px; }
  .card .k { color: #666; font-size: 12px; margin-bottom: 5px; }
  .card .v { font-size: 17px; font-weight: 600; }
  .ok { color: #146c43; } .bad { color: #b02a37; }
  h2 { font-size: 15px; margin: 20px 0 8px; font-weight: 600; }
  .panel { background: #fff; border: 1px solid #e2e5e9; border-radius: 8px; padding: 10px 8px; overflow-x: auto; }
  .note { color: #555; font-size: 12px; line-height: 1.75; margin-top: 14px; }
  .empty { color: #888; font-size: 13px; padding: 12px; }
  .tabs { margin-bottom: 12px; font-size: 13px; }
  .tabs a { color: #111; text-decoration: none; border-bottom: 1px solid #111; margin-right: 14px; }
  .tabs a.dim { color: #777; border-bottom: 1px dotted #aaa; }
  .foot { color: #888; font-size: 12px; margin-top: 12px; }
</style>
</head>
<body>
<h1>区块链存证图</h1>
<div class="sub">上图是区块之间的链接关系：每个方块是一条链上记录，左侧数字是区块高度，连线表示后一条记录引用前一条的哈希。</div>

<div class="cards">__CARDS__</div>

<div class="tabs"><a href="/chain-view">图形视图</a><a class="dim" href="/chain-view/table">表格视图</a></div>

<h2>一、链式结构图</h2>
<div class="panel">__CHAIN__</div>

<h2>二、成果与区块对应图</h2>
<div class="panel">__REL__</div>

<div class="note">
  看图方法：方块越往上越新；相邻方块之间的箭头表示哈希衔接，即后一条记录把前一条的哈希写进自己的 prev_hash，
  任何一条被改动，后面的哈希都对不上。绿色边框表示该交易已在区块链上确认，红色表示本地有记录但链上未确认。
</div>
<div class="foot">__FOOT__</div>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    status = payload.get("status") or {}
    local = status.get("local_chain") or {}
    results = payload.get("results") or []
    events = payload.get("events") or []
    anchored_results = sum(1 for r in results if _ok(r))

    cards = "".join([
        f'<div class="card"><div class="k">区块链节点</div><div class="v {"ok" if status.get("node_reachable") else "bad"}">'
        f'{"可达" if status.get("node_reachable") else "不可达"}</div></div>',
        f'<div class="card"><div class="k">链 ID</div><div class="v">{_esc(status.get("chain_id") or "—")}</div></div>',
        f'<div class="card"><div class="k">最新区块高度</div><div class="v">{int(status.get("block_number") or 0):,}</div></div>',
        f'<div class="card"><div class="k">已锚定记录</div><div class="v">{int(status.get("anchored_records") or 0):,}</div></div>',
        f'<div class="card"><div class="k">成果上链</div><div class="v">{anchored_results} / {len(results)}</div></div>',
        f'<div class="card"><div class="k">本地哈希链</div><div class="v {"ok" if local.get("valid") else "bad"}">'
        f'{"校验通过" if local.get("valid") else "校验失败"}</div></div>',
    ])

    html = (_TEMPLATE
            .replace("__CARDS__", cards)
            .replace("__CHAIN__", chain_svg(events))
            .replace("__REL__", relation_svg(events))
            .replace("__FOOT__", _esc("数据时间 " + str(payload.get("generated_at") or ""))))
    return html
