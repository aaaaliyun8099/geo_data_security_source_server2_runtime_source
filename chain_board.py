"""区块链存证看板页面：数据由服务端渲染进页面，打开即可看，不需要在浏览器里填令牌。

页面只读，不提供任何写操作。能否打开由 api.py 的路由判断：本机和内网地址直接可看，
其他来源需要在地址后带 ?token=；设置 GDS_CHAIN_VIEW_REQUIRE_TOKEN=1 可强制所有访问都要令牌。
"""
from __future__ import annotations

import json
from typing import Any

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>区块链存证看板</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; font-family: "Microsoft YaHei", "微软雅黑", sans-serif;
         font-size: 14px; color: #111; background: #f6f7f9; }
  h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
  .sub { color: #555; font-size: 13px; margin-bottom: 18px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-bottom: 20px; }
  .card { background: #fff; border: 1px solid #e2e5e9; border-radius: 6px; padding: 12px 14px; }
  .card .k { color: #666; font-size: 12px; margin-bottom: 6px; }
  .card .v { font-size: 18px; font-weight: 600; word-break: break-all; }
  .ok { color: #146c43; }
  .bad { color: #b02a37; }
  .muted { color: #888; }
  h2 { font-size: 15px; margin: 22px 0 10px; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e5e9; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid #eef0f3; font-size: 13px; }
  th { background: #fafbfc; color: #444; font-weight: 600; white-space: nowrap; }
  td.mono, .mono { font-family: Consolas, Monaco, monospace; font-size: 12px; }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 12px; border: 1px solid; }
  .b-ok { color: #146c43; border-color: #a3cfbb; background: #eefaf3; }
  .b-no { color: #b02a37; border-color: #f1aeb5; background: #fdf0f1; }
  .b-idle { color: #555; border-color: #d5d9de; background: #f4f5f7; }
  .note { color: #555; font-size: 12px; line-height: 1.7; margin-top: 18px; }
  .err { background: #fdf0f1; border: 1px solid #f1aeb5; color: #b02a37; padding: 9px 12px;
         border-radius: 4px; margin-bottom: 14px; display: none; }
  .foot { color: #888; font-size: 12px; margin-top: 10px; }
</style>
</head>
<body>
<h1>区块链存证看板</h1>
<div class="sub">每份成果是否写入区块链、写在哪个区块、交易哈希是多少。页面每 15 秒自动刷新。</div>

<div id="err" class="err"></div>
<div class="cards" id="cards"></div>

<h2>成果存证状态</h2>
<table>
  <thead><tr><th>成果编号</th><th>项目</th><th>数据类型</th><th>水印</th><th>上链</th><th>区块高度</th><th>交易哈希</th></tr></thead>
  <tbody id="rows"></tbody>
</table>

<h2>最近链上事件</h2>
<table>
  <thead><tr><th>时间</th><th>事件</th><th>成果编号</th><th>区块高度</th></tr></thead>
  <tbody id="events"></tbody>
</table>

<div class="note">
  判定口径：上链为“是”，表示该成果的链上事件确实在区块链上确认（交易状态成功，节点返回 onchain=true），
  并且本地哈希链校验通过；只有本地记录、链上没有交易的，显示“否”并给出原因。
</div>
<div class="foot" id="foot"></div>

<script>
const DATA_URL = "__DATA_URL__";
let DATA = __DATA__;

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v === null || v === undefined ? "" : v)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function short(text, head, tail) {
  if (!text) return "—";
  const t = String(text);
  return t.length <= head + tail + 3 ? t : t.slice(0, head) + "…" + t.slice(-tail);
}

function card(key, value, cls) {
  return '<div class="card"><div class="k">' + esc(key) + '</div><div class="v ' + (cls || "") + '">' + value + '</div></div>';
}

function badge(ok, yes, no) {
  return ok ? '<span class="badge b-ok">' + yes + '</span>'
            : '<span class="badge b-no">' + no + '</span>';
}

function render() {
  const status = DATA.status || {};
  const local = status.local_chain || {};
  $("cards").innerHTML = [
    card("区块链节点", status.node_reachable ? "可达" : "不可达", status.node_reachable ? "ok" : "bad"),
    card("链 ID", esc(status.chain_id === null || status.chain_id === undefined ? "—" : status.chain_id)),
    card("最新区块高度", Number(status.block_number || 0).toLocaleString()),
    card("已锚定记录", Number(status.anchored_records || 0).toLocaleString()),
    card("本地哈希链", local.valid ? "校验通过" : "校验失败", local.valid ? "ok" : "bad"),
    card("本地链记录数", Number(local.records || 0).toLocaleString())
  ].join("");

  const rows = (DATA.results || []);
  $("rows").innerHTML = rows.length ? rows.map(function (r) {
    const a = r.anchor || null;
    const anchored = !!(a && a.onchain === true && a.status === 1);
    const wm = r.watermark_enabled
      ? '<span class="badge b-ok">已加水印</span>'
      : '<span class="badge b-idle">未加水印</span>';
    const tx = anchored
      ? '<span title="' + esc(a.tx_hash) + '">' + esc(short(a.tx_hash, 12, 8)) + '</span>'
      : (r.reason ? '<span class="muted">' + esc(r.reason) + '</span>' : "—");
    return '<tr>' +
      '<td class="mono">' + esc(short(r.result_id, 8, 4)) + '</td>' +
      '<td>' + esc(r.project_id || "—") + '</td>' +
      '<td>' + esc(r.data_type || "—") + '</td>' +
      '<td>' + wm + '</td>' +
      '<td>' + badge(anchored, "已上链", "未上链") + '</td>' +
      '<td class="mono">' + (anchored ? esc(a.block_number) : "—") + '</td>' +
      '<td class="mono">' + tx + '</td>' +
      '</tr>';
  }).join("") : '<tr><td colspan="7" class="muted">暂无成果</td></tr>';

  const events = (DATA.events || []);
  $("events").innerHTML = events.length ? events.map(function (a) {
    return '<tr>' +
      '<td class="mono">' + esc(String(a.event_time || "").replace("T", " ").slice(0, 19)) + '</td>' +
      '<td>' + esc(a.event_type || "—") + '</td>' +
      '<td class="mono">' + esc(a.artifact_id ? short(a.artifact_id, 8, 4) : "—") + '</td>' +
      '<td class="mono">' + esc(a.block_number === null || a.block_number === undefined ? "—" : a.block_number) + '</td>' +
      '</tr>';
  }).join("") : '<tr><td colspan="4" class="muted">暂无记录</td></tr>';

  $("foot").textContent = "数据时间 " + (DATA.generated_at || "—") +
    "，每 15 秒自动刷新。数据来源 GET /api/v1/chain/status、GET /api/v1/workspace/status、GET /api/v1/chain/anchors。";
}

async function refresh() {
  try {
    const res = await fetch(DATA_URL, { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    DATA = await res.json();
    $("err").style.display = "none";
    render();
  } catch (e) {
    const box = $("err");
    box.textContent = "自动刷新失败，页面显示的是上一次的数据：" + (e.message || e);
    box.style.display = "block";
  }
}

render();
setInterval(refresh, 15000);
</script>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    """把数据直接嵌进页面，打开即可看到内容，无需在浏览器里填写令牌。"""
    blob = json.dumps(payload, ensure_ascii=False)
    # 数据里若出现 </script> 会提前结束脚本块，这里做转义
    blob = blob.replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA_URL__", "/chain-view/data").replace("__DATA__", blob)
