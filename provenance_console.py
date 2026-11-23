"""溯源查询控制台：服务端渲染页面 + 同源 API 调用，打开即可检测四类数据。

设计约束（与 sandbox_console / chain_board 保持一致）
------------------------------------------------------
* 页面**不含任何长期凭据**。服务令牌由服务端在渲染时注入到页面内的内存变量，
  不写入 Cookie、不写 localStorage、不落到浏览器历史里；刷新页面重新取。
* 提交检测走已有的 ``GET /api/v1/provenance/check-any``，不在页面里另开后端逻辑，
  避免出现"页面能跑、接口不一致"的两套实现。
* 结果展示必须区分三类"未检出"（沙箱路径不可读 / 引擎不可用 / 真未检出），
  不能一律显示成"不是我们的数据"。
"""
from __future__ import annotations

import json
from typing import Any

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>溯源查询</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; font-family:"Microsoft YaHei","微软雅黑",sans-serif;
         font-size:14px; color:#111; background:#f6f7f9; }
  h1 { font-size:20px; margin:0 0 4px; font-weight:600; }
  .sub { color:#555; font-size:13px; margin-bottom:18px; }
  form { background:#fff; border:1px solid #e2e5e9; border-radius:6px; padding:16px; margin-bottom:16px; }
  label { display:block; color:#444; font-size:12px; margin:12px 0 4px; }
  label:first-child { margin-top:0; }
  input[type=text] { width:100%; max-width:640px; padding:8px 10px;
    border:1px solid #cfd4da; border-radius:4px; font-size:13px;
    font-family:Consolas,Monaco,monospace; }
  .hint { color:#888; font-size:11px; margin-top:4px; }
  button { padding:9px 18px; border:1px solid #cfd4da; background:#fff; border-radius:4px;
           font-size:13px; cursor:pointer; margin-top:14px; }
  button.primary { background:#1f6feb; border-color:#1f6feb; color:#fff; }
  button:disabled { opacity:.55; cursor:not-allowed; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin-bottom:16px; }
  .card { background:#fff; border:1px solid #e2e5e9; border-radius:6px; padding:12px 14px; }
  .card .k { color:#666; font-size:12px; margin-bottom:6px; }
  .card .v { font-size:18px; font-weight:600; }
  .ok { color:#146c43; } .bad { color:#b02a37; } .warn { color:#9a6700; } .muted { color:#888; }
  table { width:100%; border-collapse:collapse; background:#fff; border:1px solid #e2e5e9; }
  th,td { text-align:left; padding:9px 10px; border-bottom:1px solid #eef0f3; font-size:13px; vertical-align:top; }
  th { background:#fafbfc; color:#444; font-weight:600; white-space:nowrap; width:130px; }
  tr:last-child td { border-bottom:none; }
  .mono { font-family:Consolas,Monaco,monospace; font-size:12px; word-break:break-all; }
  .concl { background:#fff; border:1px solid #e2e5e9; border-left-width:4px; border-radius:6px;
           padding:14px 16px; margin-bottom:16px; font-size:14px; }
  .concl.ok { border-left-color:#146c43; }
  .concl.bad { border-left-color:#b02a37; }
  .concl.warn { border-left-color:#9a6700; }
  .bar { height:6px; background:#eef0f3; border-radius:3px; overflow:hidden; margin-top:8px; }
  .bar>i { display:block; height:100%; background:#3d7ebd; }
  h2 { font-size:15px; margin:20px 0 10px; font-weight:600; }
  .spinner { display:none; color:#555; font-size:13px; margin-top:12px; }
  .spinner.on { display:block; }
  details { margin-top:10px; }
  summary { cursor:pointer; color:#1f6feb; font-size:12px; }
  pre { background:#fafbfc; border:1px solid #eef0f3; border-radius:4px; padding:10px;
        font-size:11px; overflow:auto; max-height:320px; margin:8px 0 0; }
</style>
</head>
<body>
<h1>溯源查询</h1>
<div class="sub">判断一份数据是否由本系统流出。支持 DOM / DEM / DLG(GPKG) / OSGB 四类，按文件类型自动分派检测通道。</div>

<form id="f" onsubmit="return false;">
  <label for="src">待检测文件路径（服务器绝对路径）</label>
  <input type="text" id="src" placeholder="V:\geo_data_security\projects\...\watermarked.tif">
  <div class="hint">文件必须位于系统工作区（V: 或 E: 盘），沙箱账户无权读取其他位置。</div>

  <label for="wm">登记的水印值</label>
  <input type="text" id="wm" placeholder="0x20261111">
  <div class="hint">十六进制需带 0x 前缀；纯数字会被当作十进制解析。</div>

  <label for="ref">参考图路径（可选，仅 GeoTIFF / DEM 使用）</label>
  <input type="text" id="ref" placeholder="留空则使用文件自身标签中的网格参数">
  <div class="hint">用于几何还原；不填也能检测未变换的文件。</div>

  <button class="primary" id="btn" onclick="run()">开始检测</button>
  <div class="spinner" id="spin">检测中，大文件可能需要数分钟，请勿关闭页面…</div>
</form>

<div id="out"></div>

<script>
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function fmtSec(s) {
  if (s == null) return "-";
  return s < 60 ? s.toFixed(1) + " 秒" : (s / 60).toFixed(1) + " 分钟";
}

function classifyMiss(d) {
  const via = (d.via || "") + "";
  const err = ((d.detail && d.detail.error) || "") + "";
  if (via.indexOf("trustmark_skipped") >= 0 || err.indexOf("沙箱") >= 0 || err.indexOf("工作区") >= 0)
    return {cls: "warn", label: "无法检测",
            tip: "该文件不在系统工作区内，沙箱账户无法读取。请把文件放到 V: 或 E: 盘后重试。"};
  if (err.toLowerCase().indexOf("unavailable") >= 0)
    return {cls: "warn", label: "引擎不可用",
            tip: "TrustMark 检测引擎不可用，请联系运维检查 GDS_TRUSTMARK_PYTHON 配置。"};
  return {cls: "bad", label: "未检出",
          tip: "未检出登记水印，无法证明该数据由本系统流出。"};
}

async function run() {
  const src = document.getElementById("src").value.trim();
  const wm  = document.getElementById("wm").value.trim();
  const ref = document.getElementById("ref").value.trim();
  const out = document.getElementById("out");
  const btn = document.getElementById("btn");
  const spin = document.getElementById("spin");

  if (!src || !wm) { out.innerHTML = '<div class="concl bad">请填写文件路径和水印值。</div>'; return; }

  btn.disabled = true; spin.classList.add("on"); out.innerHTML = "";
  const q = new URLSearchParams({source_path: src, expected_watermark: wm});
  if (ref) q.set("reference_path", ref);

  const t0 = performance.now();
  try {
    const res = await fetch("/provenance-view/check?" + q);
    const el = (performance.now() - t0) / 1000;
    if (!res.ok) {
      const body = await res.text();
      out.innerHTML = '<div class="concl bad">请求失败（HTTP ' + res.status + '）' +
        '<pre>' + esc(body.slice(0, 600)) + '</pre></div>';
      return;
    }
    render(await res.json(), el);
  } catch (e) {
    out.innerHTML = '<div class="concl bad">请求异常：' + esc(e.message) + '</div>';
  } finally {
    btn.disabled = false; spin.classList.remove("on");
  }
}

function render(r, elapsed) {
  const out = document.getElementById("out");
  const hit = !!r.is_probably_ours;
  let head;
  if (hit) {
    head = '<div class="concl ok"><b>是本系统数据</b><br>' + esc(r.conclusion) + '</div>';
  } else {
    const m = classifyMiss(r);
    head = '<div class="concl ' + m.cls + '"><b>' + m.label + '</b><br>' + esc(m.tip) + '</div>';
  }

  const cls = {RASTER_TRUSTMARK:"DOM (GeoTIFF)", RASTER_DEM:"DEM", GPKG:"DLG (GPKG)", OSGB:"OSGB"}[r.data_class] || r.data_class || "-";

  let cards = '<div class="cards">' +
    card("判定", hit ? "检出" : "未检出", hit ? "ok" : "bad") +
    card("数据类型", cls, "") +
    card("检测通道", r.channel || "-", "muted") +
    card("耗时", fmtSec(elapsed), "") +
  '</div>';

  const rows = [
    ["文件", r.source_path || document.getElementById("src").value],
    ["数据类型", cls],
    ["检测通道", r.channel || "-"],
    ["登记水印", r.expected_watermark],
    ["解出结果", r.resolved_watermark == null ? "—" : r.resolved_watermark],
    ["能力边界", r.capability || "-"],
  ];
  if (r.work_dir) rows.push(["工作目录", r.work_dir]);

  let tbl = '<table>' + rows.map(([k, v]) =>
    '<tr><th>' + esc(k) + '</th><td class="mono">' + esc(v) + '</td></tr>').join("") + '</table>';

  // 逐载体覆盖率条（只有 DOM 提供）
  let cov = "";
  const d = r.detail || {};
  if (typeof d.coverage === "number") {
    const pct = Math.round(d.coverage * 1000) / 10;
    cov = '<h2>逐载体覆盖率</h2><div class="cards"><div class="card">' +
      '<div class="k">命中 / 总载体</div><div class="v">' + (d.matched_carriers ?? "-") + ' / ' + (d.carriers ?? "-") + '</div>' +
      '<div class="bar"><i style="width:' + Math.min(100, pct) + '%"></i></div>' +
      '<div class="s ' + (d.meets_bar ? "ok" : "warn") + '">' + pct + '%' +
      (d.meets_bar ? '（达标，验收线 90%）' : '（未达 90% 验收线）') + '</div></div></div>';
  }

  const raw = '<details><summary>原始返回</summary><pre>' + esc(JSON.stringify(r, null, 2)) + '</pre></details>';

  out.innerHTML = head + cards + cov + '<h2>明细</h2>' + tbl + raw;

  if (!hit) {
    const m = classifyMiss(r);
    if (m.cls === "bad") {
      out.innerHTML += '<h2>可能的原因</h2><table>' +
        row("文件从未加过水印", "系统只在审核通过的成果上嵌入水印，原始上传件不含水印。") +
        row("水印值不匹配", "确认 expected_watermark 与该成果登记值一致（可在 manifest.json 中查看）。") +
        row("超出能力边界", "DOM 非 256 整数倍裁剪、DEM 缩小超 30%、OSGB 转 PNG 等情况会失效。") +
        '</table>';
    }
  }
}

function card(k, v, cls) {
  return '<div class="card"><div class="k">' + esc(k) + '</div><div class="v ' + (cls || "") + '">' + esc(v) + '</div></div>';
}
function row(k, v) {
  return '<tr><th>' + esc(k) + '</th><td>' + esc(v) + '</td></tr>';
}
</script>
</body>
</html>
"""


def render_html(service_token: str = "", generated_at: str = "") -> str:
    """渲染溯源控制台。

    页面**不需要令牌**：同源查询走 ``/provenance-view/check``，该端点由看板守卫统一鉴权。
    保留 ``service_token`` 参数只是为了兼容旧调用，默认不注入任何凭据。
    """
    # `__TOKEN__` 已从模板移除；保留替换逻辑以防旧模板残留。
    html = _TEMPLATE.replace("__TOKEN__", "null")
    return html
