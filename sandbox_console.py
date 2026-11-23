"""沙箱容量管理页面：服务端渲染，打开即可见容量、占用与清理入口。

设计约束（和 chain_board.py 保持一致）
------------------------------------
* 页面由服务端把数据渲染进 HTML，管理员打开 URL 就能看到真实容量，不需要先用 curl 换令牌。
* 页面本身**不含任何长期凭据**。写操作（改配额/清理）要求管理员在页面内输入自己的
  管理员账号口令换取短期管理令牌，令牌只存在浏览器内存里，刷新即失效。
  这样页面源码、浏览器历史、代理日志里都不会出现可用于管理的长期凭证。
* 所有数字都标注来源：`实测` = disk_usage/walk 当场读出；`策略` = 管理员配置。
  两者不混算，避免把配额当成实际占用汇报。
"""
from __future__ import annotations

from typing import Any

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>沙箱容量管理</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; padding:24px; font-family:"Microsoft YaHei","微软雅黑",sans-serif;
         font-size:14px; color:#111; background:#f6f7f9; }
  h1 { font-size:20px; margin:0 0 4px; font-weight:600; }
  .sub { color:#555; font-size:13px; margin-bottom:18px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-bottom:18px; }
  .card { background:#fff; border:1px solid #e2e5e9; border-radius:6px; padding:12px 14px; }
  .card .k { color:#666; font-size:12px; margin-bottom:6px; }
  .card .v { font-size:19px; font-weight:600; }
  .card .s { color:#888; font-size:11px; margin-top:4px; }
  .ok { color:#146c43; } .bad { color:#b02a37; } .warn { color:#9a6700; }
  h2 { font-size:15px; margin:22px 0 10px; font-weight:600; }
  table { width:100%; border-collapse:collapse; background:#fff; border:1px solid #e2e5e9; }
  th,td { text-align:left; padding:9px 10px; border-bottom:1px solid #eef0f3; font-size:13px; }
  th { background:#fafbfc; color:#444; font-weight:600; white-space:nowrap; }
  tr:last-child td { border-bottom:none; }
  .mono { font-family:Consolas,Monaco,monospace; font-size:12px; }
  .bar { height:8px; background:#eef0f3; border-radius:4px; overflow:hidden; margin-top:8px; }
  .bar>i { display:block; height:100%; background:#3d7ebd; }
  .bar>i.hot { background:#b02a37; }
  form { background:#fff; border:1px solid #e2e5e9; border-radius:6px; padding:14px; margin-bottom:14px; }
  label { display:block; color:#444; font-size:12px; margin:10px 0 4px; }
  input[type=text], input[type=password], input[type=number] {
    width:100%; max-width:320px; padding:7px 9px; border:1px solid #cfd4da; border-radius:4px; font-size:13px; }
  button { padding:8px 16px; border:1px solid #cfd4da; background:#fff; border-radius:4px;
           font-size:13px; cursor:pointer; margin-top:12px; }
  button.primary { background:#1f6feb; border-color:#1f6feb; color:#fff; }
  button.danger { background:#b02a37; border-color:#b02a37; color:#fff; }
  .vact { font-size:11px; padding:3px 7px; border-radius:3px; border:1px solid #cfd4da;
          background:#fff; color:#333; cursor:pointer; }
  .vact:hover { background:#eef2f6; }
  .vact.danger { background:#fff; border-color:#d9a0a6; color:#b02a37; }
  .vact.danger:hover { background:#fdf0f1; }
  .pill { display:inline-block; padding:2px 7px; border-radius:10px; font-size:11px;
          background:#eef2f6; color:#4a5568; }
  .pill.ok { background:#e3f4e9; color:#1e6b3a; }
  .pill.warn { background:#fdf5e3; color:#8a6100; }
  .pill.bad { background:#fdf0f1; color:#b02a37; }
  .mini { font-size:11px; color:#7a828c; }
  .acts { white-space:nowrap; }
  table tfoot td { font-weight:600; background:#f7f9fb; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .err { background:#fdf0f1; border:1px solid #f1aeb5; color:#b02a37; padding:9px 12px;
         border-radius:4px; margin-bottom:14px; display:none; }
  .msg { background:#eef7f0; border:1px solid #a3cfbb; color:#146c43; padding:9px 12px;
         border-radius:4px; margin-bottom:14px; display:none; }
  .note { color:#555; font-size:12px; line-height:1.75; margin-top:16px; }
  .foot { color:#888; font-size:12px; margin-top:10px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:flex-end; }
</style>
</head>
<body>
<h1>沙箱容量管理</h1>
<div class="sub">设置数据沙箱的容量上限、查看真实占用、清理数据。所有"实测"数字为当场读取。</div>
<div id="err" class="err"></div>
<div id="msg" class="msg"></div>

__CARDS__

__AREAS__

<h2>1. 管理员登录（写操作需要）</h2>
<form id="loginForm">
  <div class="note" style="margin-top:0">页面只显示容量，不含任何凭据。改配额与清理需要在此输入管理员账号口令，换取仅存于本页内存的管理令牌。</div>
  <div class="row">
    <div><label>管理员账号</label><input type="text" id="user" value="admin" autocomplete="username"></div>
    <div><label>口令</label><input type="password" id="pass" autocomplete="current-password"></div>
    <div><button type="submit" class="primary">登录</button></div>
  </div>
  <div class="foot" id="loginState">未登录：只能查看，不能修改。</div>
</form>

<h2>2. 设置容量上限</h2>
<form id="quotaForm">
  <div class="note" style="margin-top:0">
    上限不得超过加密卷实际容量；当前卷上限见上方"卷容量（实测）"。改大上限不会自动扩容卷，
    卷容量受宿主机加密容器文件大小限制（扩容需要停服操作，见部署文档）。
  </div>
  <div class="row">
    <div><label>沙箱容量上限 (GB)</label><input type="number" id="quotaGb" min="1" step="1" value="__QUOTA_GB__"></div>
    <div><button type="submit" class="primary">保存上限</button></div>
  </div>
  <div class="foot">
    上限生效于下一次上传：超过上限的新上传会在建会话时直接返回 507，不会传一半才失败。
    如需"到点自动清理"这类策略，本站目前没有定时执行器，不做假开关——请用下面的手动清理。
  </div>
</form>

<h2>3. 沙箱卷池（加卷 / 上线 / 下线）</h2>
<div class="note" style="margin-top:0">
  <b>容量不够时加一个卷，不要试图把现有卷撑大</b>：VeraCrypt 扩容只能 GUI 操作且要停服；
  而<b>新建卷全程命令行、业务不中断</b>。新上传会按"优先级高 + 剩余大"自动选卷。
  `state/`（哈希链、队列库、管理员凭据）只在主卷上，主卷不可卸载或移除。
</div>
__VOLUME_ROWS__
<form id="volumeForm">
  <div class="row">
    <div><label>新卷大小 (GB)</label><input type="number" id="volSize" min="1" step="1" value="100"></div>
    <div><label>盘符（留空自动分配）</label><input type="text" id="volLetter" maxlength="1" style="width:60px" placeholder="自动"></div>
    <div><label>优先级（大优先）</label><input type="number" id="volPriority" min="0" max="1000" step="1" value="50" style="width:90px"></div>
    <div><label>标签</label><input type="text" id="volLabel" placeholder="可选"></div>
    <div><button type="submit" class="primary">新建卷</button></div>
  </div>
  <div class="foot">建卷会在宿主机磁盘上写一个等大的加密容器文件（需要宿主机有对应剩余空间），并挂载、建目录骨架后加入池。</div>
</form>
<div class="foot" id="volOut"></div>

<h2>3. 清理沙箱数据</h2>
<form id="cleanForm">
  <div class="note" style="margin-top:0">
    <b>不会</b>删除：链存证（state/）、审计日志（audit/）、沙箱执行体（sandbox/worker/）。
    <b>会</b>删除所选数据区内的全部内容，并清空任务/上传/成果队列记录。
    有任务正在运行时会被拒绝，除非勾选强制。
  </div>
  <div class="row">
    <div><label>清理范围</label>
      <select id="area" style="padding:7px 9px;border:1px solid #cfd4da;border-radius:4px;font-size:13px">
        <option value="all">全部数据区</option>
        <option value="incoming">仅 incoming（上传原始件）</option>
        <option value="projects">仅 projects（处理中间件/成果）</option>
        <option value="exports">仅 exports（导出件）</option>
        <option value="sandbox_inbox">仅 sandbox_inbox</option>
        <option value="provenance_geom">仅 provenance_geom</option>
      </select>
    </div>
    <div><label><input type="checkbox" id="force"> 强制（有运行中任务也执行）</label></div>
    <div><button type="button" id="dryBtn">先预演（不删除）</button></div>
    <div><button type="button" id="cleanBtn" class="danger">确认清理</button></div>
  </div>
  <div class="foot" id="cleanOut"></div>
</form>

<div class="note">
  体积口径：1 GB = 1024³ 字节。"策略上限"是管理员设定值；"卷容量"是加密卷的物理大小；
  "沙箱已用"是 incoming/projects/exports 三个数据区的实测字节数。三者独立，不互相推导。
</div>
<div class="foot">数据生成时间：__GENERATED__　·　刷新本页获取最新实测值</div>

<script>
let TOKEN = null;
const $ = (id) => document.getElementById(id);
function show(id, text) { const e=$(id); e.textContent=text; e.style.display='block'; }
function hide(id){ $(id).style.display='none'; }
function fail(text){ hide('msg'); show('err', text); }
function okay(text){ hide('err'); show('msg', text); }

function auth() { return TOKEN ? {'X-Admin-Token': TOKEN} : {}; }

$('loginForm').addEventListener('submit', async (ev) => {
  ev.preventDefault(); hide('err'); hide('msg');
  try {
    const r = await fetch('/admin/auth/login', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({username:$('user').value, password:$('pass').value})
    });
    const body = await r.json().catch(()=>({}));
    if (!r.ok) { fail('登录失败：' + (body.detail || r.status)); return; }
    TOKEN = body.token || (body.admin_token) || null;
    $('pass').value = '';
    $('loginState').textContent = '已登录：' + (body.username || $('user').value) + '（令牌仅存本页内存）';
    okay('登录成功，现在可以修改配额或清理数据。');
  } catch (e) { fail('登录请求失败：' + e); }
});

$('quotaForm').addEventListener('submit', async (ev) => {
  ev.preventDefault(); hide('err'); hide('msg');
  if (!TOKEN) { fail('请先登录。'); return; }
  const gb = Number($('quotaGb').value);
  if (!gb || gb < 1) { fail('容量上限至少 1 GB。'); return; }
  try {
    const r = await fetch('/admin/sandbox/quota', {
      method:'POST', headers: Object.assign({'Content-Type':'application/json'}, auth()),
      body: JSON.stringify({ quota_bytes: Math.round(gb * 1024*1024*1024),
                             reason: 'management page' })
    });
    const body = await r.json().catch(()=>({}));
    if (!r.ok) { fail('保存失败：' + (body.detail || r.status)); return; }
    okay('已保存。新的上限生效于下一次上传。');
    setTimeout(()=>location.reload(), 800);
  } catch (e) { fail('请求失败：' + e); }
});

async function doClean(dryRun) {
  hide('err'); hide('msg');
  if (!TOKEN) { fail('请先登录。'); return; }
  if (!dryRun && !confirm('确认删除所选沙箱数据？队列记录也会被清空，此操作不可撤销。')) return;
  const area = $('area').value;
  const payload = { dry_run: dryRun, force: $('force').checked };
  if (area !== 'all') payload.areas = [area];
  try {
    const r = await fetch('/admin/sandbox/cleanup', {
      method:'POST', headers: Object.assign({'Content-Type':'application/json'}, auth()),
      body: JSON.stringify(payload)
    });
    const body = await r.json().catch(()=>({}));
    if (!r.ok) { fail('清理失败：' + (body.detail || r.status)); return; }
    const lines = Object.entries(body.areas || {}).map(([k,v]) =>
      `${k}: ${v.files} 个文件 / ${(v.bytes/1024/1024/1024).toFixed(2)} GB`);
    $('cleanOut').textContent = (dryRun ? '预演结果（未删除）：' : '已清理：') + lines.join('；') +
      `　释放 ${(body.freed_bytes/1024/1024/1024).toFixed(2)} GB`;
    okay(dryRun ? '预演完成，未删除任何文件。' : '清理完成。');
    if (!dryRun) setTimeout(()=>location.reload(), 1200);
  } catch (e) { fail('请求失败：' + e); }
}
$('dryBtn').addEventListener('click', ()=>doClean(true));
$('cleanBtn').addEventListener('click', ()=>doClean(false));

// ---- 卷池操作 ----
async function volAct(act, id) {
  hide('err'); hide('msg');
  if (!TOKEN) { fail('请先登录。'); return; }
  if (act === 'remove' && !confirm('把该卷移出池？容器文件与其中数据都会保留，只是不再分配新上传给它。')) return;
  if (act === 'disable' && !confirm('停止接收新数据？该卷上已有数据仍可读。')) return;
  const paths = {mount:'mount', unmount:'unmount', enable:'enable', disable:'disable', remove:''};
  const method = act === 'remove' ? 'DELETE' : 'POST';
  const url = `/admin/sandbox/volumes/${encodeURIComponent(id)}` + (paths[act] ? '/' + paths[act] : '');
  try {
    const opts = {method, headers: Object.assign({}, auth())};
    if (method === 'POST') {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify({reason: 'management page'});
    }
    const r = await fetch(url, opts);
    const body = await r.json().catch(()=>({}));
    if (!r.ok) { fail('操作失败：' + (typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))); return; }
    okay('操作成功，正在刷新…');
    setTimeout(()=>location.reload(), 700);
  } catch (e) { fail('请求失败：' + e); }
}
document.addEventListener('click', (ev) => {
  const b = ev.target.closest('.vact');
  if (!b) return;
  ev.preventDefault();
  volAct(b.dataset.act, b.dataset.id);
});

$('volumeForm').addEventListener('submit', async (ev) => {
  ev.preventDefault(); hide('err'); hide('msg');
  if (!TOKEN) { fail('请先登录。'); return; }
  const sizeGb = Number($('volSize').value);
  if (!sizeGb || sizeGb < 1) { fail('卷大小至少 1 GB。'); return; }
  const letter = ($('volLetter').value || '').trim().toUpperCase();
  const payload = { size_gb: Math.round(sizeGb), priority: Number($('volPriority').value) || 50 };
  if (letter) payload.letter = letter;
  const label = ($('volLabel').value || '').trim();
  if (label) payload.label = label;
  $('volOut').textContent = '已提交建卷任务，正在后台格式化加密容器…（大卷需要几分钟，可刷新页面查看）';
  try {
    const r = await fetch('/admin/sandbox/volumes', {
      method:'POST', headers: Object.assign({'Content-Type':'application/json'}, auth()),
      body: JSON.stringify(payload)
    });
    const body = await r.json().catch(()=>({}));
    if (!r.ok) {
      const d = body.detail;
      fail('建卷失败：' + (typeof d === 'string' ? d : JSON.stringify(d)));
      $('volOut').textContent = '';
      return;
    }
    pollJob(body.job_id);
  } catch (e) { fail('请求失败：' + e); $('volOut').textContent = ''; }
});

// 建卷是分钟级后台任务，这里轮询进度；不阻塞页面。
async function pollJob(jobId) {
  const out = $('volOut');
  const stageText = {queued:'排队中', formatting:'正在格式化加密容器', registering:'正在登记到卷池', done:'完成', failed:'失败'};
  for (let i = 0; i < 300; i++) {
    await new Promise(res => setTimeout(res, 3000));
    let job;
    try {
      const r = await fetch(`/admin/sandbox/volumes/jobs/${encodeURIComponent(jobId)}`, {headers: auth()});
      job = await r.json();
    } catch (e) { continue; }
    const st = stageText[job.stage] || job.stage || job.status;
    out.textContent = `建卷任务 ${jobId}：${st}（已等待 ${(i+1)*3}s）`;
    if (job.status === 'DONE') {
      const v = job.volume || {};
      okay(`建卷完成：${v.letter}: / ${((v.capacity_bytes||0)/1024/1024/1024).toFixed(0)} GB，已加入卷池。`);
      setTimeout(()=>location.reload(), 1000);
      return;
    }
    if (job.status === 'FAILED') {
      fail('建卷失败：' + (job.error || '未知错误'));
      $('volOut').textContent = '';
      return;
    }
  }
  out.textContent = '任务仍在后台执行，可稍后刷新本页查看结果。';
}
</script>
</body>
</html>
"""


def _gb(value: int | float | None) -> str:
    if not value:
        return "0.00"
    return f"{value / (1024 ** 3):.2f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _volume_rows(pool_status: dict[str, Any]) -> str:
    """卷池表格。每个卷的实际状态、容量、剩余均为当场实测。"""
    volumes = (pool_status or {}).get("volumes") or []
    if not volumes:
        return "<div class='foot'>卷池为空。</div>"
    rows = []
    for v in volumes:
        state = v.get("state") or "OFFLINE"
        badge = {"ONLINE": "ok", "READONLY": "warn", "OFFLINE": "bad"}.get(state, "")
        role = "主卷" if v.get("primary") else "数据卷"
        flags = []
        if not v.get("enabled"):
            flags.append("已停用")
        if v.get("primary"):
            flags.append("不可卸载")
        state_text = f"{state} / {'只读' if state == 'READONLY' else ('可写' if v.get('writable') else '不可写')}"
        actions = []
        vid = v.get("id", "")
        if not v.get("primary"):
            if state == "ONLINE":
                actions.append(f"<button type='button' class='vact' data-act='unmount' data-id='{vid}'>卸载卷</button>")
            else:
                actions.append(f"<button type='button' class='vact' data-act='mount' data-id='{vid}'>挂载卷</button>")
            if v.get("enabled"):
                actions.append(f"<button type='button' class='vact danger' data-act='disable' data-id='{vid}'>停止接收</button>")
            else:
                actions.append(f"<button type='button' class='vact' data-act='enable' data-id='{vid}'>恢复接收</button>")
            actions.append(f"<button type='button' class='vact' data-act='remove' data-id='{vid}'>移出池</button>")
        rows.append(
            f"<tr><td class='mono'>{vid}</td><td>{v.get('label') or '—'}<br>"
            f"<span class='mini'>{role}{(' / ' + ' / '.join(flags)) if flags else ''}</span></td>"
            f"<td class='mono'>{v.get('letter') or '—'}:</td>"
            f"<td><span class='pill {badge}'>{state_text}</span></td>"
            f"<td>{_gb(v.get('capacity_bytes'))} GB</td><td>{_gb(v.get('free_bytes'))} GB</td>"
            f"<td>{v.get('priority')}</td>"
            f"<td class='acts'>{' '.join(actions) or '—'}</td></tr>")
    total_cap = (pool_status or {}).get("total_capacity_bytes") or 0
    total_free = (pool_status or {}).get("total_free_bytes") or 0
    return ("<table><thead><tr><th>卷 ID</th><th>标签</th><th>盘符</th><th>状态</th>"
            "<th>容量（实测）</th><th>剩余（实测）</th><th>优先级</th><th>操作</th></tr></thead><tbody>"
            + "".join(rows)
            + f"</tbody><tfoot><tr><td colspan='4'>合计（在线且可写）</td>"
              f"<td>{_gb(total_cap)} GB</td><td>{_gb(total_free)} GB</td><td colspan='2'></td></tr></tfoot></table>")


def render_html(status: dict[str, Any], areas: dict[str, Any], generated_at: str,
                pool_status: dict[str, Any] | None = None) -> str:
    volume = status.get("volume") or {}
    quota = status.get("quota_bytes")
    used = status.get("sandbox_used_bytes") or 0
    ratio = status.get("quota_used_ratio")
    over = status.get("over_quota")
    mounted = bool(volume.get("mounted"))

    ratio_pct = (ratio or 0) * 100
    bar_cls = "hot" if (ratio or 0) >= 0.9 else ""

    cards = f"""
<div class="cards">
  <div class="card"><div class="k">卷容量（实测）</div>
    <div class="v">{_gb(volume.get('capacity_bytes'))} GB</div>
    <div class="s">{"已挂载" if mounted else "未挂载 / 不可读"}</div></div>
  <div class="card"><div class="k">沙箱已用（实测）</div>
    <div class="v">{_gb(used)} GB</div>
    <div class="s">incoming + projects + exports</div></div>
  <div class="card"><div class="k">策略上限</div>
    <div class="v">{_gb(quota)} GB</div>
    <div class="s">来源：{status.get('quota_source')}</div></div>
  <div class="card"><div class="k">剩余可用（策略）</div>
    <div class="v {'bad' if over else 'ok'}">{_gb(status.get('quota_remaining_bytes'))} GB</div>
    <div class="s">{'已超限' if over else '余量'}</div></div>
  <div class="card"><div class="k">卷剩余（实测）</div>
    <div class="v">{_gb(volume.get('free_bytes'))} GB</div>
    <div class="s">文件系统实际可写空间</div></div>
</div>
<div class="card" style="margin-bottom:18px">
  <div class="k">策略上限使用率</div>
  <div class="v {'bad' if over else ''}">{_pct(ratio)}</div>
  <div class="bar"><i class="{bar_cls}" style="width:{min(100, max(0, ratio_pct)):.1f}%"></i></div>
</div>
"""

    rows = []
    for name, info in (areas or {}).items():
        rows.append(f"<tr><td class='mono'>{name}</td><td>{info.get('files', 0)}</td>"
                    f"<td>{_gb(info.get('bytes'))} GB</td></tr>")
    areas_html = ("<h2>数据区占用（实测）</h2><table><thead><tr><th>区域</th><th>文件数</th>"
                  "<th>占用</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    quota_gb = int(round((quota or 0) / (1024 ** 3))) or 1

    html = (_TEMPLATE
            .replace("__CARDS__", cards)
            .replace("__VOLUMES__", "<h2>卷池总览</h2>" + _volume_rows(pool_status or {}))
            .replace("__VOLUME_ROWS__", _volume_rows(pool_status or {}))
            .replace("__AREAS__", areas_html)
            .replace("__QUOTA_GB__", str(quota_gb))
            .replace("__GENERATED__", generated_at))
    return html
