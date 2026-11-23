"""server2 侧沙箱计算桥：把数据处理操作经 SandboxRunner 交给受限账户 gds_sbx 执行。

环境变量启用（见 server2.env）：
  GDS_SBX_PY        V:/geo_data_security/sandbox/runtime/py/python.exe   (沙箱解释器)
  GDS_SBX_RUNNER    D:/GeoDataSecurity/sandbox/bin/SandboxRunner.exe     (JobObject 运行器)
  GDS_SBX_WORKER    V:/geo_data_security/sandbox/worker/gds_sandbox_worker.py
  GDS_SBX_USER      gds_sbx
  GDS_SBX_PASSWORD  <由密钥文件注入>

未配置时全部回退为进程内直算（开发模式），保证旧环境不破坏。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _kill_process_tree(pid: int) -> None:
    """终止 pid 及其所有子孙进程。

    为什么不能只用 process.kill()：沙箱的实际计算发生在更深一层
    （SandboxRunner.exe → python worker → python trustmark runner），
    只杀直接子进程会把中间层变成孤儿，它继续吃满 CPU。
    实测孤儿曾继续运行 1 小时 52 分、消耗 16.8 万 CPU 秒。

    Windows 用 `taskkill /F /T`（/T 递归整棵树）；POSIX 用进程组 SIGKILL。
    两条路径都吞掉异常：收尾失败不能掩盖真正的超时错误。
    """
    if pid <= 0:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, timeout=30, check=False,
            )
        else:
            import signal
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    except Exception:  # noqa: BLE001 - 尽力而为
        pass


def _cfg() -> tuple[str, str, str, str]:
    py = os.environ.get("GDS_SBX_PY", "")
    runner = os.environ.get("GDS_SBX_RUNNER", "")
    worker = os.environ.get("GDS_SBX_WORKER", "")
    user = os.environ.get("GDS_SBX_USER", "gds_sbx")
    return py, runner, worker, user


def enabled() -> bool:
    py, runner, worker, _ = _cfg()
    return bool(py and runner and worker and Path(py).exists() and Path(runner).exists() and Path(worker).exists())


def run(op: str, *args: str, profile: str = "main", timeout_s: int = 3600) -> object:
    """在沙箱内执行白名单 op，返回 worker JSON 的 result；失败抛 RuntimeError。

    超时必须终止**整棵进程树**：调用链是 api(uvicorn) → SandboxRunner.exe → python(worker)
    → python(trustmark runner)。`subprocess.run(timeout=)` 只杀它直接创建的那个进程，
    孙进程会变成孤儿继续占满 CPU。实测有一次 `verify-carriers` 超时后孤儿又跑了 1 小时 52 分、
    烧掉 16.8 万 CPU 秒（28 核满载），前端所有请求一起变慢。
    所以这里用 Popen + Job Object（Windows）显式收尾整棵树。
    """
    py, runner, worker, user = _cfg()
    password = os.environ.get("GDS_SBX_PASSWORD", "")
    if not enabled() or not password:
        raise RuntimeError("sandbox compute not configured (GDS_SBX_*)")
    env = dict(os.environ)
    env["GDS_SBX_USER"] = user
    env["GDS_SBX_PASSWORD"] = password
    # Keep sandbox scratch (per-JPEG probes, temp downloads) inside the protected
    # workspace instead of the sandbox account's C: profile, which is small.
    scratch = os.environ.get("GDS_SBX_SCRATCH", "")
    if scratch:
        try:
            Path(scratch).mkdir(parents=True, exist_ok=True)
            env["TMP"] = env["TEMP"] = env["TMPDIR"] = scratch
        except OSError:
            pass
    # Force UTF-8 in the child process; worker output is ASCII JSON as a second
    # line of defense for Windows GBK console/code-page environments.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    command = [runner, py, worker, profile, op, *args]

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="backslashreplace", env=env,
        # 独立进程组，便于整组回收（POSIX）；Windows 走下面的进程树 kill。
        start_new_session=(os.name != "nt"),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process.pid)
        try:
            process.communicate(timeout=10)
        except Exception:  # noqa: BLE001 - 收尾尽力而为，不能挡住异常上抛
            pass
        raise RuntimeError(
            f"sandbox op {op} timed out after {timeout_s}s "
            f"(process tree {process.pid} terminated)") from None
    completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"sandbox op {op} failed (exit {completed.returncode}): {detail[-400:]}")
    last_line = ""
    for line in reversed(completed.stdout.splitlines()):
        if line.strip().startswith("{"):
            last_line = line.strip()
            break
    if not last_line:
        raise RuntimeError(f"sandbox op {op}: no JSON on stdout: {completed.stdout[-300:]!r}")
    payload = json.loads(last_line)
    if op == "tm_passthrough":  # TrustMark runner 的原始 JSON（如 {"ok":..}/{"mode":..}）
        return payload
    if not payload.get("ok"):
        raise RuntimeError(f"sandbox op {op} error: {payload.get('error')}")
    return payload.get("result")
