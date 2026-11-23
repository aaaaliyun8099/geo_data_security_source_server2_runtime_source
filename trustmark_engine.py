"""TrustMark 瓦片 GeoTIFF 引擎（主环境侧子进程桥）。

主项目 venv（numpy>=2）不能 import torch/trustmark，因此本引擎通过 subprocess
调用独立 trustmark venv 的 runner（`trustmark_geotiff_runner.py`）完成嵌入/提取。

runner 路径通过环境变量 GDS_TRUSTMARK_PYTHON / GDS_TRUSTMARK_RUNNER 配置，
便于部署到服务器时指向实际安装位置；缺省指向仓库 runtime/ 下独立 venv。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


class TrustMarkEngineUnavailableError(RuntimeError):
    pass


def default_trustmark_python() -> str:
    configured = os.environ.get("GDS_TRUSTMARK_PYTHON")
    if configured:
        return configured
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / ".trustmark-deploy-venv" / "Scripts" / "python.exe",
        repo_root / "runtime" / "trustmark_venv" / "Scripts" / "python.exe",
        repo_root / ".trustmark-deploy-venv" / "bin" / "python",
        repo_root / "runtime" / "trustmark_venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise TrustMarkEngineUnavailableError(
        "TrustMark venv python not found; set GDS_TRUSTMARK_PYTHON or create runtime/trustmark_venv"
    )


def default_runner_script() -> str:
    configured = os.environ.get("GDS_TRUSTMARK_RUNNER")
    if configured:
        return configured
    return str(Path(__file__).resolve().parent / "trustmark_geotiff_runner.py")


class TrustMarkGeoTIFFEngine:
    """Embed/extract GeoTIFF watermark by shelling out to the isolated TrustMark venv."""

    def __init__(self, python: str | None = None, runner: str | None = None, timeout: int | None = None):
        self.python = python or default_trustmark_python()
        self.runner = runner or default_runner_script()
        # A full 3 GB DOM embeds 3,834 carriers; measured 211 ms/carrier (~13.5 min) on this host,
        # so the old fixed 1800 s ceiling left no margin on a busy box. Per-carrier verification of
        # the same product adds ~4 min. Override with GDS_TRUSTMARK_TIMEOUT (seconds).
        self.timeout = int(timeout or os.environ.get("GDS_TRUSTMARK_TIMEOUT", "1800") or 1800)
        if not Path(self.python).exists():
            raise TrustMarkEngineUnavailableError(f"trustmark python not found: {self.python}")
        if not Path(self.runner).exists():
            raise TrustMarkEngineUnavailableError(f"trustmark runner not found: {self.runner}")
        try:
            probe = subprocess.run(
                [self.python, "-c", "import sys; print(sys.version_info[:2])"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                check=False,
            )
        except OSError as exc:
            raise TrustMarkEngineUnavailableError(f"trustmark python cannot start: {self.python}") from exc
        if probe.returncode != 0:
            detail = (probe.stderr or probe.stdout or "").strip()
            raise TrustMarkEngineUnavailableError(
                f"trustmark python cannot start: {self.python}; {detail}"
            )

    def _call(self, arguments: list[str], timeout: int | None = None) -> dict:
        sbx_runner = os.environ.get("GDS_SBX_RUNNER", "")
        sbx_py = os.environ.get("GDS_SBX_PY", "")
        sbx_worker = os.environ.get("GDS_SBX_WORKER", "")
        use_sandbox = bool(sbx_runner and sbx_py and sbx_worker)
        if use_sandbox:  # 数据处理在受控沙箱（受限账户 gds_sbx + Job Object + 断外联）内执行
            command = [sbx_runner, sbx_py, sbx_worker, "tm", "tm_passthrough", self.runner, *arguments]
            env = dict(os.environ)
            env.setdefault("GDS_SBX_USER", "gds_sbx")
            env.setdefault("GDS_SBX_PASSWORD", "")
            kwargs = {"env": env}
        else:
            command = [self.python, self.runner, *arguments]
            kwargs = {}
        # 与 sandbox_compute.run 同样的问题：subprocess.run(timeout=) 只杀直接子进程，
        # SandboxRunner 拉起的 python 孙进程会变孤儿并继续占满 CPU。用 Popen + 进程树收尾。
        from .sandbox_compute import _kill_process_tree
        effective_timeout = int(timeout) if timeout else self.timeout

        def _exec(cmd, kw):
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                start_new_session=(os.name != "nt"), **kw)
            try:
                stdout, stderr = process.communicate(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                _kill_process_tree(process.pid)
                try:
                    process.communicate(timeout=10)
                except Exception:  # noqa: BLE001
                    pass
                raise RuntimeError(
                    f"TrustMark runner timed out after {effective_timeout}s "
                    f"(process tree {process.pid} terminated): {cmd}") from None
            completed = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(f"TrustMark runner failed (exit {completed.returncode}): {detail}")
            try:
                return json.loads(completed.stdout.strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise RuntimeError(f"TrustMark runner returned non-JSON output: {completed.stdout!r}") from exc

        try:
            return _exec(command, kwargs)
        except RuntimeError as exc:
            # 0xC0000142（3221225794 / -1073741502）= DLL 初始化失败：受限账户 gds_sbx
            # 起不动 torch —— uv 管理的 base python 装在 C:\Users\Administrator 下，
            # 受限账户读不到，加载 torch 的 DLL 时直接失败。
            # 只是解码像素、不执行任何载荷代码，退回服务账户直接跑；嵌入路径仍走沙箱。
            msg = str(exc)
            soft = ("exit -1073741502" in msg or "exit 3221225794" in msg
                    or "exit -1073741819" in msg)
            if not (use_sandbox and soft):
                raise
            return _exec([self.python, self.runner, *arguments], {})

    def embed(self, source: Path, destination: Path, watermark: int) -> int:
        if not 0 <= watermark <= 0xFFFFFFFF:
            raise ValueError("watermark must be an unsigned 32-bit integer")
        result = self._call(["embed", str(source), str(destination), format(watermark, "08X")])
        if "error" in result:
            raise RuntimeError(f"TrustMark embed failed: {result['error']}")
        return int(result.get("tiles", 1))

    def embed_partial(self, source: Path, destination: Path, watermark: int) -> dict:
        if not 0 <= watermark <= 0xFFFFFFFF:
            raise ValueError("watermark must be an unsigned 32-bit integer")
        result = self._call(["partial-embed", str(source), str(destination), format(watermark, "08X")])
        if "error" in result:
            raise RuntimeError(f"TrustMark partial embed failed: {result['error']}")
        return result

    def extract(self, source: Path, watermark: int) -> dict:
        result = self._call(["extract", str(source), format(watermark, "08X")])
        if "error" in result:
            raise RuntimeError(f"TrustMark extract failed: {result['error']}")
        return result

    def extract_carriers(self, source: Path, watermark: int, workers: int | None = None) -> dict:
        """Decode EVERY carrier of a tiled product (whole-image per-carrier coverage).

        Runs the same sandboxed runner entry point as embed/extract, so the work still happens as
        ``gds_sbx`` inside the sandbox. Returns ``samples``/``matched`` over all carriers plus the
        timing and any failed carrier origins.

        超时单独设置（默认 900s，`GDS_TRUSTMARK_CARRIER_TIMEOUT`）：逐载体全量解码的耗时
        与载体数成正比，大图会远超普通提取。此前它继承 1800s 的通用超时，实测有任务跑满
        1 小时 52 分仍未返回、把 28 核吃满，前端所有接口跟着变慢。
        宁可超时返回"未通过"，也不能让一个校验请求无限期霸占 CPU。
        """
        arguments = ["verify-carriers", str(source), format(watermark, "08X")]
        if workers:
            arguments.append(str(workers))
        carrier_timeout = os.environ.get("GDS_TRUSTMARK_CARRIER_TIMEOUT", "")
        result = self._call(arguments, timeout=int(carrier_timeout) if carrier_timeout.isdigit() else None)
        if "error" in result:
            raise RuntimeError(f"TrustMark per-carrier verification failed: {result['error']}")
        return result

    def embed_image(self, source: Path, destination: Path, watermark: int) -> int:
        """Embed single-code-word watermark into an image/texture file."""
        if not 0 <= watermark <= 0xFFFFFFFF:
            raise ValueError("watermark must be an unsigned 32-bit integer")
        result = self._call(["image-embed", str(source), str(destination), format(watermark, "08X")])
        if "error" in result:
            raise RuntimeError(f"TrustMark image embed failed: {result['error']}")
        return 1

    def extract_image(self, source: Path, watermark: int) -> dict:
        result = self._call(["image-extract", str(source), format(watermark, "08X")])
        if "error" in result:
            raise RuntimeError(f"TrustMark image extract failed: {result['error']}")
        return result

    def extract_blind(self, source: Path, watermark: int, max_windows: int = 800) -> dict:
        """Extract after metadata removal using pixel-only sliding-window search."""
        result = self._call(["blind-extract", str(source), format(watermark, "08X"), str(max_windows)])
        if "error" in result:
            raise RuntimeError(f"TrustMark blind extract failed: {result['error']}")
        return result
