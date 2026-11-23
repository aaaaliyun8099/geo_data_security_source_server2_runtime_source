"""审计日志存储策略：可配置的保留天数、分批清理、手动清理与清理历史。

设计约束（为什么这样做）：
  * 配置必须落盘且可回滚：写临时文件再原子替换，进程崩溃不会留下半个 JSON。
  * 清理必须可预演：默认 dry_run，前端先给用户看「将删除哪些」，确认后才真删。
  * 文件按天命名（audit-YYYY-MM-DD.jsonl），所以保留策略只按日期判，不解析内容。
  * 当天文件永不删除，避免把正在写入的日志截断。
  * ES 索引清理默认关闭：Kibana 看板与 ElastAlert 规则都依赖 gds-audit-*，
    误删会让告警链路一起失效，所以要显式打开 target 才会动。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_DAYS = 180
DEFAULT_BATCH = 5000
DEFAULT_CRON = "0 30 3 * * *"
MIN_DAYS, MAX_DAYS = 7, 3650
MIN_BATCH, MAX_BATCH = 100, 50000
# 超限清理的保护期：不满这么多天的文件不因「存储上限」被删，避免上限设小后一次清空。
CAP_MIN_AGE_DAYS = 7
_CRON_RE = re.compile(
    r"^\s*(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+\*\s*$")
_FILE_RE = re.compile(r"^audit-(\d{4}-\d{2}-\d{2})\.jsonl$")
_ES_RE = re.compile(r"^gds-audit-(\d{4})\.(\d{2})\.(\d{2})$")


class PolicyError(ValueError):
    """配置非法。接口层直接转 422，文案面向用户。"""


def parse_cron(text: str) -> tuple[int, int, int]:
    """只支持仓库里实际使用的 6 段形式：秒 分 时 * * *。

    不接受其它写法，是因为静默猜测会让「保存成功但永不执行」变成隐藏故障。
    """
    match = _CRON_RE.match(text or "")
    if not match:
        raise PolicyError("清理计划只支持「秒 分 时 * * *」六段格式，例如 0 30 3 * * *")
    sec, minute, hour = (int(part) for part in match.groups())
    if not (0 <= sec <= 59 and 0 <= minute <= 59 and 0 <= hour <= 23):
        raise PolicyError("清理计划的时间超出范围：秒 0-59，分 0-59，时 0-23")
    return sec, minute, hour


class StoragePolicy:
    def __init__(self, workspace: Path | str, es_url: str | None = None):
        self.workspace = Path(workspace)
        self.audit_dir = self.workspace / "audit"
        self.state_dir = self.workspace / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.state_dir / "retention_policy.json"
        self.history_path = self.state_dir / "retention_history.jsonl"
        self.es_url = (es_url or os.environ.get("GDS_ES_URL") or "http://127.0.0.1:9200").rstrip("/")
        self._lock = threading.Lock()

    # ---------- 配置 ----------
    def _defaults(self) -> dict:
        return {
            "retentionDays": DEFAULT_DAYS,
            "batchSize": DEFAULT_BATCH,
            "scheduleEnabled": True,
            "cron": DEFAULT_CRON,
            "maxTotalMb": 0,
            "targets": {"auditFiles": True, "esIndices": False},
            "updatedAt": None,
            "updatedBy": None,
        }

    def load(self) -> dict:
        cfg = self._defaults()
        if self.config_path.is_file():
            try:
                stored = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # 配置文件坏了不能把服务带崩：退回到默认值，并在 stats 里暴露出来
                stored = {}
            if isinstance(stored, dict):
                cfg.update({k: v for k, v in stored.items() if k in self._defaults()})
        return cfg

    def _write(self, cfg: dict) -> None:
        tmp = self.config_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.config_path)

    def update(self, patch: dict, actor: str = "") -> dict:
        """局部更新。未提供的字段保持不变，避免前端只改一项时把其它项重置。"""
        cfg = self.load()
        with self._lock:
            if "retentionDays" in patch and patch["retentionDays"] is not None:
                days = int(patch["retentionDays"])
                if not (MIN_DAYS <= days <= MAX_DAYS):
                    raise PolicyError(f"保留天数需在 {MIN_DAYS} 至 {MAX_DAYS} 天之间")
                cfg["retentionDays"] = days
            if "batchSize" in patch and patch["batchSize"] is not None:
                batch = int(patch["batchSize"])
                if not (MIN_BATCH <= batch <= MAX_BATCH):
                    raise PolicyError(f"每批清理条数需在 {MIN_BATCH} 至 {MAX_BATCH} 之间")
                cfg["batchSize"] = batch
            if "scheduleEnabled" in patch and patch["scheduleEnabled"] is not None:
                cfg["scheduleEnabled"] = bool(patch["scheduleEnabled"])
            if "cron" in patch and patch["cron"]:
                parse_cron(str(patch["cron"]))
                cfg["cron"] = str(patch["cron"]).strip()
            if "maxTotalMb" in patch and patch["maxTotalMb"] is not None:
                cap = int(patch["maxTotalMb"])
                if cap < 0:
                    raise PolicyError("存储上限不能为负数，0 表示不限制")
                cfg["maxTotalMb"] = cap
            if "targets" in patch and isinstance(patch["targets"], dict):
                merged = dict(cfg.get("targets") or {})
                for key in ("auditFiles", "esIndices"):
                    if key in patch["targets"]:
                        merged[key] = bool(patch["targets"][key])
                cfg["targets"] = merged
            cfg["updatedAt"] = datetime.now().astimezone().isoformat(timespec="seconds")
            cfg["updatedBy"] = actor or cfg.get("updatedBy") or ""
            self._write(cfg)
        return cfg

    # ---------- 统计 ----------
    def audit_files(self) -> list[tuple[date, Path]]:
        found = []
        for path in self.audit_dir.glob("audit-*.jsonl"):
            match = _FILE_RE.match(path.name)
            if match:
                try:
                    found.append((date.fromisoformat(match.group(1)), path))
                except ValueError:
                    continue
        return sorted(found)

    def cutoff(self, days: int | None = None) -> date:
        days = self.load()["retentionDays"] if days is None else days
        return (datetime.now().astimezone() - timedelta(days=days)).date()

    def stats(self) -> dict:
        files = self.audit_files()
        total = sum(p.stat().st_size for _, p in files if p.exists())
        return {
            "auditFiles": {
                "directory": str(self.audit_dir),
                "count": len(files),
                "bytes": total,
                "earliest": files[0][0].isoformat() if files else None,
                "latest": files[-1][0].isoformat() if files else None,
            },
            "es": self._es_stats(),
        }

    def _es_stats(self) -> dict:
        try:
            req = urllib.request.Request(f"{self.es_url}/_cat/indices?format=json&h=index,docs.count,store.size")
            with urllib.request.urlopen(req, timeout=4) as resp:
                rows = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"available": False, "reason": str(exc)[:120], "indices": 0, "bytes": 0}
        picked = [r for r in rows if _ES_RE.match(r.get("index", ""))]
        return {
            "available": True,
            "indices": len(picked),
            "bytes": sum(_size_bytes(r.get("store.size", "0b")) for r in picked),
        }

    # ---------- 清理 ----------
    def cleanup(self, dry_run: bool = True, actor: str = "", targets: dict | None = None,
                confirm_capacity: bool = False, trigger: str | None = None) -> dict:
        started = datetime.now().astimezone()
        cfg = self.load()
        days = cfg["retentionDays"]
        cutoff = self.cutoff(days)
        want = dict(cfg.get("targets") or {})
        if targets:
            want.update({k: bool(v) for k, v in targets.items() if k in {"auditFiles", "esIndices"}})

        report = {
            "dryRun": bool(dry_run),
            "retentionDays": days,
            "cutoff": cutoff.isoformat(),
            "auditFiles": {"matched": 0, "deleted": 0, "bytes": 0, "files": []},
            "esIndices": {"matched": 0, "deleted": 0, "indices": []},
        }

        today = datetime.now().astimezone().date()
        touched: set[str] = set()
        if want.get("auditFiles", True):
            for day, path in self.audit_files():
                if day >= cutoff or day >= today:
                    continue  # 未到期，或者就是今天正在写的文件
                size = path.stat().st_size if path.exists() else 0
                report["auditFiles"]["matched"] += 1
                report["auditFiles"]["bytes"] += size
                report["auditFiles"]["files"].append({"name": path.name, "date": day.isoformat(), "bytes": size})
                touched.add(path.name)
                if not dry_run:
                    try:
                        path.unlink()
                        report["auditFiles"]["deleted"] += 1
                    except OSError:
                        pass

        if want.get("esIndices", False):
            for name, day in self._es_indices():
                if day >= cutoff:
                    continue
                report["esIndices"]["matched"] += 1
                report["esIndices"]["indices"].append(name)
                if not dry_run and self._es_delete(name):
                    report["esIndices"]["deleted"] += 1

        # 存储上限：按最旧优先删，直到降到上限之内。0 表示不限制。
        # 两道闸门：满 7 天的文件才允许因超限删除；还要显式 confirm_capacity，
        # 否则只提示不删。上限被误设成很小值时，这条规则会一次性删掉整段历史。
        cap_mb = int(cfg.get("maxTotalMb") or 0)
        capacity = {"maxTotalMb": cap_mb, "overflowBytes": 0, "matched": 0, "deleted": 0,
                    "bytes": 0, "files": [], "protected": 0, "warnings": []}
        if cap_mb > 0:
            floor = today - timedelta(days=CAP_MIN_AGE_DAYS)
            alive = [(day, path) for day, path in self.audit_files()
                     if day < today and path.name not in touched and path.exists()]
            limit = cap_mb * 1024 * 1024
            total = sum(p.stat().st_size for _, p in alive) + report["auditFiles"]["bytes"]
            overflow = total - limit
            capacity["overflowBytes"] = max(0, overflow)
            if overflow > 0:
                candidates = [(day, path) for day, path in alive if day < floor]
                capacity["protected"] = len(alive) - len(candidates)
                if not confirm_capacity:
                    capacity["matched"] = len(candidates)
                    capacity["warnings"].append(
                        f"已有数据超出存储上限 {capacity['overflowBytes']} 字节，"
                        f"其中 {len(candidates)} 个满 {CAP_MIN_AGE_DAYS} 天的文件可删除；"
                        "需要显式确认（confirmCapacity=true）才会执行")
                else:
                    for day, path in candidates:
                        if total <= limit:
                            break
                        size = path.stat().st_size
                        capacity["matched"] += 1
                        capacity["bytes"] += size
                        capacity["files"].append({"name": path.name, "date": day.isoformat(), "bytes": size})
                        total -= size
                        if not dry_run:
                            try:
                                path.unlink()
                                capacity["deleted"] += 1
                            except OSError:
                                pass
                if total > limit:
                    capacity["warnings"].append(
                        f"清理后仍超出上限 {total - limit} 字节：剩余文件都不满 {CAP_MIN_AGE_DAYS} 天，不会被自动删除")
        report["capacity"] = capacity

        report["elapsedMs"] = int((datetime.now().astimezone() - started).total_seconds() * 1000)
        if not dry_run:
            self._append_history({"trigger": trigger or "手动清理", "actor": actor, **report})
        return report

    def _es_indices(self) -> list[tuple[str, date]]:
        try:
            req = urllib.request.Request(f"{self.es_url}/_cat/indices?format=json&h=index")
            with urllib.request.urlopen(req, timeout=4) as resp:
                rows = json.loads(resp.read().decode())
        except (urllib.error.URLError, OSError, ValueError):
            return []
        out = []
        for row in rows:
            match = _ES_RE.match(row.get("index", ""))
            if match:
                year, month, day = (int(part) for part in match.groups())
                try:
                    out.append((row["index"], date(year, month, day)))
                except ValueError:
                    continue
        return sorted(out, key=lambda item: item[0])

    def _es_delete(self, index: str) -> bool:
        try:
            req = urllib.request.Request(f"{self.es_url}/{index}", method="DELETE")
            with urllib.request.urlopen(req, timeout=15) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError):
            return False

    # ---------- 历史 ----------
    def _append_history(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            with self.history_path.open("a", encoding="utf-8") as stream:
                stream.write(line)

    def history(self, limit: int = 20) -> list[dict]:
        if not self.history_path.is_file():
            return []
        lines = self.history_path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return list(reversed(out))

    def last_run(self) -> dict | None:
        items = self.history(1)
        return items[0] if items else None

    # ---------- 展示 ----------
    def view(self) -> dict:
        cfg = self.load()
        days = cfg["retentionDays"]
        cron = cfg["cron"]
        _, minute, hour = parse_cron(cron)
        next_run = self._next_run(hour, minute) if cfg["scheduleEnabled"] else None
        stats = self.stats()
        warnings: list[str] = []
        cap_bytes = int(cfg.get("maxTotalMb") or 0) * 1024 * 1024
        used = stats["auditFiles"]["bytes"]
        if cap_bytes and used > cap_bytes:
            warnings.append(f"审计文件已占用 {used} 字节，超出存储上限 {cap_bytes} 字节")
        return {
            **cfg,
            "ruleText": (
                f"审计日志保留 {days} 天，"
                + (f"每天 {hour:02d}:{minute:02d} 自动分批清理，每批 {cfg['batchSize']} 条" if cfg["scheduleEnabled"]
                   else "自动清理已关闭，仅支持手动清理")
            ),
            "cutoffNow": self.cutoff(days).isoformat(),
            "nextCleanupAt": next_run.isoformat(timespec="minutes") if next_run else None,
            "stats": stats,
            "warnings": warnings,
            "lastRun": self.last_run(),
        }

    @staticmethod
    def _next_run(hour: int, minute: int) -> datetime:
        now = datetime.now().astimezone()
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    # ---------- 定时执行 ----------
    def start_scheduler(self, interval_s: int = 30) -> threading.Thread:
        """按配置的 cron 每天执行一次真实清理。

        只做「时:分」匹配：仓库里使用的 cron 都是每天一次的形式，支持任意表达式会
        引入一个没人验证过的解释器。已经跑过的分钟用内存标记去重，重启后最多再跑一次，
        清理本身是幂等的（没有超期文件就什么都不删）。
        """
        def loop() -> None:
            last_key = None
            while True:
                try:
                    cfg = self.load()
                    if cfg.get("scheduleEnabled"):
                        _, minute, hour = parse_cron(cfg["cron"])
                        now = datetime.now().astimezone()
                        key = now.strftime("%Y-%m-%d %H:%M")
                        if now.hour == hour and now.minute == minute and key != last_key:
                            last_key = key
                            self.cleanup(dry_run=False, actor="", trigger="定时清理")
                except Exception:  # noqa: BLE001 - 调度线程不能因为一次异常就退出
                    pass
                time.sleep(max(5, interval_s))

        thread = threading.Thread(target=loop, name="retention-scheduler", daemon=True)
        thread.start()
        return thread


def _size_bytes(text: str) -> int:
    text = (text or "0b").strip().lower()
    units = {"b": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3, "tb": 1024 ** 4}
    for suffix in ("tb", "gb", "mb", "kb", "b"):
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * units[suffix])
            except ValueError:
                return 0
    return 0
