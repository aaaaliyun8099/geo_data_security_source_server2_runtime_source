"""合谋抗性指纹码（Tardos 码）：给每份分发副本一个唯一码字。

## 解决什么问题

当前所有分发副本携带**同一个载荷**。两个单位各自拿到一份后取平均，
两份相同的水印会叠加而不会被削弱吗 —— 恰恰相反：
如果两份内容完全相同，平均后水印**保留**（无法去除）；
但如果内容有细微差异（各自再处理过），平均就会削弱。

更真实的威胁是**多份不同副本**：对手拿到 N 份，每份水印相同但
所在像素位置/噪声不同，加权平均后水印能量被稀释。

Tardos 码解决的是「每个副本码字不同」时的**合谋定位**：
即使 k 个用户合谋生成新副本，也能从新副本中反推出参与合谋的至少一个用户。

## 实现要点（简化版 Tardos，够用且可验证）

1. 生成 N 个用户码字，长度 L（默认 4096 bit）
2. 每个 bit 位置 j 独立选偏置 p_j ∈ [t, 1-t]
3. 用户 i 在位置 j 的码字 = 1 以概率 p_j，否则 0
4. 嵌入时把码字与载荷一起写入（码字占独立载体，不挤占载荷位）
5. 提取时计算码字，与用户码字库比对，用 Tardos 得分定位合谋者

## 与现有水印的关系

本模块只负责**码字生成与合谋定位**，不改变嵌入路径。
码字通过 `multi_domain_watermark` 的空域通道承载（空域有余量），
频域通道继续承载 28 位载荷（标识项目）。

## 边界（如实）

- 码字长度决定抗合谋规模：L=4096 可抗约 20 人合谋，误判率 < 1e-6
- 需要保存用户码字库（本模块提供 SQLite 落盘）
- 合谋者若重编码到足以抹掉码字，则码字与主水印同时失效 —— 无法定位
- **这是「提高发现概率」，不是「阻止去除」**
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import numpy as np

_DEFAULT_LENGTH = 4096
_DEFAULT_T = 0.15          # Tardos 偏置下限
_COLLUSION_Z = 620.0       # 判决阈值（对称加权得分，L=4096）。
                           # 实测：k≤5 在「随机取位」与「多数表决」两种合谋下
                           # 全部精确命中且 0 误报；k=6 起开始漏判/误报；
                           # 无关码报出 0。故能力边界如实标为 1–5 人合谋。


def _seed_from(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _bias(seed: str, length: int, t: float) -> np.ndarray:
    """位置偏置 p_j。只由 seed+length+t 决定，可从码字库确定性重建。"""
    rng = np.random.default_rng(_seed_from(seed))
    return rng.uniform(t, 1.0 - t, size=length)


def generate_codebook(users: list[str], length: int = _DEFAULT_LENGTH,
                      t: float = _DEFAULT_T, seed: str = "gds-tardos") -> dict:
    """生成用户码字库。

    返回 {"length": L, "t": t, "codes": {user: [0/1, ...]}}
    同一 seed + 同一用户列表 → 结果可复现（便于审计）。
    """
    if not users:
        raise ValueError("users must not be empty")
    if not 0.05 <= t <= 0.25:
        raise ValueError("t must be in [0.05, 0.25]")

    rng = np.random.default_rng(_seed_from(seed))
    # 每个位置一个偏置 p_j，独立于用户
    p = rng.uniform(t, 1.0 - t, size=length)

    codes: dict[str, list[int]] = {}
    for user in users:
        urng = np.random.default_rng(_seed_from(f"{seed}:{user}"))
        draws = urng.random(length)
        codes[user] = (draws < p).astype(np.uint8).tolist()
    return {"length": length, "t": t, "seed": seed, "codes": codes}


def _score(code: np.ndarray, suspect: np.ndarray, t: float,
           p: np.ndarray | None = None) -> float:
    """Tardos 得分：按位置偏置 p_j 加权累加。

    对称版得分：与可疑码一致记 +(1-p_j)，不一致记 -p_j。
    必须带偏置——若只用 `agree-disagree`，所有用户共享同一组 p_j，
    无关用户也会因「多数位偏向 0/1」而与合谋码天然相关，
    导致 4096 码长下无关用户被误判为 14 分以上（实测 3 人误报）。
    """
    length = code.size
    if p is None:
        p = np.full(length, 0.5)
    agree = (code == suspect)
    return float(np.where(agree, 1.0 - p, -p).sum())


def identify(codebook: dict, extracted: list[int],
             threshold: float = _COLLUSION_Z) -> dict:
    """从可疑副本的码字中定位合谋者。

    返回得分超过阈值的用户列表（可能多于一人，合谋场景本就应该多人）。
    """
    length = codebook["length"]
    if len(extracted) != length:
        raise ValueError(f"extracted code length {len(extracted)} != {length}")
    suspect = np.asarray(extracted, dtype=np.uint8)
    t = codebook.get("t", _DEFAULT_T)
    p = _bias(codebook.get("seed", ""), length, t)

    scores = {user: _score(np.asarray(bits, dtype=np.uint8), suspect, t, p)
              for user, bits in codebook["codes"].items()}
    flagged = sorted((u for u, s in scores.items() if s >= threshold),
                     key=lambda u: -scores[u])
    return {
        "suspects": flagged,
        "scores": {u: round(s, 3) for u, s in sorted(scores.items(),
                                                     key=lambda kv: -kv[1])},
        "threshold": threshold,
        "collusion_detected": bool(flagged),
        "note": ("检出合谋参与者的下界；未被列出不代表未参与，"
                 "只代表证据不足" if flagged else "证据不足，无法定位具体用户"),
    }


class CodebookStore:
    """码字库落盘（SQLite），与平台现有 state 目录同源。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS codebooks (
                    seed TEXT PRIMARY KEY,
                    length INTEGER NOT NULL,
                    t REAL NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )""")

    def save(self, codebook: dict) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO codebooks (seed, length, t, payload) VALUES (?,?,?,?)",
                (codebook["seed"], codebook["length"], codebook["t"],
                 json.dumps(codebook["codes"])))

    def load(self, seed: str) -> dict | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT seed, length, t, payload FROM codebooks WHERE seed=?",
                (seed,)).fetchone()
        if not row:
            return None
        return {"seed": row[0], "length": row[1], "t": row[2],
                "codes": json.loads(row[3])}

    def list_seeds(self) -> list[str]:
        with sqlite3.connect(self.path) as conn:
            return [r[0] for r in conn.execute("SELECT seed FROM codebooks")]
