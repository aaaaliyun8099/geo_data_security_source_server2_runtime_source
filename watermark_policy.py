"""二次嵌入策略：输入已带 _wm 时是覆盖还是拒绝。

本平台导出的成果再导入是正常业务动作（同一份数据换项目或换借用人重新签发），一刀切拒绝会让
整条链路走不通。默认覆盖为本次标记，并记录旧标记摘要以保留「上一手」痕迹；
GDS_STRICT_REEMBED=1 可恢复「拒绝二次嵌入」的严格取证策略。
"""
from __future__ import annotations

import hashlib
import os

STRICT_ENV = "GDS_STRICT_REEMBED"


def reembed_allowed() -> bool:
    return os.environ.get(STRICT_ENV, "0").strip().lower() not in {"1", "true", "yes", "on"}


def marker_digest(values) -> str:
    """旧水印值的摘要：只留指纹，不落明文。"""
    digest = hashlib.sha256()
    for value in values:
        digest.update((value or "").encode("utf-8", "replace"))
        digest.update(b"\x00")
    return digest.hexdigest()
