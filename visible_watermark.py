"""可见水印（VISIBLE / HYBRID）与不可见水印叠加。

设计要点
--------
* **VISIBLE**：只在像素域叠加可见文字（半透明、带描边），不做不可见水印。
* **HYBRID**：先嵌不可见水印（FORENSIC 路径），再叠加可见文字——两者共存。
* 叠加位置按图像尺寸自适应，**不覆盖中心工作区**（默认右下角），保证成果可用。
* 所有函数返回结构化报告（位置、字号、不透明度、覆盖比例），供 manifest 与前端展示。
* 可见水印是**有损**的（改了像素），因此 PSNR 有限，必须如实报告。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

__all__ = [
    "WATERMARK_MODES",
    "VisibleWatermarkError",
    "overlay_visible_text",
    "visible_watermark_report",
]

#: 支持的水印模式。FORENSIC = 仅不可见；VISIBLE = 仅可见；HYBRID = 两者叠加。
WATERMARK_MODES = ("FORENSIC", "VISIBLE", "HYBRID")

#: 叠加九宫格位置。
ANCHOR_POSITIONS = ("bottom-right", "bottom-left", "top-right", "top-left", "center")


class VisibleWatermarkError(ValueError):
    """可见水印参数非法。"""


def _load_font(size: int):
    from PIL import ImageFont

    # Windows 常见中文字体，按可用性回退；全部失败则用位图默认字体。
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",     # 微软雅黑
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",   # 黑体
        r"C:\Windows\Fonts\simsun.ttc",   # 宋体
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_bbox(draw, text: str, font) -> tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font, stroke_width=2)


def overlay_visible_text(image, text: str, *, opacity: float = 128,
                         anchor: str = "bottom-right", font_scale: float = 0.045,
                         margin_scale: float = 0.012, color=(255, 255, 255),
                         stroke_color=(0, 0, 0)) -> dict[str, Any]:
    """在 PIL Image 上叠加可见文字水印，返回报告。

    ``numpy`` 数组覆盖比例可以作为「可见水印副作用的量化」交给前端展示。
    """
    from PIL import Image, ImageDraw

    if not text or not text.strip():
        raise VisibleWatermarkError("visible watermark text must not be empty")
    if not 0 < opacity <= 255:
        raise VisibleWatermarkError("opacity must be in (0, 255]")
    if anchor not in ANCHOR_POSITIONS:
        raise VisibleWatermarkError(f"unsupported anchor: {anchor}; use one of {list(ANCHOR_POSITIONS)}")
    if not 0 < font_scale < 0.5:
        raise VisibleWatermarkError("font_scale must be in (0, 0.5)")

    base = image.convert("RGBA") if image.mode != "RGBA" else image
    width, height = base.size
    font_size = max(12, int(min(width, height) * font_scale))
    font = _load_font(font_size)
    margin = max(8, int(min(width, height) * margin_scale))

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    left, top, right, bottom = _text_bbox(draw, text, font)
    text_w, text_h = right - left, bottom - top

    if anchor == "bottom-right":
        x, y = width - text_w - margin, height - text_h - margin
    elif anchor == "bottom-left":
        x, y = margin, height - text_h - margin
    elif anchor == "top-right":
        x, y = width - text_w - margin, margin
    elif anchor == "top-left":
        x, y = margin, margin
    else:  # center
        x, y = (width - text_w) // 2, (height - text_h) // 2

    draw.text((x - left, y - top), text, font=font, fill=(*color, int(opacity)),
              stroke_width=2, stroke_fill=(*stroke_color, int(opacity)))
    composed = Image.alpha_composite(base, layer)

    covered = (text_w + 4) * (text_h + 4)
    return {
        "mode": "VISIBLE",
        "text": text,
        "anchor": anchor,
        "position": {"x": int(x), "y": int(y)},
        "font_size": font_size,
        "opacity": int(opacity),
        "image_size": [width, height],
        "text_box": {"width": int(text_w), "height": int(text_h)},
        "coverage_ratio": round(covered / float(width * height), 8),
        "lossy": True,
        "_image": composed,
    }


def visible_watermark_report(report: dict[str, Any]) -> dict[str, Any]:
    """剥掉内部 PIL 对象，得到可 JSON 序列化的报告。"""
    return {k: v for k, v in report.items() if not k.startswith("_")}


def overlay_visible_on_raster(source: Path, destination: Path, text: str, **kwargs) -> dict[str, Any]:
    """对 GeoTIFF/DEM 叠加可见水印（保留地理参考），返回报告。"""
    import numpy as np  # noqa: F401
    import rasterio
    from PIL import Image
    from rasterio.enums import ColorInterp

    with rasterio.open(source) as dataset:
        profile = dataset.profile.copy()
        bands = dataset.count
        data = dataset.read()

    # 3/4 波段按 RGB(A) 处理；单波段转灰度后叠字再写回。
    if bands >= 3:
        rgb = np.transpose(data[:3], (1, 2, 0)).astype("uint8")
        image = Image.fromarray(rgb, "RGB")
    else:
        gray = data[0].astype("uint8") if data[0].dtype == "uint8" else _normalize_to_uint8(data[0])
        image = Image.fromarray(gray, "L").convert("RGB")

    result = overlay_visible_text(image, text, **kwargs)
    composed = result.pop("_image").convert("RGB")
    overlay = np.asarray(composed)

    out = data.copy()
    if bands >= 3:
        out[:3] = np.transpose(overlay, (2, 0, 1))
    else:
        out[0] = np.asarray(composed.convert("L"))

    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as sink:
        sink.write(out)

    before = data[:3] if bands >= 3 else data[:1]
    after = out[:3] if bands >= 3 else out[:1]
    result["psnr_db"] = psnr(before, after)
    result["lossy"] = True
    result["_arrays"] = (before, after)
    return result


def _normalize_to_uint8(array):
    import numpy as np

    low, high = float(np.min(array)), float(np.max(array))
    if high <= low:
        return np.zeros(array.shape, dtype="uint8")
    return ((array.astype("float64") - low) / (high - low) * 255.0).astype("uint8")


def psnr(before, after) -> float:
    """两幅同形数组的 PSNR（dB）；完全相同返回 inf。"""
    import numpy as np

    a = before.astype("float64")
    b = after.astype("float64")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return math.inf
    return round(10 * math.log10((255.0 ** 2) / mse), 3)
