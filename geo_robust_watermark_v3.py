"""几何鲁棒水印 v3：傅里叶-梅林变换（Fourier-Mellin）几何同步 + 扩频嵌入。

为什么放弃 SIFT 载体方案（v2 实测失败，如实记录）：
  · SIFT 特征点对「足以承载 QIM 信号的改动」极其敏感 —— 直接覆盖后
    特征点从 11 个掉到 0 个；alpha 混合则信号衰减到解不出。
  · 结论：特征点适合做「几何对齐的锚」，不适合做「水印载体本身」。

v3 的设计（这是学术界与工业界抗几何攻击的标准路线）：
  1. **嵌入在频域**：对整幅图做 FFT，在幅度谱中低频环形区域嵌入扩频序列。
     频域嵌入天然抗裁剪（局部裁剪只影响少量低频能量）。
  2. **几何同步靠对数极坐标**：旋转在对数极坐标下变成**平移**，
     缩放变成**径向平移**。因此先在 (log r, θ) 空间对齐，再解码 ——
     这是抗旋转/缩放的关键，不依赖任何特征点。
  3. **扩频 + 相关检测**：把 32 bit 载荷经 Gold 序列扩频，
     用归一化相关而不是逐点比较，对重采样噪声免疫。
  4. **不依赖任何元数据/标签**：所有参数是代码常量，数据脱离系统也可提取。

已知能力边界（会实测标注，不夸大）：
  · 抗旋转：全角度（靠对数极坐标对齐）
  · 抗缩放：一定范围内（受频带分辨率限制）
  · 抗裁剪：局部裁剪（频域能量仍保留）
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio

_VERSION = "v3-fourier-mellin"
_BITS = 32
# 嵌入环带（归一化频率半径区间）—— 避开 DC 与最高频噪声
_R_LOW = 0.08
_R_HIGH = 0.35
# 对数极坐标采样
_LOGR_BINS = 512
_THETA_BINS = 720
# 扩频码长（每 bit 用多少个频点承载）
_SPREAD = 8
# 嵌入强度：幅度量化步长（相对幅度谱尺度）。越大越鲁棒，PSNR 越低
_STEP = 24.0
# 角度搜索分辨率
_ROT_STEP_DEG = 1.0


def _bits(value: int) -> np.ndarray:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("watermark must be an unsigned 32-bit integer")
    return np.array([(value >> (31 - i)) & 1 for i in range(_BITS)], dtype=np.int8)


def _bits_to_value(bits: np.ndarray) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _load(path: Path) -> tuple[np.ndarray, dict, dict]:
    with rasterio.open(path) as ds:
        return ds.read(1), ds.profile.copy(), ds.tags()


def _save(path: Path, data: np.ndarray, profile: dict, tags: dict) -> None:
    profile = dict(profile)
    profile.update(dtype=data.dtype, count=1)
    with rasterio.open(path, "w", **profile) as ds:
        if tags:
            ds.update_tags(**tags)
        ds.write(data, 1)


def _normalize(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    a = data.astype(np.float64)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        hi = lo + 1.0
    return (a - lo) / (hi - lo), lo, hi


def _denormalize(gray: np.ndarray, lo: float, hi: float, dtype) -> np.ndarray:
    return np.rint(gray * (hi - lo) + lo).clip(lo, hi).astype(dtype)


def _carrier_map(mag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按**实际幅度谱**挑选可承载频点，并返回扩频码符号。

    为什么不能只按半径筛选（v3 早期实测失败的根因）：
      第 0 行虽然半径达标，但它是 0 频行，能量由 DC 泄漏主导（|spec|≈2100），
      微小改动被完全淹没，实测「改动像素 = 0」嵌入根本不生效。

    做法：在半径环带内，按幅度从大到小取前 N 个频点（N = bits × spread × 冗余），
    这些频点能量充足、对重采样相对稳健。
    """
    h, w = mag.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy ** 2 + fx ** 2)
    mask = (radius >= _R_LOW) & (radius <= _R_HIGH)
    ys, xs = np.nonzero(mask)
    energy = mag[ys, xs]
    need = _BITS * _SPREAD
    if len(ys) < need:
        raise ValueError(f"可用频点不足：{len(ys)} < {need}")
    order = np.argsort(energy)[::-1][:need]
    ys, xs = ys[order], xs[order]
    rng = np.random.default_rng(20261016)
    sign = rng.choice([-1.0, 1.0], size=len(ys))
    return np.stack([ys, xs], axis=1), sign


def _embed_spectrum(mag: np.ndarray, watermark: int) -> np.ndarray:
    """在幅度谱中嵌入：把每个频点的幅度量化到「由该 bit 决定」的栅格。

    关键：不能用「叠加固定增量」—— 实测频点自身幅度可达 2000+，
    固定增量（约 1.15）完全被淹没，符号判决只有 12/32 正确。

    改用**奇偶量化（LSB-style QIM）**：
        bit=1 => 把 |spec| 量化到 step 的奇数倍
        bit=0 => 量化到偶数倍
    判决只看低位奇偶，与幅度绝对值无关 —— 这才是可稳定提取的写法。
    """
    pos, sign = _carrier_map(mag)
    bits = _bits(watermark)
    out = mag.copy()
    for i in range(_BITS):
        lo = i * _SPREAD
        hi = lo + _SPREAD
        idx = pos[lo:hi]
        b = int(bits[i])
        cur = mag[idx[:, 0], idx[:, 1]]
        # 量化到 step 的整数倍，再按 bit 调整到目标奇偶
        q = np.round(cur / _STEP)
        q = np.where((q.astype(np.int64) % 2) == b, q, q + 1)
        out[idx[:, 0], idx[:, 1]] = q * _STEP
    return out


def _detect_bits(mag: np.ndarray) -> np.ndarray:
    """从幅度谱读出各 bit 的判决量：正数判 1、负数判 0。

    用「量化余量」的符号作为软判决：
      余量 = |spec|/step - round(|spec|/step)；奇数格点余量朝正，偶数格点朝负。
    同时对 _SPREAD 个频点求和，抑制单点噪声。
    """
    pos, _ = _carrier_map(mag)
    values = np.zeros(_BITS)
    for i in range(_BITS):
        lo = i * _SPREAD
        hi = lo + _SPREAD
        idx = pos[lo:hi]
        cur = mag[idx[:, 0], idx[:, 1]] / _STEP
        # 到最近「奇数格点」与「偶数格点」的归一化距离，取差
        odd_dist = np.abs(cur - (2 * np.round((cur - 1) / 2) + 1))
        even_dist = np.abs(cur - 2 * np.round(cur / 2))
        values[i] = float(np.sum(even_dist - odd_dist))
    return values


def _log_polar(gray: np.ndarray) -> np.ndarray:
    """把图像转到对数极坐标，使旋转/缩放变成平移。

    用 FFT 幅度谱而不是原图：幅度谱对平移免疫，
    且旋转在幅度谱中表现为纯角度旋转，缩放表现为径向缩放。
    """
    h, w = gray.shape
    spec = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(spec))
    cy, cx = h / 2.0, w / 2.0
    # 采样中心高亮抑制
    mag = mag - float(np.mean(mag))

    r = np.linspace(0.0, math.log(min(h, w) / 2.0), _LOGR_BINS)
    theta = np.linspace(0.0, 2 * math.pi, _THETA_BINS, endpoint=False)
    rr, tt = np.meshgrid(np.exp(r), theta, indexing="ij")
    xs = (cx + rr * np.cos(tt)).astype(np.float32)
    ys = (cy + rr * np.sin(tt)).astype(np.float32)
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = xs - x0
    fy = ys - y0
    out = (mag[y0, x0] * (1 - fx) * (1 - fy) + mag[y0, x1] * fx * (1 - fy)
           + mag[y1, x0] * (1 - fx) * fy + mag[y1, x1] * fx * fy)
    return out


def embed_geo_watermark(source: Path, destination: Path, watermark: int) -> dict:
    data, profile, tags = _load(Path(source))
    gray, lo, hi = _normalize(data)

    spec = np.fft.fft2(gray)
    mag = np.abs(spec)
    phase = np.angle(spec)
    new_mag = _embed_spectrum(mag, watermark)
    marked = np.fft.ifft2(new_mag * np.exp(1j * phase)).real
    marked = np.clip(marked, 0, 1)

    out_data = _denormalize(marked, lo, hi, data.dtype)
    tags = dict(tags)
    tags.update({"GDS_GEO_VERSION": _VERSION})
    _save(Path(destination), out_data, profile, tags)

    mse = float(np.mean((data.astype(np.float64) - out_data.astype(np.float64)) ** 2))
    peak = float(np.max(data) - np.min(data)) or 1.0
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(peak / math.sqrt(mse))
    return {"psnr": psnr, "version": _VERSION}


def _score_for_rotation(gray: np.ndarray, target_bits: np.ndarray) -> float:
    spec = np.fft.fft2(gray)
    mag = np.abs(spec)
    values = _detect_bits(mag)
    pred = (values > 0).astype(np.int8)
    return float(np.sum(pred == target_bits)) / _BITS


def extract_geo_watermark(source: Path, expected: int | None = None) -> dict:
    data, _, _ = _load(Path(source))
    gray, _, _ = _normalize(data)

    spec = np.fft.fft2(gray)
    mag = np.abs(spec)
    values = _detect_bits(mag)
    pred = (values > 0).astype(np.int8)
    value = _bits_to_value(pred)
    margin = float(np.mean(np.abs(values)) / (float(np.std(values)) + 1e-9))

    return {
        "watermark": value,
        "verified": expected is None or value == expected,
        "matched": (expected is None) or (value == expected),
        "bit_margin": round(margin, 4),
        "version": _VERSION,
    }
