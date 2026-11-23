"""TrustMark 瓦片 GeoTIFF 水印 runner（由独立 trustmark venv 的 python 执行）。

主项目环境 numpy>=2 与 torch 2.1.2 冲突，因此本 runner 必须用
runtime/trustmark_venv/Scripts/python.exe 执行（该环境 numpy<2 + torch 2.1.2+cpu +
trustmark 0.9.0 editable + rasterio + pyproj）。

用法:
    trustmark_geotiff_runner.py embed <source.tif> <dest.tif> <hex_watermark>
    trustmark_geotiff_runner.py extract <source.tif> <expected_hex>

约定:
- GeoTIFF 单波段灰度; 复制为 RGB 供 TrustMark; 嵌入后取 R 通道写回原 profile。
- 1024x1024 瓦片网格, 每瓦片独立嵌入同一 40bit BCH_SUPER 码字(32bit 业务位)。
- 保留原 dtype/CRS/transform/tags, 并写入 GDS_TM_SRC_CRS/TF/W/H 记录原空间参考。
- extract: 先读记录的原空间参考; 若当前 CRS 不同(重投影攻击)则 GDAL 逆投影回原网格;
  然后 bbox 检测 + 多尺度网格/细滑窗口解码。stdout 单行 JSON; 失败非零退出。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from queue import Queue

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import Affine
from trustmark import TrustMark

# Thread count for the model. Decoding is memory-bandwidth bound and the default torch pool (one
# thread per logical core) made a single decode cost 3.6-4.5 s under load, while one thread cost
# 249 ms. The batch harness therefore pins one thread per runner process and runs several
# processes; a service call that is the only work on the host keeps the default.
try:
    _THREADS = int(os.environ.get("GDS_TM_THREADS") or 0)
except ValueError:
    _THREADS = 0
_THREADS_OVERRIDE = _THREADS > 0
if _THREADS_OVERRIDE:
    import torch as _torch
    _torch.set_num_threads(_THREADS)

EXPECTED_HEX_TAG = "GDS_TM_PAYLOAD"
TILE = 512
TILE_STRIDE = 512


# TrustMark write strength for the tiled raster path. Measured 2026-10-17 on real L14 imagery
# (9 carriers x 8 attack types, deterministic across repeats, zero false positives on unmarked
# content). Mark strength decides whether the detector SEES the mark at all; 1.0 left 56% of
# carriers undetectable on real imagery, which is what was pushing carrier coverage to 66.7%:
#
#   strength  PSNR     base  resize50  resize200  crop16  rot1   rot5/90/180
#   1.0       44.28     44%      56%       56%      89%     33%       0%
#   1.5       41.20    100%     100%      100%     100%    100%       0%
#   2.0       38.85    100%     100%      100%     100%    100%       0%
#
# 1.5 is chosen: same robustness as 2.0 but ~2.3 dB better PSNR. 3.0 collapses again (33%), so the
# usable window is narrow - do not "push it higher for safety".
DEFAULT_WM_STRENGTH = 1.5


def _wm_strength() -> float:
    """Write strength for tiled raster embedding (override with GDS_TM_WM_STRENGTH)."""
    try:
        value = float(os.environ.get("GDS_TM_WM_STRENGTH") or 0)
    except ValueError:
        value = 0.0
    return value if value > 0 else DEFAULT_WM_STRENGTH


# Small rasters (a single carrier or a tiny grid) get a stronger write: they are the case where a
# crop, a resize or a recompression leaves only one tile to decode from, so the mark must carry more
# residual margin. Measured on the L14 2048 bases at strength 2.0: every carrier decodes at its
# natural phase, PSNR 37.5 dB (1.5 gives 39.9 dB). 3.0 was measured to collapse to undetectable,
# so this stays at 2.0 and is capped by SINGLE_CARRIER_BOOST_MAX carriers.
SINGLE_CARRIER_STRENGTH = 2.0
SINGLE_CARRIER_BOOST_MAX = 4
# A single-band (grey) product hides the mark in the only band it has, so it is written at the same
# raised strength as a single-carrier raster. Measured on the L14 2048 bases (1 band, 16 carriers):
# 40.4 dB PSNR, carrier grid read back 95/96.
GRAYSCALE_STRENGTH = 2.0
# Multi-band products keep the calibrated 1.5: measured PSNR on them was 50 dB at 1.5, so there is
# no budget pressure and no measured failure to justify spending more fidelity.
PRODUCT_STRENGTH = 1.5


def _grayscale_strength() -> float:
    try:
        value = float(os.environ.get("GDS_TM_GRAY_STRENGTH") or 0)
    except ValueError:
        value = 0.0
    return value if value > 0 else GRAYSCALE_STRENGTH


def _product_strength() -> float:
    try:
        value = float(os.environ.get("GDS_TM_PRODUCT_STRENGTH") or 0)
    except ValueError:
        value = 0.0
    return value if value > 0 else PRODUCT_STRENGTH


def _single_carrier_strength() -> float:
    """Write strength used for rasters that only yield a handful of carriers."""
    try:
        value = float(os.environ.get("GDS_TM_SMALL_STRENGTH") or 0)
    except ValueError:
        value = 0.0
    return value if value > 0 else SINGLE_CARRIER_STRENGTH


# Floor for embedding at all. Below this the payload is spread over so few pixels that a
# re-extraction cannot be trusted, and the task is better rejected up front than "succeeding".
MIN_EMBED_SIDE = 128
# Carrier side below which a carrier carries no recoverable mark. Measured 2026-10-18: a full
# watermarked carrier still decodes when resampled down to 224 px, and a 2048 base at 25% leaves a
# 128-px carrier that decodes in 8/8 cases (the blind searcher used to find it); 160 px is the floor
# for the rescale search so that it stops just below what is still readable.
_MIN_DECODABLE_SIDE = 128
# A carrier window flatter than this cannot hold a recoverable mark; same threshold the platform's
# robust GeoTIFF path applies before embedding.
FLAT_STD = 2.0
_ENCODING = TrustMark.Encoding.BCH_SUPER
_TM_CACHE: dict[tuple[str, bool], TrustMark] = {}
_TM_LOCAL = threading.local()


def _make_tm(load_bbox: bool = False) -> TrustMark:
    model_type = os.environ.get("GDS_TM_MODEL", "Q")
    key = (model_type, load_bbox)
    if key not in _TM_CACHE:
        _TM_CACHE[key] = TrustMark(
        verbose=False,
        model_type=model_type,
        encoding_type=_ENCODING,
        loadRemover=False,
        loadBBoxDetector=load_bbox,
        )
    return _TM_CACHE[key]


def _make_tm_threadlocal() -> TrustMark:
    """One decoder per thread, for the concurrent per-carrier verify path.

    TrustMark's BCH decoder keeps its working state on the decoder instance
    (``bchecc.py`` writes ``self.ECCstate.elp`` / ``self.ECCstate.errloc``), so two threads calling
    ``decode`` on the SAME instance corrupt each other's error locator polynomial. Measured on the
    14,061-carrier product: a shared instance raised ``IndexError: list index out of range`` inside
    ``bchecc.decode`` partway through the full run, and the same shape had been silently losing
    carriers before. A per-thread instance removes the race entirely.

    Cost is acceptable: one instance holds ~140 MB and takes ~1.0 s to load, so 12 workers is
    ~1.7 GB (the host has 128 GB). Instances are loaded lazily on first use in each thread, so a
    run that finishes with fewer workers never pays for the idle ones.
    """
    tm = getattr(_TM_LOCAL, "tm", None)
    if tm is None:
        # 必须直接构造，不能走 _make_tm()：后者按 (model_type, load_bbox) 缓存并返回同一个
        # 对象，"线程本地"就成了摆设（实测正是这个写法让竞态和串行化都还在）。
        tm = TrustMark(
            verbose=False,
            model_type=os.environ.get("GDS_TM_MODEL", "Q"),
            encoding_type=_ENCODING,
            loadRemover=False,
            loadBBoxDetector=False,
        )
        _TM_LOCAL.tm = tm
    return tm


def _payload_bits(watermark: int) -> str:
    return format(watermark, "032b")


def embed_image(source: Path, destination: Path, watermark: int, strength: float = 1.0) -> dict:
    """Embed one single-code-word watermark into an RGB image (texture) file.

    Textures are typically 256-2048 px photos; single-code-word TrustMark covers
    JPEG re-compression, uniform resize and noise (verified on photo-like textures).
    Cropping is NOT covered in single-code-word mode - real OSGB texture workflow
    never crops a texture, and the tiled GeoTIFF path covers crop-robust cases.
    Output keeps the source container (PNG lossless / JPEG), watermark survives.
    """
    source, destination = Path(source), Path(destination)
    with Image.open(source) as opened:
        mode = opened.mode
        rgb = opened.convert("RGB")
    tm = _make_tm()
    payload = _payload_bits(watermark)
    encoded = tm.encode(rgb, payload, MODE="binary", WM_STRENGTH=strength)
    suffix = destination.suffix.lower()
    save_kwargs: dict = {}
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 95}
    encoded.save(destination, **save_kwargs)
    with Image.open(destination) as saved:
        w, h = saved.size
    # PSNR over the RGB array vs the pre-encode cover
    mse = float(np.mean((np.asarray(rgb).astype(np.float64) - np.asarray(Image.open(destination).convert("RGB")).astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    return {"mode": "single_code_word", "width": w, "height": h, "psnr_db": round(psnr, 3), "source_mode": mode}


def extract_image(source: Path, expected_hex: str) -> dict:
    """Decode a single-code-word texture watermark. Multi-scale fallback covers
    resized textures (256-2048 px); plain decode covers JPEG/noise."""
    source = Path(source)
    expected = int(expected_hex, 16)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    w, h = image.size
    tm_plain = _make_tm()
    # try original size first (full/JPEG/noise keep size)
    bits, present, _ = tm_plain.decode(image, MODE="binary")
    if present and len(bits) >= 32 and int(bits[:32], 2) == expected:
        return {"detected": True, "method": "plain", "expected": expected_hex, "width": w, "height": h}
    # resized texture: TrustMark resizes to 224 internally; try a small lattice of
    # window sizes around 224px views is unnecessary - instead decode the resized
    # image directly since resize preserves the code word (verified resize50_jpeg).
    # Distinct path: image may be larger than trained range -> downscale lattice.
    scales = []
    for scale in (0.5, 0.25, 1.0):
        if scale != 1.0 and max(w, h) * scale >= 224:
            scales.append(scale)
    for scale in scales:
        resized = image.resize((max(224, int(w * scale)), max(224, int(h * scale))), Image.Resampling.BILINEAR)
        bits, present, _ = tm_plain.decode(resized, MODE="binary")
        if present and len(bits) >= 32 and int(bits[:32], 2) == expected:
            return {"detected": True, "method": f"rescale_{scale}", "expected": expected_hex, "width": w, "height": h}
    return {"detected": False, "method": "plain", "expected": expected_hex, "width": w, "height": h}


def embed(source: Path, destination: Path, watermark: int) -> dict:
    """Embed tile-by-tile so multi-gigabyte DOMs are never loaded as one array."""
    # 串行嵌入路径必须自己钉住 torch 线程数。不钉的话 torch 默认按物理核开池
    # （本机 36），而单个 1024 载体的编码张量很小，几十个线程全耗在同步上：
    # 实测一个 2.73 GB DOM 跑 13 分钟消耗 22,000 CPU 秒、进程 93 个线程，
    # 却一个字节都没写出。6 线程是实测最优（见 _embed_workers 的测量表）。
    try:
        if not _THREADS_OVERRIDE:
            import torch as _t
            _t.set_num_threads(max(1, min(8, (os.cpu_count() or 4) // 6)))
    except Exception:  # noqa: BLE001 - 线程钉定尽力而为
        pass
    source, destination = Path(source), Path(destination)
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        tags = src.tags()
        h, w = src.height, src.width
        if src.count < 1 or src.dtypes[0] != "uint8":
            raise ValueError(f"GeoTIFF must have a uint8 first band for TrustMark tiled embed, got {src.dtypes}")
        if min(h, w) < MIN_EMBED_SIDE:
            raise ValueError(f"GeoTIFF too small for TrustMark embed: {w}x{h}, need >= {MIN_EMBED_SIDE} px on both sides")
        tm = _make_tm()
        payload = _payload_bits(watermark)
        tile_size = _throughput_tile_size(w, h)
        tile_stride = tile_size
        # Working canvas. A raster shorter than one carrier on either side is scaled up so its short
        # side equals one carrier, and the marked result is scaled back to the original size, so the
        # output keeps its exact dimensions and georeferencing. Both padding variants were measured
        # to fail: a zero-padded (mostly flat) carrier does not decode at all, and a pad rebuilt from
        # watermarked edge pixels differs from the pad that was encoded. Scaling round-trips
        # (measured 384x256 -> 768x512 -> back, aligned window decodes on the second attempt).
        upscale = None
        if min(h, w) < tile_size:
            factor = tile_size / float(min(h, w))
            upscale = (int(round(w * factor)), int(round(h * factor)))
        work_w, work_h = ((max(tile_size, upscale[0]), max(tile_size, upscale[1])) if upscale
                          else (max(w, tile_size), max(h, tile_size)))
        row_starts = list(range(0, work_h - tile_size + 1, tile_stride))
        col_starts = list(range(0, work_w - tile_size + 1, tile_stride))
        # Extend to the far edge ONLY when the remaining strip is a whole carrier wide, so the extra
        # carrier cannot overlap the previous one. An overlapping carrier rewrites the same pixels
        # and destroys both payloads: with the old `last != work - tile` test a 791x718 raster got
        # carriers at y=0 and y=206 (306 px overlap) - 3 of the 4 code words were destroyed, embed
        # still reported success, and read-back only worked by luck on the last-written one.
        if row_starts[-1] + tile_size <= work_h - tile_size:
            row_starts.append(work_h - tile_size)
        if col_starts[-1] + tile_size <= work_w - tile_size:
            col_starts.append(work_w - tile_size)
        tf = profile.get("transform")
        tags["GDS_TM_ENGINE"] = "TrustMark-Q-TILED1024"
        tags["GDS_TM_ECC"] = "BCH_SUPER"
        tags[EXPECTED_HEX_TAG] = format(watermark, "08X")
        tags["GDS_TM_SRC_W"] = str(w)
        tags["GDS_TM_SRC_H"] = str(h)
        tags["GDS_TM_TILE_STRIDE"] = str(tile_stride)
        tags["GDS_TM_TILE_SIZE"] = str(tile_size)
        tags["GDS_TM_CARRIERS"] = str(len(row_starts) * len(col_starts))
        if len(row_starts) * len(col_starts) <= 256:
            # Small rasters are the case that used to fail, so record the exact carrier origins.
            # Large rasters keep the regular grid and are read back by the lattice/phase search.
            tags["GDS_TM_WINDOWS"] = ";".join(f"{tx},{ty}" for ty in row_starts for tx in col_starts)
        if upscale:
            tags["GDS_TM_CANVAS"] = f"{upscale[0]},{upscale[1]}"
        if profile.get("crs") is not None:
            tags["GDS_TM_SRC_CRS"] = str(profile["crs"].to_string() if hasattr(profile["crs"], "to_string") else profile["crs"])
        if tf is not None:
            tags["GDS_TM_SRC_TF"] = ",".join(str(round(float(v), 12)) for v in (tf.a, tf.b, tf.c, tf.d, tf.e, tf.f))
        tiles = 0
        sum_squared_error = 0.0
        samples = 0
        # Residual-strength boost: a crop smaller than one carrier leaves only ONE tile to read the
        # mark from, and the de-facto protection then depends on that single carrier's residual
        # margin. Measured 2026-10-18 on L14 imagery: carriers written at 1.5 failed on a tenth of
        # the tiles (base1 row0 all four, base4 (0,0)/(512,0) of a 2048 base) while a re-encode of
        # the SAME cover decodes at 2.0; the whole 4x4 grid then read 95/96 at PSNR 40.4 dB. A tiny
        # raster (<= SINGLE_CARRIER_BOOST_MAX carriers) is also promoted to 2.0, because for it the
        # carrier count decides the redundancy budget. 3.0 was measured to collapse to undetectable.
        carrier_count = len(row_starts) * len(col_starts)
        base_strength = _wm_strength()
        tiny = carrier_count <= SINGLE_CARRIER_BOOST_MAX
        strength = max(base_strength, _product_strength()) if not tiny else max(base_strength, _single_carrier_strength())
        if profile.get("count", 1) <= 1 and not upscale:
            strength = max(strength, _grayscale_strength())
        tags["GDS_TM_STRENGTH"] = repr(round(float(strength), 3))
        # Preserve every source band. The first band is the validated carrier;
        # other bands are copied unchanged rather than silently discarded.
        if upscale:
            # Scale up, embed the carrier(s) on the enlarged canvas, then scale the marked result
            # back to the original size (the file keeps its dimensions and georeferencing).
            base = np.asarray(Image.fromarray(src.read(1)).resize(upscale, Image.Resampling.BILINEAR), dtype=np.uint8)
            canvas = base.copy()
            for ty in row_starts:
                for tx in col_starts:
                    cover = base[ty:ty + tile_size, tx:tx + tile_size]
                    rgb = np.repeat(cover[:, :, None], 3, axis=2)
                    encoded = np.asarray(tm.encode(Image.fromarray(rgb, mode="RGB"), payload, MODE="binary", WM_STRENGTH=strength))[:, :, 0]
                    block = encoded.astype(np.uint8)
                    delta = cover.astype(np.float64) - block.astype(np.float64)
                    sum_squared_error += float(np.square(delta).sum())
                    samples += int(cover.size)
                    canvas[ty:ty + tile_size, tx:tx + tile_size] = block
                    tiles += 1
            back = np.asarray(Image.fromarray(canvas).resize((w, h), Image.Resampling.BILINEAR), dtype=np.uint8)
            with rasterio.open(destination, "w", **profile) as dst:
                dst.update_tags(**tags)
                for band in range(1, src.count + 1):
                    dst.write(src.read(band), band)
                dst.write(back, 1)
        else:
            with rasterio.open(destination, "w", **profile) as dst:
                dst.update_tags(**tags)
                for _, window in src.block_windows(1):
                    block = src.read(window=window)
                    dst.write(block, window=window)
                # 载体逐个串行编码只能吃满约 6 个线程（38 核的机器上大半空闲）。
                # 改成"读一行 -> 一行编码并行 -> 顺序写回"的流水线：读写在主线程（rasterio
                # 数据集非线程安全），只有 encode 进线程池。与 partial_embed 用的是同一套
                # 实测形状（6 工作者 x 6 torch 线程 = 105 ms/载体，vs 串行 238 ms）。
                _workers = _embed_workers()
                try:
                    import torch as _t
                    if not _THREADS_OVERRIDE:
                        _t.set_num_threads(max(1, (os.cpu_count() or 4) // (_workers * 2)))
                except Exception:  # noqa: BLE001 - 线程钉定尽力而为
                    pass
                with ThreadPoolExecutor(max_workers=_workers) as pool:
                    for ty in row_starts:
                        row_covers = []
                        for tx in col_starts:
                            w_tx = rasterio.windows.Window(tx, ty, tile_size, tile_size)
                            row_covers.append((tx, src.read(1, window=w_tx)))
                        row_blocks = pool.map(
                            partial(_encode_carrier_with_strength, tm, payload, strength=strength),
                            [cover for _, cover in row_covers])
                        for (tx, cover), block in zip(row_covers, row_blocks):
                            w_tx = rasterio.windows.Window(tx, ty, tile_size, tile_size)
                            dst.write(block, 1, window=w_tx)
                            delta = cover.astype(np.float64) - block.astype(np.float64)
                            sum_squared_error += float(np.square(delta).sum())
                            samples += int(cover.size)
                            tiles += 1
    mse = sum_squared_error / samples if samples else 0.0
    psnr = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    return {"tiles": tiles, "psnr_db": round(psnr, 3), "dtype": "uint8", "width": w, "height": h, "streaming": True}


def partial_embed(source: Path, destination: Path, watermark: int) -> dict:
    """Embed every readable carrier of a truncated DOM and record what could not be read.

    The platform cannot re-export a defective source, so the deliverable for a truncated raster is
    a full-extent product in which every readable source block is carried through, every carrier
    that can be read is watermarked, and the unreadable region is written as an explicit zero
    (NoData) representation. The valid extent and the skipped-carrier count are recorded in the
    tags so ``verify_carriers`` replays exactly the carriers that were embedded instead of
    searching (and failing) inside the missing region.

    Measured 2026-10-13 on this host: 211 ms per carrier including model load (64 carriers in
    13.5 s at PSNR 51.6 dB), i.e. ~13.5 min for a 3,834-carrier product, so the whole readable
    area is embedded - there is NO sampling cap. ``GDS_TM_PARTIAL_MAX_CARRIERS`` (0 = every
    readable carrier, the default) exists only for a deliberate bounded-sample policy.
    """
    source, destination = Path(source), Path(destination)
    max_carriers = max(0, int(os.environ.get("GDS_TM_PARTIAL_MAX_CARRIERS", "0") or 0))
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        tags = src.tags()
        h, w = src.height, src.width
        if src.count < 1 or src.dtypes[0] != "uint8":
            raise ValueError(f"GeoTIFF must have a uint8 first band for TrustMark tiled embed, got {src.dtypes}")
        if min(h, w) < MIN_EMBED_SIDE:
            raise ValueError(f"GeoTIFF too small for TrustMark embed: {w}x{h}, need >= {MIN_EMBED_SIDE} px on both sides")
        tm = _make_tm()
        payload = _payload_bits(watermark)
        tile_size = _throughput_tile_size(w, h)
        tile_stride = tile_size
        work_w, work_h = max(w, tile_size), max(h, tile_size)
        # Identical lattice to embed(): no two carriers overlap, and the far-edge carrier is added
        # only when a whole one still fits - an overlapping carrier destroys both payloads.
        row_starts = list(range(0, work_h - tile_size + 1, tile_stride))
        col_starts = list(range(0, work_w - tile_size + 1, tile_stride))
        if row_starts[-1] + tile_size <= work_h - tile_size:
            row_starts.append(work_h - tile_size)
        if col_starts[-1] + tile_size <= work_w - tile_size:
            col_starts.append(work_w - tile_size)
        tf = profile.get("transform")
        tags["GDS_TM_ENGINE"] = "TrustMark-Q-TILED1024"
        tags["GDS_TM_ECC"] = "BCH_SUPER"
        tags[EXPECTED_HEX_TAG] = format(watermark, "08X")
        tags["GDS_TM_SRC_W"] = str(w)
        tags["GDS_TM_SRC_H"] = str(h)
        tags["GDS_TM_TILE_STRIDE"] = str(tile_stride)
        tags["GDS_TM_TILE_SIZE"] = str(tile_size)
        tags["GDS_TM_LATTICE_CARRIERS"] = str(len(row_starts) * len(col_starts))
        if profile.get("crs") is not None:
            tags["GDS_TM_SRC_CRS"] = str(profile["crs"].to_string() if hasattr(profile["crs"], "to_string") else profile["crs"])
        if tf is not None:
            tags["GDS_TM_SRC_TF"] = ",".join(str(round(float(v), 12)) for v in (tf.a, tf.b, tf.c, tf.d, tf.e, tf.f))
        unreadable, skipped = [], []
        embedded = 0
        copy_blocks_total = 0
        valid_rows, contiguous = 0, True
        sum_squared_error, samples = 0.0, 0
        with rasterio.open(destination, "w", **profile) as dst:
            dst.update_tags(**tags)
            # Pass 1: carry the source through, writing zeros ONLY where it cannot be read.
            for _, window in src.block_windows(1):
                copy_blocks_total += 1
                try:
                    block = src.read(window=window)
                except Exception as exc:  # noqa: BLE001 - a truncated block is data, not a crash
                    unreadable.append({"row": int(window.row_off), "col": int(window.col_off),
                                       "width": int(window.width), "height": int(window.height),
                                       "error": f"{type(exc).__name__}: {exc}"})
                    block = np.zeros((src.count, int(window.height), int(window.width)), dtype=src.dtypes[0])
                dst.write(block, window=window)
            # Pass 2: watermark every carrier whose pixels can be read.
            #
            # Reading stays on this thread because a rasterio DatasetReader is not thread safe; only
            # the encode runs in the pool. A row of carriers is read, then its encodes are dispatched
            # to the pool together, so reads and encodes overlap instead of strictly alternating
            # (strict alternation measured 363 s on the verify path versus 230 s pipelined). Only one
            # row of covers is held at a time to bound memory: one 1536^2 row is a few MB.
            workers = _embed_workers()
            # Cap torch's own pool to match the worker count. Without this every worker thread opens
            # a full-size intra-op pool and the 6 workers oversubscribe the CPU - the measured shape
            # (6 workers x 6 torch threads, 105.3 ms/carrier) assumes this cap is in place.
            try:
                import torch
                if not _THREADS_OVERRIDE:
                    torch.set_num_threads(max(1, (os.cpu_count() or 4) // (workers * 2)))
            except Exception:  # noqa: BLE001 - thread pinning is best-effort
                pass
            seen = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for ty in row_starts:
                    row_complete = True
                    cells: list[np.ndarray | None] = []
                    for tx in col_starts:
                        if max_carriers and seen >= max_carriers:
                            cells.append(None)
                            continue
                        window = rasterio.windows.Window(tx, ty, tile_size, tile_size)
                        try:
                            cells.append(src.read(1, window=window))
                        except Exception:  # noqa: BLE001 - unreadable carrier, recorded below
                            cells.append(None)
                    # Encode only the readable carriers, keeping the column positions in step.
                    todo = [cover for cover in cells if cover is not None]
                    encoded_row = iter(pool.map(partial(_encode_carrier, tm, payload), todo))
                    for tx, cover in zip(col_starts, cells):
                        if cover is None:
                            row_complete = False
                            skipped.append([tx, ty])
                            continue
                        window = rasterio.windows.Window(tx, ty, tile_size, tile_size)
                        block = next(encoded_row)
                        dst.write(block, 1, window=window)
                        delta = cover.astype(np.float64) - block.astype(np.float64)
                        sum_squared_error += float(np.square(delta).sum())
                        samples += int(cover.size)
                        embedded += 1
                        seen += 1
                    if row_complete and contiguous:
                        valid_rows += 1
                    elif not row_complete:
                        contiguous = False
        valid_h = min(h, valid_rows * tile_stride)
        final = dict(tags)
        final.update({
            "GDS_PARTIAL_MODE": "1",
            "GDS_TM_CARRIERS": str(embedded),
            "GDS_TM_VALID_H": str(valid_h),
            "GDS_TM_SKIPPED_CARRIERS": str(len(skipped)),
            "GDS_TM_UNREADABLE_BLOCKS": str(len(unreadable)),
        })
        if embedded and embedded <= 256:
            # Small products record exact origins (same rule embed() uses); large ones are replayed
            # from the lattice plus the valid extent.
            final["GDS_TM_WINDOWS"] = ";".join(f"{tx},{ty}" for ty in row_starts for tx in col_starts
                                               if [tx, ty] not in skipped)
        with rasterio.open(destination, "r+") as dst:
            dst.update_tags(**final)
    mse = sum_squared_error / samples if samples else 0.0
    psnr = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    return {"carriers_total": len(row_starts) * len(col_starts), "carriers_embedded": embedded,
            "carriers_skipped": len(skipped), "valid_rows": valid_rows, "valid_h": valid_h,
            "copy_blocks_total": copy_blocks_total, "copy_blocks_unreadable": len(unreadable),
            "tile_size": tile_size, "tile_stride": tile_stride, "psnr_db": round(psnr, 3),
            "width": w, "height": h, "streaming": True,
            "skipped_carriers": skipped[:200], "unreadable_blocks": unreadable[:200]}


def _encode_carrier(tm, payload, cover: np.ndarray) -> np.ndarray:
    """Encode one grey carrier and return the single-band uint8 result.

    Called from a thread pool. The cover array is owned by the caller for the whole call; the model
    and payload are read-only, so concurrent calls are safe.
    """
    rgb = np.repeat(cover[:, :, None], 3, axis=2)
    encoded = np.asarray(tm.encode(Image.fromarray(rgb, mode="RGB"), payload,
                                   MODE="binary", WM_STRENGTH=_wm_strength()))[:, :, 0]
    return encoded.astype(np.uint8)


def _encode_carrier_with_strength(tm, payload, cover: np.ndarray, strength: float) -> np.ndarray:
    """Encode one grey carrier at an explicit strength and return single-band uint8.

    与 ``_encode_carrier`` 的区别只是强度由调用方传入：``embed()`` 会按产品规模挑强度
    （单载体产品用更强的 2.0），不能共用模块级的 ``_wm_strength()``。
    """
    rgb = np.repeat(cover[:, :, None], 3, axis=2)
    encoded = np.asarray(tm.encode(Image.fromarray(rgb, mode="RGB"), payload,
                                   MODE="binary", WM_STRENGTH=strength))[:, :, 0]
    return encoded.astype(np.uint8)


def _embed_workers() -> int:
    """Worker count for the partial-embed pool.

    Measured 2026-10-16 on this host (2 sockets x 18 physical cores, 72 logical): encode cost per
    carrier at several (workers x torch threads) shapes, projected to a 3,689-carrier product:
        1 x 36   238-256 ms   14.7-15.8 min   (serial: the previous behaviour)
        2 x 18   135.9 ms      8.4 min
        3 x 12   116.7 ms      7.2 min
        4 x  9   106.5 ms      6.6 min
        6 x  6   105.3 ms      6.5 min   <-- best
        9 x  4   111.2 ms      6.8 min
       12 x  3   116.4 ms      7.2 min
       18 x  2   120.8 ms      7.4 min
    Past six workers the per-thread torch pool stops being able to share cache and it gets worse,
    so this is a throughput ceiling and not a tuning problem. Pinning to a single socket changed
    nothing (104.3 vs 104.4 ms), so no affinity handling is used.
    """
    configured = int(os.environ.get("GDS_TM_EMBED_WORKERS", "0") or 0)
    return configured if configured > 0 else 6


def _read_orig_georef(source: Path) -> dict:
    """Read recorded original georeferencing written at embed time (may be absent)."""
    try:
        with rasterio.open(source) as ds:
            tags = ds.tags()
        out = {
            "crs": tags.get("GDS_TM_SRC_CRS"),
            "tf": tags.get("GDS_TM_SRC_TF"),
            "w": tags.get("GDS_TM_SRC_W"),
            "h": tags.get("GDS_TM_SRC_H"),
            "stride": tags.get("GDS_TM_TILE_STRIDE"),
        }
        return {k: v for k, v in out.items() if v is not None}
    except Exception:
        return {}


def _warp_back_to_orig_grid(source: Path, orig: dict) -> np.ndarray:
    """Reproject attacked image onto the recorded original CRS/grid via GDAL.

    Exact inverse of the warp attack: destination transform == recorded original
    transform and destination size == recorded original size, so the embedded
    1024-px tile grid aligns perfectly afterwards.
    """
    from rasterio.warp import reproject

    with rasterio.open(source) as ds:
        current_crs = ds.crs
        current_tf = ds.transform
        data = ds.read(1)
    if current_crs is None:
        raise ValueError("attacked image has no CRS; cannot reproject back")
    orig_crs = rasterio.crs.CRS.from_string(orig["crs"])
    if str(current_crs) == str(orig_crs):
        raise ValueError("CRS already matches original; no warp needed")
    tf_parts = [float(v) for v in orig["tf"].split(",")]
    orig_tf = Affine(*tf_parts)
    out_width, out_height = int(orig["w"]), int(orig["h"])
    destination = np.empty((out_height, out_width), dtype=np.uint8)
    reproject(
        source=data,
        destination=destination,
        src_transform=current_tf,
        src_crs=current_crs,
        dst_transform=orig_tf,
        dst_crs=orig_crs,
        resampling=Resampling.bilinear,
    )
    return destination


def _decode_ok(tm: TrustMark, image: Image.Image, expected: int) -> bool:
    bits, present, schema = tm.decode(image.convert("RGB"), MODE="binary")
    return bool(present) and len(bits) >= 32 and int(bits[:32], 2) == expected


# Carrier side chosen by raster extent.
#
# A larger carrier is markedly cheaper per pixel - measured 2026-10-16 on a 27,904x36,352 product:
#     512 px  90.4 ms x 3,834 carriers = 5.8 min serial
#    1024 px 188.9 ms x   945 carriers = 3.0 min serial
#    1536 px 342.0 ms x   414 carriers = 2.4 min serial
# because the fixed per-call cost of a model forward dominates. It is NOT used, however: on a
# truncated raster the unreadable edge covers a fixed number of pixels, so fewer, larger carriers
# mean a larger FRACTION of them are destroyed and coverage falls under the 0.90 acceptance bar.
# Measured on the same 93%-truncated 6144-px raster:
#     512 px  132/144 carriers = 91.67%  clears 0.90
#    1024 px   30/36          = 83.33%  FAILS
#    1536 px   12/16          = 75.00%  FAILS
# 512 is also the size the TrustMark acceptance matrices are calibrated on, so it stays.
def _adaptive_tile_size(side: int) -> int:
    """Carrier side for a raster.

    Measured 2026-10-10: a 256-px carrier DOES round-trip in isolation (embed 16 tiles then
    extract, window [0,0,256], attempts 2), but switching the carrier size per raster size is
    not safe: doing that regressed the TrustMark round-trip suite (tests/trustmark_*.py) with
    `blind_no_match`, because the engine/delegate and the acceptance matrices are calibrated on
    one carrier size. So 512 stays the production carrier for every raster that fits one (the
    same reason the OSGB path upsamples 224-511 px carriers to 512), and a 25% crop of a
    2048-px window stays a geometry limit. Use `GDS_TM_TILE_SIZE` for a deliberate experiment
    and re-run the whole attack matrix whenever the carrier changes.
    """
    return 1024 if side == 1024 else TILE


# Dense-raster carrier side.
#
# 512 is the size TrustMark and the attack matrices are calibrated on, but it fails the
# arbitrary-offset crop requirement: a 25 % crop of a 2048² product is exactly one carrier, so if
# the cut does not land on the carrier grid the crop holds nothing but a wrong-phase window and
# reads 0/20 (measured).
#
# A 256-px carrier fixes that. Measured 2026-10-18 on a real L14 2048² window, embedding the same
# payload and searching the crop at 32-px phase steps:
#     512 carrier, 512 stride : 16 carriers,   2.8 s, arbitrary 512-crop read   0/8
#     256 carrier, 256 stride : 64 carriers,   9.4 s, arbitrary 512-crop read  12/12 (step 32)
#     256 carrier, 128 stride : 225 carriers, 33.3 s, OVERLAP DESTROYS the payload (aligned read
#                                                     fails) - overlapping carriers stay forbidden
# So the carrier is halved for dense raster products only; the cost is 4x the carriers and ~3.4x
# the embed time (175 -> 147 ms per carrier, so the fixed per-call model cost does not shrink).
#
# Mesh/OSGB deliberately stays on 512: each OSGB member is its own image and its 10 GiB / 30 min
# budget is already at 26.8 min, so a 3.4x embed cost there would blow the acceptance limit.
DENSE_RASTER_TILE = 256


def _throughput_tile_size(width: int, height: int) -> int:
    """Carrier side for embedding a dense raster (GeoTIFF/DOM). See DENSE_RASTER_TILE above.

    载体边长必须跟着影像面积走，不能一律回退 256px：实测 27904x36352 的 2.73 GB DOM
    按 256 网格切成 15,478 个载体，单载体约 250 ms，串行要一小时以上，直接把
    「10 GB 倾斜模型/正射 ≤30 分钟」的指标打爆（实测跑到 13 分钟只烧掉 22,000 CPU 秒，
    一个字节都没写出）。按目标载体数反推边长，取 2 的幂并夹在 [256, 2048]。
    """
    override = int(os.environ.get("GDS_TM_TILE_SIZE") or 0)
    if override:
        return override
    # _adaptive_tile_size() keeps 1024-px rasters on a 1024 lattice (fewer, larger carriers);
    # every other dense raster used to drop to the 256-px carrier - now it only does so when the
    # image is small enough that 256 still yields a bounded carrier count.
    adaptive = _adaptive_tile_size(min(width, height))
    if adaptive != TILE:
        return adaptive
    try:
        target = max(256, int(os.environ.get("GDS_TM_MAX_CARRIERS", "2000") or 2000))
    except ValueError:
        target = 2000
    area = max(1, int(width) * int(height))
    side = int((area / target) ** 0.5)
    side = 1 << max(8, (side - 1).bit_length())  # 向上取 2 的幂，最小 256
    return max(DENSE_RASTER_TILE, min(side, 2048))


def _phase_lattice(ds, tm, expected: int, tile: int, step: int | None = None) -> dict | None:
    """Probe the carrier phase lattice directly.

    A crop shifts the whole carrier grid by one fixed (px, py) in [0, tile); probing those
    exact offsets therefore lands on complete carriers. Measured tolerance is about +-64 px on
    x and +-16 px on y, so a 16-px phase step still lands inside it while costing only
    (tile/step)^2 decodes instead of a full sliding search.
    """
    if step is None:
        step = int(os.environ.get("GDS_TM_PHASE_STEP", "16"))
    attempts = 0
    for py in range(0, tile, step):
        for px in range(0, tile, step):
            if px + tile > ds.width or py + tile > ds.height:
                continue
            attempts += 1
            if _decode_window(tm, ds, px, py, tile, expected):
                return {"detected": True, "method": "phase_lattice_window", "window": [px, py, tile],
                        "attempts": attempts, "expected": f"{expected:08X}"}
    return None


PHASE_SEARCH_STEP = 8


def _phase_sweep(ds, tm, expected: int, tile: int, step: int | None = None,
                 budget: int | None = None) -> dict | None:
    """Slide the carrier-sized window over the whole raster to recover a crop's phase.

    ``_phase_lattice`` only probes offsets inside [0, tile) of the *origin* lattice, which assumes
    the file still starts on some carrier boundary. An arbitrary crop does not: a 25 % crop of a
    2048² product holds one carrier's worth of pixels cut at an unknown phase, so the only window
    it can produce is the wrong one and the mark reads back absent.

    With the 256-px dense-raster carrier (DENSE_RASTER_TILE) a 512-px crop contains a full 2x2 of
    carriers and this sweep finds one: measured 12/12 arbitrary crops read back at a 32-px step
    (12/12 again at 16 px), where the same crops read 0/8 on the old 512-px lattice. The step is
    half the measured phase tolerance, so no true alignment can be skipped.

    Cost is bounded by ``budget`` decodes (default 600) so a hostile file cannot turn extraction
    into an unbounded scan.
    """
    if step is None:
        step = int(os.environ.get("GDS_TM_PHASE_STEP", str(PHASE_SEARCH_STEP)))
    if budget is None:
        budget = int(os.environ.get("GDS_TM_PHASE_BUDGET", "1600"))
    if tile > ds.width or tile > ds.height:
        return None
    attempts = 0
    for y in range(0, ds.height - tile + 1, step):
        for x in range(0, ds.width - tile + 1, step):
            if attempts >= budget:
                return None
            attempts += 1
            if _decode_window(tm, ds, x, y, tile, expected):
                return {"detected": True, "method": "phase_sweep_window",
                        "window": [x, y, tile], "attempts": attempts,
                        "expected": f"{expected:08X}"}
    return None


def _anisotropic_sweep(ds, tm, expected: int, tile_candidates: list[int],
                       step: int | None = None, budget: int | None = None) -> dict | None:
    """Recover a watermark from a product whose two axes were rescaled differently.

    Reprojection is the common cause: EPSG:3857 -> EPSG:4326 is not a uniform scale, so a
    2048x2048 product becomes e.g. 2234x1843. Each square carrier turns into a rectangle of
    roughly 279x230 px, and every square window the other paths probe is misaligned with that
    lattice by construction - measured 1/5 on reprojected products while the same products read
    100% for every other attack.

    The canvas was a whole number of carriers, so the axis factors are recoverable from the
    output size: for each candidate carrier side and each plausible carrier count along an axis,
    factor = size / (count * tile). The two axes are then probed independently and the resulting
    rectangle is read with ``out_shape=(tile, tile)``, which resamples the anisotropic carrier
    back to the square shape the decoder expects.

    Cost is bounded by ``budget`` decodes so a hostile file cannot make this unbounded.
    """
    if step is None:
        step = int(os.environ.get("GDS_TM_ANISO_STEP", "32"))
    if budget is None:
        budget = int(os.environ.get("GDS_TM_ANISO_BUDGET", "2400"))
    attempts = 0
    for tile in tile_candidates:
        if tile > ds.width or tile > ds.height:
            continue
        # The canvas held a whole number of carriers, and a reprojection keeps that COUNT while
        # changing the side lengths. Enumerate the plausible counts directly instead of deriving
        # one from round(width/tile): a 2048-px canvas of 256-px carriers is 8x8, and reprojecting
        # to 2225x1854 gives 278x232 - but round(2225/256)=9 and round(1854/256)=7, which is the
        # wrong grid (247x265). Ordering by |side - tile| then ranks the true shape behind
        # 279x263-style near-misses, which is exactly why the first attempt read 0 hits while a
        # direct 278x232 sweep found 33.
        shapes: list[tuple[float, int, int]] = []
        for nx in range(1, min(64, ds.width // 16) + 1):
            side_x = ds.width / nx
            if not (tile * 0.4 <= side_x <= tile * 3.0):
                continue
            for ny in range(1, min(64, ds.height // 16) + 1):
                side_y = ds.height / ny
                if not (tile * 0.4 <= side_y <= tile * 3.0):
                    continue
                if abs(side_x - side_y) / max(side_x, side_y) < 0.02:
                    continue  # uniform resize: already covered by the square paths
                sx, sy = int(round(side_x)), int(round(side_y))
                if sx < 16 or sy < 16 or sx > ds.width or sy > ds.height:
                    continue
                # A reprojection stretches one axis and squeezes the other, so the two sides
                # straddle the original tile size. Rank by how close BOTH sides are to it.
                penalty = abs(side_x - tile) + abs(side_y - tile)
                shapes.append((penalty, sx, sy))
        shapes.sort()
        ordered: list[tuple[int, int]] = []
        for _penalty, sx, sy in shapes:
            if (sx, sy) not in ordered:
                ordered.append((sx, sy))
        # Per-shape budget, not a global one. Each shape holds ~3100 windows at a 32-px stride on
        # a 2225x1854 image, so a single global budget of 2400 was consumed entirely by the FIRST
        # shape and the real carrier (5th) was never reached - measured 0 hits through the function
        # while a direct 278x232 sweep found 24. Give every shape its own slice so the search
        # actually covers several geometries.
        per_shape = max(64, budget // max(1, min(len(ordered), 24)))
        for sx, sy in ordered:
            spent = 0
            for y in range(0, ds.height - sy + 1, step):
                for x in range(0, ds.width - sx + 1, step):
                    if spent >= per_shape or attempts >= budget:
                        break
                    spent += 1
                    attempts += 1
                    try:
                        data = ds.read(1, window=rasterio.windows.Window(x, y, sx, sy),
                                       out_shape=(tile, tile), resampling=Resampling.cubic)
                    except Exception:  # noqa: BLE001
                        continue
                    image = Image.fromarray(
                        np.repeat(data[:, :, None], 3, axis=2).astype(np.uint8), mode="RGB")
                    if _decode_ok(tm, image, expected):
                        return {"detected": True, "method": "anisotropic_tile_window",
                                "window": [x, y, sx, sy], "attempts": attempts,
                                "expected": f"{expected:08X}"}
                else:
                    continue
                break
            if attempts >= budget:
                return None
    return None


def _scale_candidates(width: int, height: int, tile: int) -> list[float]:
    """Plausible resize factors of an embedded raster, most likely first.

    Embedding wrote ``tile``-sized carriers on a ``work_w x work_h`` canvas that was the original
    size rounded up to a whole number of carriers, so an attacked image of ``width x height`` was
    resized by ``width / work_w`` for some integer grid. Recovering the exact factor therefore only
    needs the recorded canvas when the tags survived; without them the candidate set is every
    factor that maps an integer carrier grid onto this exact size. Factors whose carrier side is
    far from an integer pixel edge are indistinguishable by pixel reading anyway, so at most
    ``_SCALE_CANDIDATE_LIMIT`` of them are tried, and exact-ratio matches (a plain x1.25, x1.5,
    x0.8 resize) are ordered first because that is what image-resize tooling actually produces.
    """
    limit = int(os.environ.get("GDS_TM_SCALE_CANDIDATES", "10") or 10)
    # Ordering matters for speed, not for correctness - every candidate is eventually probed. The
    # ratios resize tooling actually produces are probed first, and a factor whose carrier side
    # lands on a whole pixel comes before a fractional one (measured: a 2048 base at 150% is
    # 3072 px, i.e. 6x4 carriers of exactly 512 px, but a 1.00 candidate found one by chance on the
    # first window and the real factor 1.50 was never reached, burning 1,288 decodes / 95 s).
    common = (1.25, 1.5, 1.75, 2.0, 0.8, 0.75, 0.5, 0.25, 1.0, 0.9, 1.1, 1.2, 1.33, 1.6, 1.4, 3.0)
    geometric = [2 ** (k / 8) for k in range(-16, 17)]
    geometric += [1 + k / 20 for k in range(-16, 21)]
    seen: set[float] = set()
    candidates: list[float] = []
    for factor in list(common) + geometric:
        if factor in seen:
            continue
        seen.add(factor)
        side = tile * factor
        if 0.2 <= factor <= 4.0 and width >= side and height >= side:
            candidates.append(factor)
    integral = sorted(candidates, key=lambda f: (abs(tile * f - round(tile * f)), abs(f - 1.0)))
    # grid-derived factors: width = n * tile * factor  ->  factor = width / (n * tile)
    for axis in (width, height):
        for n in range(1, max(1, axis // tile) + 1):
            factor = round(axis / (n * tile), 4)
            if factor not in seen and 0.2 <= factor <= 4.0 and width >= tile * factor and height >= tile * factor:
                seen.add(factor)
                integral.append(factor)
    return integral[:limit]


def _agreed_decodes(tm, dataset, expected: int, tile_size: int, scale_value: float,
                    width: int, height: int) -> int:
    """How many windows of one candidate grid read the payload correctly.

    Used to tell a genuine grid (many independent windows agree, because every carrier holds the
    same code word) from a lucky single decode. Eight windows spread across the image are enough:
    a crop's phase shift is constant, so a real grid shows several hits immediately.
    """
    side = max(64, int(round(tile_size * scale_value)))
    if side > width or side > height or side < _MIN_DECODABLE_SIDE:
        return 0
    step_y = max(1, (height - side) // 3) if height > side else 0
    step_x = max(1, (width - side) // 3) if width > side else 0
    hits = 0
    for y in (0, step_y, 2 * step_y, 3 * step_y):
        for x in (0, step_x, 2 * step_x, 3 * step_x):
            if y + side > height or x + side > width:
                continue
            data = dataset.read(1, window=rasterio.windows.Window(x, y, side, side),
                                out_shape=(tile_size, tile_size), resampling=Resampling.cubic)
            image = Image.fromarray(np.repeat(data[:, :, None], 3, axis=2).astype(np.uint8), mode="RGB")
            if _decode_ok(tm, image, expected):
                hits += 1
    return hits


def extract_blind(source: Path, expected_hex: str, max_windows: int = 800) -> dict:
    """Blindly locate a tiled watermark using pixels only.

    This deliberately ignores all GDS tags, CRS and transform. It is intended
    for files that have been cropped or rewritten by software that discarded
    synchronization metadata.
    """
    expected = int(expected_hex, 16)
    with rasterio.open(source) as ds:
        width, height = ds.width, ds.height
        tm = _make_tm()
        # Candidate window sides for the blind search. A 25 % downscale of a 256-px-carrier
        # product leaves 64-px carriers, and that size was simply absent from this list - which is
        # why the hardest downscale cases exhausted the budget at a=35 and reported "absent". The
        # small sides are appended after the large ones so untouched full-size products still hit
        # on their first windows.
        sizes = [1024, 768, 512, 384, 256, 192, 128, 96, 64]
        attempts = 0
        for size in sizes:
            if size > min(width, height):
                continue
            # Step floor must scale with the window: a 64-px carrier with a 64-px stride probes 8
            # positions per axis and can walk straight past the true phase (measured phase
            # tolerance is only ~1/8 of the carrier side). Large windows keep the old 64-px floor
            # so full-size products stay as cheap as before.
            step = max(16, size // 8) if size < 256 else max(64, size // 8)
            coords = [(x, y) for y in range(0, height - size + 1, step)
                      for x in range(0, width - size + 1, step)]
            coords.sort(key=lambda p: abs(p[0] + size / 2 - width / 2) + abs(p[1] + size / 2 - height / 2))
            for x, y in coords:
                attempts += 1
                if attempts >= max_windows:
                    return {"detected": False, "method": "blind_window_exhausted", "attempts": attempts, "expected": expected_hex}
                data = ds.read(1, window=rasterio.windows.Window(x, y, size, size), out_shape=(size, size), resampling=Resampling.bilinear)
                image = Image.fromarray(np.repeat(data[:, :, None], 3, axis=2).astype(np.uint8), mode="RGB")
                if _decode_ok(tm, image, expected):
                    return {"detected": True, "method": "blind_pixel_window", "window": [x, y, size], "attempts": attempts, "expected": expected_hex}
    return {"detected": False, "method": "blind_no_match", "attempts": attempts, "expected": expected_hex}


def _read_window_padded(dataset, x: int, y: int, size: int, canvas: tuple[int, int] | None = None) -> np.ndarray:
    """Read a `size`x`size` window, rebuilding the canvas the carrier was encoded on.

    embed() scales a sub-carrier raster up so its short side equals one carrier, encodes there and
    scales the marked result back down, recording the enlarged canvas in ``GDS_TM_CANVAS`` - so
    extraction scales the stored raster back up the same way before decoding. Without a recorded
    canvas this is a plain read; any part of the window outside the raster is zero-FILLED only as a
    last resort for out-of-range probes.
    """
    x0, y0 = int(x), int(y)
    if canvas and (dataset.width, dataset.height) != (int(canvas[0]), int(canvas[1])):
        block = dataset.read(1)
        enlarged = Image.fromarray(block.astype(np.uint8)).resize((int(canvas[0]), int(canvas[1])), Image.Resampling.BILINEAR)
        data = np.asarray(enlarged, dtype=np.float64)
        return data[y0:y0 + size, x0:x0 + size]
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(dataset.width, x0 + size), min(dataset.height, y0 + size)
    padded = np.zeros((size, size), dtype=np.float64)
    if right > left and bottom > top:
        block = dataset.read(1, window=rasterio.windows.Window(left, top, right - left, bottom - top))
        padded[top - y0:bottom - y0, left - x0:right - x0] = block
    return padded


def _decode_window(tm: TrustMark, dataset, x: int, y: int, size: int, expected: int,
                   canvas: tuple[int, int] | None = None) -> bool:
    """Decode one raster window without materializing the full DOM."""
    data = _read_window_padded(dataset, x, y, size, canvas)
    image = Image.fromarray(np.repeat(np.clip(data, 0, 255)[:, :, None], 3, axis=2).astype(np.uint8), mode="RGB")
    return _decode_ok(tm, image, expected)


def _decode_array_window(tm: TrustMark, data: np.ndarray, x: int, y: int, size: int,
                         expected: int) -> bool:
    """Decode a bounded window from an already warped single-band array."""
    window = data[int(y):int(y) + int(size), int(x):int(x) + int(size)]
    if window.shape != (int(size), int(size)):
        return False
    image = Image.fromarray(np.repeat(np.clip(window, 0, 255)[:, :, None], 3, axis=2).astype(np.uint8), mode="RGB")
    return _decode_ok(tm, image, expected)


def extract(source: Path, expected_hex: str, max_windows: int = 400) -> dict:
    """Extract by reading bounded candidate windows; full DOM arrays are prohibited."""
    source = Path(source)
    expected = int(expected_hex, 16)
    orig = _read_orig_georef(source)
    # A CRS-changing resample preserves the GDS tags but shifts every carrier away from its
    # original pixel lattice. Rebuild the recorded grid before any regular extraction attempt.
    # This is intentionally before opening the normal streaming path: that path is bounded for
    # multi-gigabyte untouched/cropped products, while a reprojection necessarily needs a full
    # destination grid to undo the coordinate transform.
    if all(orig.get(key) for key in ("crs", "tf", "w", "h")):
        with rasterio.open(source) as current:
            current_crs = current.crs
        original_crs = rasterio.crs.CRS.from_string(orig["crs"])
        if current_crs is not None and current_crs != original_crs:
            warped = _warp_back_to_orig_grid(source, orig)
            tm_plain = _make_tm()
            tile_size = int(orig.get("stride") or TILE)
            attempts = 0
            for y in range(0, warped.shape[0] - tile_size + 1, tile_size):
                for x in range(0, warped.shape[1] - tile_size + 1, tile_size):
                    attempts += 1
                    if _decode_array_window(tm_plain, warped, x, y, tile_size, expected):
                        return {"detected": True, "method": "inverse_reproject_tile_window",
                                "expected": expected_hex, "window": [x, y, tile_size],
                                "attempts": attempts, "warped": True}
            return {"detected": False, "method": "inverse_reproject_no_match",
                    "expected": expected_hex, "attempts": attempts, "warped": True}
    with rasterio.open(source) as ds:
        w, h = ds.width, ds.height
        tm_plain = _make_tm()
        # Reprojection changes width and height by DIFFERENT factors, so the carrier grid stops
        # being square. Detect that shape up front and try the rectangular sweep FIRST: running
        # every square-window path before it costs ~10 minutes per reprojected product (measured
        # 575 s on a 2225x1854 file where the anisotropic sweep alone needs 15 s / 201 decodes).
        # The signal is the aspect ratio drifting away from the whole-carrier grid the embedder
        # wrote, which no uniform resize produces.
        if "GDS_TM_TILE_SIZE" in ds.tags():
            _tagged_tile = int(ds.tags().get("GDS_TM_TILE_SIZE") or TILE)
        else:
            _tagged_tile = DENSE_RASTER_TILE
        _canvas_tag = (ds.tags().get("GDS_TM_CANVAS", "") or "").split(",")
        _uniform = True
        if len(_canvas_tag) == 2 and all(p.strip().isdigit() for p in _canvas_tag):
            _cw, _ch = int(_canvas_tag[0]), int(_canvas_tag[1])
            _sx, _sy = w / _cw, h / _ch
            _uniform = abs(_sx - _sy) / max(_sx, _sy) < 0.01
        else:
            # No canvas tag: a uniform resize keeps width:height, a reprojection does not.
            _nx, _ny = w / _tagged_tile, h / _tagged_tile
            _uniform = abs(_nx - round(_nx)) < 0.02 and abs(_ny - round(_ny)) < 0.02
        if not _uniform:
            early = _anisotropic_sweep(ds, tm_plain, expected, [_tagged_tile, TILE, DENSE_RASTER_TILE])
            if early:
                return early
        # The first centre window is deterministic and sufficient for the untampered
        # grid written by embed().  Fall back to a bounded lattice for partial crops.
        tile_size = int(ds.tags().get("GDS_TM_TILE_SIZE", TILE))
        # A crop/format tool drops the GDS tags, so tile_size falls back to TILE (512) even when
        # the product was embedded with the 256-px dense-raster carrier. Searching a 256-px
        # lattice with 512-px windows cannot succeed - measured: one arbitrary-offset crop yielded
        # two valid 256-px carriers at (224,0)/(224,256) yet extract() reported blind_no_match
        # because it only probed 512. So when the tag is absent, probe every carrier size the
        # product could plausibly carry. Cheap: a wrong size fails in a handful of decodes.
        tile_candidates: list[int] = [tile_size]
        if "GDS_TM_TILE_SIZE" not in ds.tags():
            # Dense-raster products (the only ones embedded at 256 px) are the common case for a
            # stripped-tag crop, so probe that lattice FIRST: probing 512 first burned 378 wasted
            # decodes and made every arbitrary-offset crop cost ~115 s instead of ~15 s.
            for size in (DENSE_RASTER_TILE, TILE):
                if size not in tile_candidates and size <= min(w, h):
                    tile_candidates.insert(0 if size == DENSE_RASTER_TILE else len(tile_candidates), size)
            tile_candidates = sorted(set(tile_candidates), key=lambda s: (s != DENSE_RASTER_TILE, s))
            tile_size = tile_candidates[0]
        # A sub-carrier raster was scaled up before embedding; GDS_TM_CANVAS records that enlarged
        # canvas so the same scale-up can be replayed here before decoding.
        canvas_tag = (ds.tags().get("GDS_TM_CANVAS", "") or "").split(",")
        canvas = (int(canvas_tag[0]), int(canvas_tag[1])) if len(canvas_tag) == 2 and all(part.strip().isdigit() for part in canvas_tag) else None
        candidates: list[tuple[int, int, int]] = []
        # Carriers are recorded in the file whenever there are few of them (small rasters, and the
        # zero-padded ones): try exactly those origins first. A raster shorter than one carrier has
        # no origin lattice at all, so this is its only deterministic read path.
        for pair in (ds.tags().get("GDS_TM_WINDOWS", "") or "").split(";"):
            parts = pair.split(",")
            if len(parts) >= 2:
                try:
                    candidates.append((int(parts[0]), int(parts[1]), tile_size))
                except ValueError:
                    continue
        if w >= tile_size and h >= tile_size:
            candidates.append(((w - tile_size) // 2, (h - tile_size) // 2, tile_size))
            for y in range(0, h - tile_size + 1, tile_size):
                for x in range(0, w - tile_size + 1, tile_size):
                    candidates.append((x, y, tile_size))
                    if len(candidates) >= max_windows:
                        break
                if len(candidates) >= max_windows:
                    break
        seen: set[tuple[int, int, int]] = set()
        attempts = 0
        hit: dict = {}

        def try_grid(scale_value: float, label: str, offset: int = 0, coarse: int = 4,
                     budget: int | None = None) -> bool:
            """Decode carriers on the grid this image would have at a candidate scale.

            A crop of a 512-carrier lattice keeps the lattice; a RESIZE does not - it multiplies the
            carrier side by the scale factor, so a 125% resize leaves 640-px carriers and every
            512-px window straddles four of them (measured: 125% read back 15/18 only through the
            blind searcher). The carrier geometry is recoverable without any metadata: carrier side
            = recorded side * scale, and any complete carrier fills a whole window of that size.

            The phase is searched coarse-to-fine: a stride of ``step``/``coarse`` first (a crop's
            phase shift is constant for the whole image, so any one carrier landing on the grid
            proves it), then a dense pass only for the factor that succeeded at coarse stride.
            """
            nonlocal attempts
            side = max(64, int(round(tile_size * scale_value)))
            if side > w or side > h:
                return None
            # A carrier smaller than the decode floor (<128 px) is still readable: TrustMark accepts
            # an upscaled carrier (the rescaled path already feeds it out_shape=(tile, tile)). So
            # instead of giving up, read the small window and let the decode resample it up. This is
            # what makes a 25% downscale of a 256-carrier product (carrier -> 64 px) survive; it used
            # to be rejected outright here, which is why scale25 read 0/15 after the carrier dropped
            # from 512 to 256.
            pass
            # Phase step for this carrier size. The floor used to be 64 px unconditionally, which
            # silently skipped 3 of every 4 phases once the dense-raster carrier dropped to 256 px
            # (256/4 = 64 px stride covers only 4 of 16 phases at 32-px resolution) - that is why
            # arbitrary-offset crops read 11/19 while the same crops read 12/12 through the
            # dedicated sweep. Tie the stride to the carrier instead of a fixed constant.
            step = max(16, side // max(1, coarse))
            spent = 0
            for y in range(offset, h - side + 1, step):
                for x in range(offset, w - side + 1, step):
                    if budget is not None and spent >= budget:
                        return None
                    spent += 1
                    attempts += 1
                    window = rasterio.windows.Window(x, y, side, side)
                    data = ds.read(1, window=window, out_shape=(tile_size, tile_size),
                                   resampling=Resampling.cubic)
                    image = Image.fromarray(np.repeat(data[:, :, None], 3, axis=2).astype(np.uint8), mode="RGB")
                    if _decode_ok(tm_plain, image, expected):
                        return {"detected": True, "method": f"rescaled_tile_window_{label}",
                                "expected": expected_hex, "window": [x, y, side],
                                "attempts": attempts, "warped": False}
            return None

        # Candidate scale factors: the image size itself is the strongest signal. Try each factor on
        # a coarse lattice first - that is ~8 decodes per factor - and only refine the factors that
        # actually decoded. A blind dense sweep at the wrong factor costs hundreds of decodes for
        # nothing (measured 1,288 decodes / 95 s on a 125% resize before this ordering).
        supported: list[tuple[float, str]] = []
        for scale_value in _scale_candidates(w, h, tile_size):
            # Budget the coarse pass: an incorrect factor only needs a few decodes to be ruled out,
            # but the old loop walked the entire grid for every candidate (measured 970 decodes /
            # 466 s on one 25 % downscale). Cap the coarse probe per factor and let the two-stage
            # refinement (`supported` + coarse=8) spend the budget on the factors that actually
            # decode.
            found = try_grid(scale_value, f"{scale_value:.3f}", coarse=4, budget=64)
            if found:
                return found
            if _agreed_decodes(tm_plain, ds, expected, tile_size, scale_value, w, h) >= 2:
                supported.append((scale_value, f"{scale_value:.3f}"))
        # A factor is only refined once two independent windows read the SAME payload: one decode is
        # noise-level evidence, and spending ~600 decodes on every candidate made a 125% resize cost
        # 95 s when the correct factor had already decoded 84 times.
        for scale_value, label in supported[:2]:
            found = try_grid(scale_value, f"{label}_fine", coarse=8)
            if found:
                return found

        for x, y, size in candidates:
            candidate = (x, y, size)
            if candidate in seen:
                continue
            seen.add(candidate)
            attempts += 1
            if _decode_window(tm_plain, ds, x, y, size, expected, canvas):
                return {"detected": True, "method": "streaming_tile_window", "expected": expected_hex, "window": [x, y, size], "attempts": attempts, "warped": False}
        # The origin lattice only hits when the file still sits on its original carrier grid.
        # A crop (or any tool that drops the GDS tags) shifts that grid by one fixed phase, so
        # probe the phase lattice directly before giving up.
        phased = _phase_lattice(ds, tm_plain, expected, tile_size)
        if phased:
            phased["after_aligned_attempts"] = attempts
            return phased
        # A crop smaller than one carrier-worth of the original lattice still holds complete
        # carriers when the carrier is small (256 px): slide the carrier window over the whole
        # crop to recover the cut phase. Measured 12/12 on arbitrary 25 % crops, versus 0/8 with
        # only the origin lattice. Every candidate carrier size is swept, because a stripped-tag
        # crop does not tell us which one the product was embedded with.
        for swept_tile in tile_candidates:
            swept = _phase_sweep(ds, tm_plain, expected, swept_tile)
            if swept:
                swept["after_aligned_attempts"] = attempts
                return swept
        # A reprojection (EPSG:3857 -> 4326 and back) rescales the two axes by DIFFERENT factors,
        # so every square carrier becomes a slightly non-square rectangle. Measured: a 2048x2048
        # product reprojected to 4326 becomes 2225x1854, i.e. x*1.086 but y*0.905 - a square
        # window can never align with that lattice, which is why reprojected products read 1/5.
        # Recover the two axis factors from the size ratio (the canvas was a whole number of
        # carriers) and probe rectangular windows of that shape.
        anisotropic = _anisotropic_sweep(ds, tm_plain, expected, tile_candidates)
        if anisotropic:
            anisotropic["after_aligned_attempts"] = attempts
            return anisotropic
    # A reprojection keeps the GDS tags (rasterio copies them through), but they describe the
    # PRE-reprojection lattice: GDS_TM_TILE_SIZE still says 256 while the carriers are now
    # 278x232. Every tag-driven read path therefore replays a grid that no longer exists, so the
    # anisotropic sweep has to be reachable even when the tags survived.
    if "GDS_TM_TILE_SIZE" in ds.tags():
        anisotropic = _anisotropic_sweep(ds, tm_plain, expected, [DENSE_RASTER_TILE, TILE])
        if anisotropic:
            anisotropic["after_aligned_attempts"] = attempts
            return anisotropic
    # Crop/format software usually drops the GDS tile tags and moves the origin off the
    # 512-px carrier grid, which makes every origin-anchored window miss. Fall back to a
    # bounded pixel-only search so an arbitrary-offset crop is still recognised instead of
    # being reported as "watermark absent".
    blind = extract_blind(source, expected_hex, max_windows=int(os.environ.get("GDS_TM_BLIND_WINDOWS", "2400")))
    if blind.get("detected"):
        blind["after_aligned_attempts"] = attempts
    return blind


def _carrier_plan(dataset) -> dict:
    """Where this product's carriers are, taken from the product's own tags.

    The full-image product records ``GDS_TM_CARRIERS`` together with the tile/stride it embedded
    on; sub-carrier rasters additionally record ``GDS_TM_WINDOWS`` (exact origins) and
    ``GDS_TM_CANVAS`` (the enlarged canvas they were embedded on). Verification therefore never
    has to search: it replays the recorded lattice, which is what makes whole-image per-carrier
    verification cheap enough to run on every request.
    """
    tags = dataset.tags()
    tile = int(tags.get("GDS_TM_TILE_SIZE", TILE) or TILE)
    stride = int(tags.get("GDS_TM_TILE_STRIDE", tile) or tile)
    origins: list[tuple[int, int]] = []
    for pair in (tags.get("GDS_TM_WINDOWS", "") or "").split(";"):
        if "," in pair:
            px, py = pair.split(",")
            origins.append((int(px), int(py)))
    origins_source = "recorded"
    if not origins:
        origins_source = "lattice"
        for y in range(0, dataset.height - tile + 1, stride):
            for x in range(0, dataset.width - tile + 1, stride):
                origins.append((x, y))
    canvas_tag = (tags.get("GDS_TM_CANVAS", "") or "").split(",")
    canvas = (int(canvas_tag[0]), int(canvas_tag[1])) if len(canvas_tag) == 2 and canvas_tag[0].strip() else None
    lattice_total = (len(range(0, dataset.height - tile + 1, stride))
                     * len(range(0, dataset.width - tile + 1, stride)))
    # A partial product records how far down the raster is actually readable. Carriers below that
    # extent live in the unreadable region of a truncated source; verification must replay only the
    # carriers that were embedded, not count the missing region as misses.
    valid_h = int(tags.get("GDS_TM_VALID_H", 0) or 0)
    in_extent = [origin for origin in origins if not valid_h or origin[1] + tile <= valid_h]
    return {"tile": tile, "stride": stride, "origins": in_extent or origins, "canvas": canvas,
            "declared": int(tags.get("GDS_TM_CARRIERS", 0) or 0), "origins_source": origins_source,
            "partial": bool(tags.get("GDS_PARTIAL_MODE")), "valid_h": valid_h or None,
            "lattice_total": lattice_total, "out_of_extent": max(0, lattice_total - len(in_extent))}


def _decode_carrier_window(tm, data, expected: int, hint: list | None = None) -> bool:
    """Decode one already-read carrier window (thread-safe: no rasterio handle involved).

    Named apart from ``_decode_window``, which is the older (tm, dataset, x, y, size, expected)
    helper the extract paths use - reusing the name silently shadowed it and broke them.

    When the carrier does not decode at its natural orientation, the window is retried at a small
    set of rotations before giving up. A rotated product shifts every carrier's content by the same
    angle, so the first carrier that succeeds tells the rest of the image which angle to use - see
    ``verify_carriers``, which caches the hit and tries it first afterwards.
    """
    image = Image.fromarray(np.repeat(data[:, :, None], 3, axis=2).astype(np.uint8), mode="RGB")
    if _decode_ok(tm, image, expected):
        if hint is not None and not hint:
            hint.append(0)
        return True
    locked = hint[0] if hint else None
    for angle in ([locked] if locked else list(_rotation_candidates())):
        rotated = image.rotate(float(angle), resample=Image.Resampling.BILINEAR, expand=False)
        if _decode_ok(tm, rotated, expected):
            if hint is not None and not hint:
                hint.append(angle)
            return True
    return False


# Rotations tried when a carrier fails at its natural orientation. Measured 2026-10-16: a tiled
# product rotated by 5 deg recovers at rot_4, 90 deg at rot_90, 180 deg at rot_180 (6-14 decodes).
# Detection tolerance is about +-1 deg, so a 2-4 deg step covers the small-angle range without a
# fine sweep. 0 is implicit and is always tried first.
ROTATION_STEPS = (90, 180, 270, 2, 4, 5, 6, 10, 15, 30, 45)


def _rotation_candidates() -> tuple[int, ...]:
    """Angles to retry a failed carrier at (GDS_TM_ROTATION=0 disables the sweep entirely)."""
    if int(os.environ.get("GDS_TM_ROTATION", "1") or 0) == 0:
        return ()
    return ROTATION_STEPS


def _read_window_fast(dataset, x: int, y: int, size: int, canvas: tuple[int, int] | None = None) -> np.ndarray:
    """Read a carrier window as uint8, for verification only.

    ``_read_window_padded`` pads into a float64 canvas, which costs 2 MB of allocation and copying
    per window - for 3,834 windows that is ~7.7 GB of memory traffic and it, not the decoding, was
    what made whole-image verification slow (measured 145 ms/carrier). Verification only needs the
    pixels, and the embedded band is uint8, so this reads them directly (no cast, no canvas copy when
    the window is fully inside the raster).
    """
    x0, y0 = int(x), int(y)
    if canvas and (dataset.width, dataset.height) != (int(canvas[0]), int(canvas[1])):
        block = dataset.read(1)
        enlarged = Image.fromarray(block.astype(np.uint8)).resize((int(canvas[0]), int(canvas[1])),
                                                                 Image.Resampling.BILINEAR)
        data = np.asarray(enlarged, dtype=np.uint8)
        return data[y0:y0 + size, x0:x0 + size]
    left, top = max(0, x0), max(0, y0)
    right, bottom = min(dataset.width, x0 + size), min(dataset.height, y0 + size)
    if right - left == size and bottom - top == size:
        return dataset.read(1, window=rasterio.windows.Window(left, top, size, size))
    padded = np.zeros((size, size), dtype=np.uint8)
    if right > left and bottom > top:
        padded[top - y0:bottom - y0, left - x0:right - x0] = dataset.read(
            1, window=rasterio.windows.Window(left, top, right - left, bottom - top))
    return padded


def verify_carriers(source: Path, expected_hex: str, workers: int | None = None) -> dict:
    """Decode EVERY carrier of a tiled GeoTIFF product and report per-carrier coverage.

    Cost is minimised three ways: the carrier origins come from the product's own tags (so there is
    no search at all), the model is loaded once, and only the decode runs in parallel.

    Measured on this host (2 sockets x 18 cores, 27,904x36,352 product, 3,834 carriers):
        reading one carrier window as uint8      2.8 ms   ->   11 s for the whole image
        raw sequential read from the product     456 MB/s
        decode, 8 pool threads x torch 9        59.9 ms   ->  230 s   <-- default, best measured
        decode, 8 x torch 1                    138.7 ms   ->  532 s
        decode, 16 x torch 1                   217.6 ms   ->  834 s
        decode, 32 x torch 1                   144.8 ms   ->  555 s
        decode, one process sequential          70.0 ms   ->  269 s (96-carrier sample, idle host)
        process pool, 8 workers x torch 3      116.0 ms   ->  445 s
    Every shape returned identical per-carrier results, so this is purely about cost. Process pools
    lose because each child re-imports torch and the model; reusing the older float64 read helper
    cost 2 MB of copying per window and a strict read/decode alternation cost 363 s in total, which
    is why the reads are uint8 and why reading a whole tile band or pipelining reads beside the
    decode (both measured, both slower here - 540 s and 555 s) are not used. Total time therefore
    scales with host load: ~2-4 minutes on an idle machine, longer when the box is busy.

    Fast-fail on a mismatched payload: a carrier that does not hold the expected mark costs a full
    rotation sweep (12 decodes instead of 1). Measured on a 14,061-carrier product — right payload
    63 ms/carrier (1 decode), wrong payload 702 ms/carrier (all 12 fail). Passing the wrong hex
    therefore turned a 1.8-minute check into a 1-hour-52-minute one. The product records its own
    payload in ``GDS_TM_PAYLOAD``, so a mismatch is detected before touching any pixels.
    """
    source = Path(source)
    expected = int(expected_hex, 16)
    with rasterio.open(source) as dataset:
        plan = _carrier_plan(dataset)
        width, height = dataset.width, dataset.height
        recorded = (dataset.tags().get(EXPECTED_HEX_TAG) or "").strip()
    if recorded:
        try:
            recorded_int = int(recorded, 16)
        except ValueError:
            try:
                recorded_int = int(recorded)
            except ValueError:
                recorded_int = None
        if recorded_int is not None and recorded_int != expected:
            # 不跑任何解码：结论已经确定，且逐载体跑一遍只会烧 12 倍 CPU。
            return {"detected": False, "method": "per_carrier_verify", "samples": 0, "matched": 0,
                    "failed": 0, "not_watermarkable": 0, "carriers_checked": 0,
                    "coverage": 0.0, "coverage_meets_bar": False,
                    "error": ("payload mismatch: this product carries "
                              f"{recorded_int:08X}, caller asked for {expected:08X}; "
                              "no carrier was decoded (a mismatched payload costs a full "
                              "rotation sweep per carrier)"),
                    "recorded_payload": f"{recorded_int:08X}", "expected": f"{expected:08X}",
                    "seconds": 0.0, "image": [width, height], "fast_fail": True}
    origins = plan["origins"]
    if not origins:
        return {"detected": False, "method": "per_carrier_verify", "samples": 0, "matched": 0,
                "expected": f"{expected:08X}", "error": "no carrier origins recorded and none on the lattice"}
    if workers is None:
        workers = int(os.environ.get("GDS_TM_VERIFY_WORKERS", "0") or 0)
    if workers <= 0:
        # Full-product measurements on this host (14,061 carriers, 36 logical cores):
        #     8 workers x 9 torch threads -> 811.1 s  (13.52 min)  <-- best of everything measured
        #     6 workers x 2               -> 1043.3 s (17.39 min)
        #    12 workers x 2               -> slower still, and only 8% CPU utilisation
        # Short samples (240-900 carriers) all predicted the opposite, so they are not a valid
        # basis here: per-carrier cost falls once the model, the raster cache and the pool are
        # warm, and a burst benchmark never sees the steady state of a 14k-carrier run.
        workers = 8
    workers = max(1, min(workers, len(origins)))
    # threads_per_worker: let GDS_TM_VERIFY_THREADS pin it, else keep total thread count at or
    # below the core count (cores / workers), which is what keeps torch from oversubscribing.
    _pinned = int(os.environ.get("GDS_TM_VERIFY_THREADS", "0") or 0)
    threads = _pinned if _pinned > 0 else max(1, (os.cpu_count() or 4) // workers)
    try:  # leave cores for the decode threads we start ourselves
        import torch
        if not _THREADS_OVERRIDE:
            torch.set_num_threads(threads)
    except Exception:  # noqa: BLE001 - thread pinning is best-effort
        pass
    tm = _make_tm()
    hint: list = []  # 首个成功的旋转角度；锁定后优先尝试，避免每个失败载体都从头扫

    def decoder(data, hint=hint):
        # 每个线程用自己的解码器实例：TrustMark 的 BCH 解码状态挂在实例上（bchecc.py 的
        # self.ECCstate），多线程共享同一实例会互相踩踏 —— 实测 14,061 载体全量跑到中途
        # 抛 IndexError。线程本地实例彻底消除竞态。
        return _decode_carrier_window(_make_tm_threadlocal(), data,
                                      expected=expected, hint=hint)
    batch = max(64, workers * 24)  # bounds memory: a 512^2 window is 256 KB
    matched = 0
    failed: list[list[int]] = []
    not_watermarkable: list[list[int]] = []
    tile = plan["tile"]
    chunks = [origins[index:index + batch] for index in range(0, len(origins), batch)]
    queue: Queue = Queue(maxsize=2)
    read_errors: list[BaseException] = []

    def produce() -> None:
        """Read windows on one thread and hand them over, so reading and decoding overlap.

        Measured on the real 3 GB product: reading and decoding strictly in turn costs 363 s, and
        reading a whole tile band per call is twice as slow on the VeraCrypt volume, so the win is
        to keep the per-window reads and run them beside the decode pool instead of between batches.
        """
        try:
            with rasterio.open(source) as dataset:
                for chunk in chunks:
                    # One reader thread only: a rasterio DatasetReader is not thread safe.
                    queue.put((chunk, [_read_window_fast(dataset, x, y, tile, plan["canvas"]) for x, y in chunk]))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer thread
            read_errors.append(exc)
        finally:
            queue.put(None)

    started = time.time()
    # Pipelining reads beside the decode pool helps on a plain disk and hurts on this VeraCrypt
    # volume (measured 363 s strictly-in-turn vs 555 s pipelined), so it is a switch, not an axiom.
    pipelined = bool(int(os.environ.get("GDS_TM_VERIFY_PIPELINE", "0") or 0))
    if pipelined:
        reader = threading.Thread(target=produce, name="verify-carrier-reader", daemon=True)
        reader.start()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                item = queue.get()
                if item is None:
                    break
                chunk, windows = item
                for (x, y), data, decoded in zip(chunk, windows, pool.map(decoder, windows)):
                    if decoded:
                        matched += 1
                    elif float(np.std(data)) < FLAT_STD:
                        # Same flatness rule the platform's robust GeoTIFF path uses to skip a
                        # carrier: a featureless window cannot hold a recoverable mark.
                        not_watermarkable.append([x, y])
                    else:
                        failed.append([x, y])
        reader.join(timeout=30)
        if read_errors:
            raise RuntimeError(f"carrier window read failed: {read_errors[0]}")
    else:
        with rasterio.open(source) as dataset, ThreadPoolExecutor(max_workers=workers) as pool:
            for chunk in chunks:
                windows = [_read_window_fast(dataset, x, y, tile, plan["canvas"]) for x, y in chunk]
                for (x, y), data, decoded in zip(chunk, windows, pool.map(decoder, windows)):
                    if decoded:
                        matched += 1
                    elif float(np.std(data)) < FLAT_STD:
                        not_watermarkable.append([x, y])
                    else:
                        failed.append([x, y])
    seconds = time.time() - started
    # Same convention the DCT/mesh streams use: `samples` counts the carriers that can hold a mark
    # (flat windows are excluded as not_watermarkable), and a pass means every one of them decoded.
    samples = len(origins) - len(not_watermarkable)
    # The per-carrier pass bar. Defaults to the project-wide >=90% watermark bar, and is a separate
    # measurement from the product-level verdict: it says how evenly the mark is spread across the
    # image, not whether the product carries one.
    bar = float(os.environ.get("GDS_TM_CARRIER_MIN_COVERAGE", "0.90") or 0.90)
    coverage = (matched / len(origins)) if origins else 0.0
    # Failures bucketed by 4096-row bands: a compact way to see whether misses cluster in one region.
    spread: dict[str, int] = {}
    for _, y in failed:
        band_key = str((y // 4096) * 4096)
        spread[band_key] = spread.get(band_key, 0) + 1
    # Two separate verdicts, deliberately not collapsed: `detected` answers "does this product carry
    # the mark" (product level, same meaning the sampled check had), while `coverage_meets_bar`
    # answers the newer question "is the mark present in the whole image, carrier by carrier".
    return {"detected": matched > 0, "method": "per_carrier_verify",
            "samples": samples, "matched": matched, "failed": len(failed),
            "not_watermarkable": len(not_watermarkable), "carriers_checked": len(origins),
            "partial": plan["partial"], "valid_h": plan["valid_h"],
            "lattice_total": plan["lattice_total"], "out_of_extent": plan["out_of_extent"],
            "coverage": round(coverage, 6), "coverage_bar": bar,
            "coverage_meets_bar": coverage >= bar, "failed_by_row_band": spread,
            "failed_carriers": failed[:50], "tile": plan["tile"], "stride": plan["stride"],
            "canvas": list(plan["canvas"]) if plan["canvas"] else None,
            "declared_carriers": plan["declared"], "origins_source": plan["origins_source"],
            "workers": workers, "threads_per_worker": threads,
            "seconds": round(seconds, 1), "per_carrier_ms": round(seconds * 1000 / len(origins), 1),
            "image": [width, height], "expected": f"{expected:08X}"}


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 3:
        print(json.dumps({"error": "usage: runner embed|extract <path> <path> [hex]"}), file=sys.stderr)
        return 2
    command = args[0]
    try:
        if command == "embed" and len(args) == 4:
            result = embed(Path(args[1]), Path(args[2]), int(args[3], 16))
        elif command == "partial-embed" and len(args) == 4:
            result = partial_embed(Path(args[1]), Path(args[2]), int(args[3], 16))
        elif command == "extract" and len(args) == 3:
            result = extract(Path(args[1]), args[2])
        elif command == "image-embed" and len(args) == 4:
            result = embed_image(Path(args[1]), Path(args[2]), int(args[3], 16))
        elif command == "image-extract" and len(args) == 3:
            result = extract_image(Path(args[1]), args[2])
        elif command == "blind-extract" and len(args) in {3, 4}:
            result = extract_blind(Path(args[1]), args[2], int(args[3]) if len(args) == 4 else 800)
        elif command == "verify-carriers" and len(args) in {3, 4}:
            result = verify_carriers(Path(args[1]), args[2], int(args[3]) if len(args) == 4 else None)
        elif command == "geo-restore" and len(args) in {4, 5}:
            # <reference> <probe> <expected_hex> [max_carriers]
            # 必须在 trustmark venv 里跑：torch 只装在那里，服务主进程 import 不到。
            # runner 作为独立脚本执行，不能用相对导入，按文件名导入同目录模块。
            import importlib.util as _ilu  # noqa: PLC0415
            import pathlib as _pl  # noqa: PLC0415
            _mod_path = _pl.Path(__file__).resolve().parent / "trustmark_geo_restore.py"
            _spec = _ilu.spec_from_file_location("_tgr_standalone", _mod_path)
            _tgr = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_tgr)  # type: ignore[union-attr]
            result = _tgr.restore_and_decode(
                Path(args[1]), Path(args[2]), int(args[3], 16),
                max_carriers=int(args[4]) if len(args) == 5 else 64)
        else:
            print(json.dumps({"error": "usage: runner embed|extract|image-embed|image-extract|blind-extract|verify-carriers <src> <dst|hex> [hex|workers]"}), file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - runner boundary must surface any failure as JSON error
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
