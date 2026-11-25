"""OSGB JPEG record boundary and tile retention regressions."""

import json
import os
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import unittest
from unittest.mock import patch

from PIL import Image

from geo_security import osgb_texture_watermark as osgb
from geo_security.texture_watermark import extract_texture_watermark_adaptive_bytes


WATERMARK = 0x20261125
PREFIX = b"scene-prefix"
SUFFIX = b"scene-geometry"


def jpeg(size=(32, 32)):
    buffer = BytesIO()
    Image.new("RGB", size, (30, 60, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


def record(image, trailer=b"", declared=None):
    size = len(image) + len(trailer) if declared is None else declared
    return size.to_bytes(4, "little") + image + trailer


class OsgbJpegLayoutTests(unittest.TestCase):
    def test_canonical_record_still_rewrites_length(self):
        original = jpeg()
        data = PREFIX + record(original) + SUFFIX
        replacement = osgb.embed_osgb_jpeg_watermark(original, WATERMARK)
        self.assertEqual(
            osgb.rewrite_osgb_bytes(data, osgb.embed_osgb_jpeg_watermark, WATERMARK),
            PREFIX + record(replacement) + SUFFIX,
        )

    def test_padded_tile_preserves_all_offsets_including_neighbor_record(self):
        original = jpeg()
        large_slot = osgb._pad_jpeg(original, len(original) + 96)
        trailer = bytes(8)
        between = b"between-geometry"
        data = PREFIX + record(large_slot) + between + record(large_slot, trailer) + SUFFIX
        replacement = osgb.embed_osgb_jpeg_watermark(original, WATERMARK)
        expected_jpeg = osgb._pad_jpeg(replacement, len(large_slot))
        result = osgb.rewrite_osgb_bytes(data, lambda *_: replacement, WATERMARK)
        self.assertEqual(len(result), len(data))
        self.assertEqual(result, PREFIX + record(expected_jpeg) + between + record(expected_jpeg, trailer) + SUFFIX)
        for start, end in osgb._jpeg_spans(result):
            with Image.open(BytesIO(result[start:end])) as decoded:
                decoded.load()
                self.assertEqual(decoded.size, (32, 32))
            self.assertTrue(osgb.extract_osgb_jpeg_watermark(result[start:end], WATERMARK)["detected"])

    def test_jpeg_padding_is_valid_for_minimum_and_multi_segment_sizes(self):
        original = jpeg()
        for slack in (4, 5, 65538):
            with self.subTest(slack=slack):
                padded = osgb._pad_jpeg(original, len(original) + slack)
                self.assertEqual(list(osgb._jpeg_spans(padded)), [(0, len(padded))])
                self.assertEqual(len(padded), len(original) + slack)
                with Image.open(BytesIO(padded)) as decoded:
                    decoded.load()

    def test_nonzero_short_and_out_of_bounds_records_fail_before_any_transform(self):
        original = jpeg()
        for invalid in (
            record(original, b"nonzero-trailer"),
            record(original, declared=len(original) - 1),
            record(original, declared=len(original) + 100_000),
            original,  # The preceding JPEG ends where this record starts.
        ):
            with self.subTest(invalid_length=len(invalid)):
                prefix = PREFIX + record(original)
                with self.assertRaises(osgb._UnsupportedJpegLayout) as caught:
                    osgb.rewrite_osgb_bytes(prefix + invalid, self.unexpected_transform, WATERMARK)
                self.assertEqual(caught.exception.jpeg_length, len(original))
                self.assertGreaterEqual(caught.exception.offset, len(prefix))

    def unexpected_transform(self, *_):
        self.fail("The tile should be checked before calling any encoder")

    def test_com_detection_requires_an_actual_com_marker(self):
        original = jpeg()
        marker = osgb._small_jpeg_marker(WATERMARK)
        app_segment = b"\xff\xef" + (len(marker) + 2).to_bytes(2, "big") + marker
        self.assertFalse(osgb.extract_osgb_jpeg_watermark(original[:2] + app_segment + original[2:], WATERMARK)["detected"])
        self.assertFalse(osgb.extract_osgb_jpeg_watermark(original + marker, WATERMARK)["detected"])
        marked = osgb.embed_osgb_jpeg_watermark(original, WATERMARK)
        self.assertTrue(osgb.extract_osgb_jpeg_watermark(marked, WATERMARK)["detected"])
        self.assertFalse(osgb.extract_osgb_jpeg_watermark(marked, WATERMARK + 1)["detected"])


class OsgbTileRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="gds-osgb-layout-")
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(gettempdir()).resolve()))
        self.addCleanup(temporary.cleanup)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "result"

    def assert_mixed_result(self, executor, tiles):
        original = jpeg()
        valid = PREFIX + record(original) + SUFFIX
        unsupported = PREFIX + record(original) + record(original, b"unexpected") + SUFFIX
        folder = self.source / "nested"
        folder.mkdir()
        (folder / "unsupported.osgb").write_bytes(unsupported)
        for number in range(tiles - 1):
            (self.source / f"tile-{number:03}.osgb").write_bytes(valid)
        with patch.dict(os.environ, {"GDS_OSGB_EXECUTOR": executor, "GDS_OSGB_WORKERS": "2", "GDS_OSGB_PROCESSES": "2"}):
            with self.assertLogs(osgb.__name__, level="WARNING"):
                self.assertEqual(osgb.embed_osgb_tree(self.source, self.output, WATERMARK), 0)
        self.assertEqual((self.output / "nested/unsupported.osgb").read_bytes(), unsupported)
        output_tile = (self.output / "tile-000.osgb").read_bytes()
        start, end = next(osgb._jpeg_spans(output_tile))
        self.assertTrue(osgb.extract_osgb_jpeg_watermark(output_tile[start:end], WATERMARK)["detected"])
        report = json.loads((self.output / "processing_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["files_processed"], tiles)
        self.assertEqual(report["marked_files"], tiles - 1)
        self.assertEqual(report["watermarked_files"], 0)
        self.assertEqual(report["watermark_skipped_files"], 1)
        self.assertEqual(report["small_texture_fallbacks"], tiles - 1)
        self.assertEqual(report["unsupported_layout_tiles"], [{
            "path": "nested/unsupported.osgb",
            "offset": len(PREFIX + record(original)) + 4,
            "declared_length": len(original) + len(b"unexpected"),
            "jpeg_length": len(original),
            "reason": "JPEG record has an unrecognized nonzero trailer",
        }])

    def test_thread_executor_preserves_unsupported_tile(self):
        self.assert_mixed_result("thread", 3)

    def test_process_executor_preserves_unsupported_tile(self):
        self.assert_mixed_result("process", 200)

    def test_padded_tile_embeds_real_pixel_watermark_without_shifting_trailer(self):
        texture = jpeg((256, 256))
        slot = osgb._pad_jpeg(texture, len(texture) + 16384)
        trailer = bytes(8)
        original = PREFIX + record(slot, trailer) + SUFFIX
        source = self.source / "tile.osgb"
        destination = self.output / "tile.osgb"
        source.write_bytes(original)
        self.assertEqual(osgb.embed_osgb_file(source, destination, WATERMARK), 1)
        result = destination.read_bytes()
        start = len(PREFIX) + 4
        end = start + len(slot)
        self.assertEqual(len(result), len(original))
        self.assertEqual(result[:start], original[:start])
        self.assertEqual(result[end:], trailer + SUFFIX)
        self.assertEqual(list(osgb._jpeg_spans(result)), [(start, end)])
        self.assertTrue(extract_texture_watermark_adaptive_bytes(result[start:end], WATERMARK)["detected"])

    def test_padded_tile_with_encoder_failure_is_not_reported_as_layout_skip(self):
        image = jpeg((256, 256))
        (self.source / "tile.osgb").write_bytes(PREFIX + record(image, bytes(8)) + SUFFIX)
        with patch.object(osgb, "_watermark_jpeg_fixed_unbounded", side_effect=RuntimeError("encoder failed")):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                osgb.embed_osgb_tree(self.source, self.output, WATERMARK)
        self.assertFalse((self.output / "tile.osgb").exists())

    def test_padded_tile_with_insufficient_jpeg_slot_is_preserved(self):
        image = jpeg()
        original = PREFIX + record(image, bytes(8)) + SUFFIX
        (self.source / "tile.osgb").write_bytes(original)
        with self.assertLogs(osgb.__name__, level="WARNING"):
            self.assertEqual(osgb.embed_osgb_tree(self.source, self.output, WATERMARK), 0)
        report = json.loads((self.output / "processing_report.json").read_text(encoding="utf-8"))
        self.assertEqual((self.output / "tile.osgb").read_bytes(), original)
        self.assertEqual(report["marked_files"], 0)
        self.assertEqual(report["unsupported_layout_tiles"][0]["path"], "tile.osgb")
        self.assertIn("fixed OSGB slot", report["unsupported_layout_tiles"][0]["reason"])
