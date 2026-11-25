"""Regression coverage for OSGB review outcomes and supported-tile verification."""

from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import unittest
from unittest.mock import patch

from PIL import Image

from geo_security.processing import ProcessingService, sandbox_compute


class PartialOsgbProcessingTests(unittest.TestCase):
    watermark = 0x20261125

    def setUp(self):
        temporary = TemporaryDirectory(prefix="gds-osgb-partial-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(gettempdir()).resolve()))
        self.source = self.root / "source"
        self.source.mkdir()
        self.service = ProcessingService(self.root / "workspace", b"fixture-only")
        with BytesIO() as buffer:
            Image.new("RGB", (32, 32), (28, 70, 128)).save(buffer, format="JPEG")
            self.jpeg = buffer.getvalue()

    def write_tile(self, name: str, *, length_delta: int = 0) -> bytes:
        tile = b"fixture-tile" + (len(self.jpeg) + length_delta).to_bytes(4, "little") + self.jpeg + b"geometry"
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tile)
        return tile

    def process(self):
        # Run the same raster verifier in its normal serial fallback so these fixtures
        # are independent of machine process limits and the test runner's import path.
        with patch.object(sandbox_compute, "enabled", return_value=False), patch(
            "multiprocessing.get_context", side_effect=OSError("test serial fallback")
        ):
            return self.service.process_osgb_tree("project", self.source, self.watermark)

    def test_mixed_tiles_require_review_after_a_supported_watermark_is_confirmed(self):
        good = self.write_tile("a.osgb")
        bad = self.write_tile("nested/b.osgb", length_delta=-1)
        with patch.object(self.service, "_trace_manifest") as trace:
            artifact = self.process()
        output = Path(artifact.output)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        report = json.loads((output / "processing_report.json").read_text(encoding="utf-8"))
        self.assertEqual(artifact.status, "PENDING")
        self.assertEqual(manifest["status"], "PENDING")
        self.assertEqual(manifest["verification_status"], "VERIFIED_PARTIAL")
        self.assertEqual(manifest["approval_policy"], "MANUAL_REVIEW_REQUIRED")
        self.assertEqual(manifest["partial_processing"], {
            "reason": "OSGB_UNSUPPORTED_TEXTURE_LAYOUT",
            "carrier_unit": "osgb_tile",
            "carriers_total": 2,
            "carriers_embedded": 1,
            "carriers_skipped": 1,
            "unsupported_layout_tiles": 1,
            "report": "processing_report.json",
        })
        verification = manifest["verification"]
        self.assertFalse(verification["verified"])
        self.assertTrue(verification["supported_subset_verified"])
        self.assertEqual(verification["metadata_matched"], 1)
        self.assertEqual(verification["sampled_files"], 1)
        self.assertEqual(verification["excluded_files"], 1)
        self.assertEqual(report["unsupported_layout_tiles"][0]["path"], "nested/b.osgb")
        self.assertEqual((output / "nested/b.osgb").read_bytes(), bad)
        self.assertNotEqual((output / "a.osgb").read_bytes(), good)
        trace.assert_called_once_with(output, "WATERMARK_PARTIAL")

    def test_fully_supported_result_keeps_the_normal_verification_status(self):
        self.write_tile("a.osgb")
        with patch.object(self.service, "_trace_manifest") as trace:
            artifact = self.process()
        output = Path(artifact.output)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["verification_status"], "VERIFIED")
        self.assertTrue(manifest["verification"]["verified"])
        self.assertNotIn("partial_processing", manifest)
        trace.assert_called_once_with(output, "WATERMARK_VERIFIED")

    def test_no_marked_tile_leaves_no_partial_artifact(self):
        self.write_tile("bad.osgb", length_delta=-1)
        with self.assertRaisesRegex(ValueError, "no supported tile was watermarked"):
            self.process()
        self.assertEqual(list((self.service.workspace / "projects/project/results").iterdir()), [])

    def test_failed_extraction_on_supported_tile_leaves_no_partial_artifact(self):
        self.write_tile("a.osgb")
        self.write_tile("bad.osgb", length_delta=-1)
        with patch("geo_security.osgb_texture_watermark.extract_osgb_jpeg_watermark", return_value={"detected": False}), patch(
            "geo_security.texture_watermark.extract_texture_watermark_adaptive_bytes", side_effect=ValueError("bad marker")
        ):
            with self.assertRaisesRegex(ValueError, "bad marker"):
                self.process()
        self.assertEqual(list((self.service.workspace / "projects/project/results").iterdir()), [])

    def test_enabled_sandbox_never_uses_direct_subset_verification(self):
        self.write_tile("a.osgb")
        self.write_tile("bad.osgb", length_delta=-1)

        def embed_in_fake_sandbox(operation, source, output, watermark_hex):
            self.assertEqual(operation, "osgb_tree_embed")
            from geo_security.osgb_texture_watermark import embed_osgb_tree

            return embed_osgb_tree(Path(source), Path(output), int(watermark_hex, 16))

        with patch.object(sandbox_compute, "enabled", return_value=True), patch.object(
            sandbox_compute, "run", side_effect=embed_in_fake_sandbox
        ) as sandbox:
            with self.assertRaisesRegex(ValueError, "sandbox OSGB verification does not support"):
                self.service.process_osgb_tree("project", self.source, self.watermark)
        self.assertEqual(sandbox.call_count, 1)
        self.assertEqual(list((self.service.workspace / "projects/project/results").iterdir()), [])
