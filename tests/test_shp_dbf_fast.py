"""SHP byte preservation, fallback, and marker authentication regression tests."""

from pathlib import Path
import struct
from tempfile import TemporaryDirectory, gettempdir
import unittest
from unittest.mock import patch

from cryptography.exceptions import InvalidTag
import shapefile

from geo_security.crypto import decrypt_text
from geo_security.vector_watermark import embed_shapefile_watermark, extract_shapefile_watermark, sha256_file


class ShapefileWatermarkTests(unittest.TestCase):
    key = b"test-only shapefile watermark key"

    def setUp(self):
        temporary = TemporaryDirectory(prefix="gds-shp-fast-")
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(gettempdir()).resolve()))
        self.addCleanup(temporary.cleanup)
        self.output = self.root / "result"
        self.output.mkdir()

    def write_source(self, shape_type=shapefile.POLYLINE):
        source = self.root / "source.shp"
        with shapefile.Writer(str(source), shapeType=shape_type, encoding="gbk") as writer:
            writer.field("name", "C", 18)
            writer.field("amount", "N", 8, 2)
            for number, name in enumerate(("朝阳", "海淀")):
                if shape_type == shapefile.POLYLINEZ:
                    writer.linez([[[0, 0, 10], [1, 1, 20]]])
                else:
                    writer.line([[[0, 0], [1, 1]]])
                writer.record(name, number + 0.25)
        source.with_suffix(".cpg").write_text("GBK", encoding="ascii")
        return source

    def test_fresh_layer_keeps_geometry_and_original_dbf_bytes(self):
        source = self.write_source()
        destination = self.output / source.name
        report = {}
        with patch("geo_security.vector_watermark.shapefile.Writer", side_effect=AssertionError("legacy writer called")):
            digest, name = embed_shapefile_watermark(source, destination, "layer-watermark", self.key, report)
        self.assertEqual((digest, name), (sha256_file(source), "_wm"))
        self.assertEqual(report, {"rewatermarked": False, "previous_marker_sha256": None})
        for suffix in (".shp", ".shx", ".cpg"):
            self.assertEqual(source.with_suffix(suffix).read_bytes(), destination.with_suffix(suffix).read_bytes())

        original = source.with_suffix(".dbf").read_bytes()
        modified = destination.with_suffix(".dbf").read_bytes()
        rows, header, width = struct.unpack_from("<IHH", original, 4)
        self.assertEqual(struct.unpack_from("<IHH", modified, 4), (rows, header + 32, width + 254))
        self.assertEqual(original[32:header - 1], modified[32:header - 1])
        tokens = []
        for number in range(rows):
            original_start = header + number * width
            new_start = header + 32 + number * (width + 254)
            self.assertEqual(original[original_start:original_start + width], modified[new_start:new_start + width])
            token = modified[new_start + width:new_start + width + 254].strip().decode("ascii")
            self.assertEqual(decrypt_text(token, self.key), "layer-watermark")
            tokens.append(token)
        self.assertEqual(len(set(tokens)), rows)
        self.assertEqual(extract_shapefile_watermark(destination, self.key), ["layer-watermark"] * rows)
        with patch("geo_security.shp_dbf_fast.mmap.mmap", side_effect=OSError("mapping unavailable")):
            self.assertEqual(extract_shapefile_watermark(destination, self.key), ["layer-watermark"] * rows)

    def test_corrupt_index_is_rejected_without_publishing(self):
        source = self.write_source()
        index = source.with_suffix(".shx")
        with index.open("r+b") as stream:
            stream.seek(100)
            stream.write(struct.pack(">i", 49))
        with self.assertRaisesRegex(ValueError, "index"):
            embed_shapefile_watermark(source, self.output / source.name, "marker", self.key)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_corrupt_dbf_record_count_is_rejected_without_publishing(self):
        source = self.write_source()
        with source.with_suffix(".dbf").open("r+b") as stream:
            stream.seek(4)
            stream.write(struct.pack("<I", 3))
        with self.assertRaisesRegex(ValueError, "DBF record count"):
            embed_shapefile_watermark(source, self.output / source.name, "marker", self.key)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_unsupported_geometry_uses_compatible_writer(self):
        source = self.write_source(shapefile.POLYLINEZ)
        destination = self.output / source.name
        embed_shapefile_watermark(source, destination, "legacy path", self.key)
        self.assertEqual(extract_shapefile_watermark(destination, self.key), ["legacy path"] * 2)
        with shapefile.Reader(str(destination)) as reader:
            self.assertEqual(list(reader.shape(0).z), [10, 20])

    def test_watermarked_source_preserves_reembed_policy(self):
        source = self.write_source()
        first = self.output / "first.shp"
        second = self.output / "second.shp"
        embed_shapefile_watermark(source, first, "first", self.key)
        with patch.dict("os.environ", {"GDS_STRICT_REEMBED": "1"}):
            with self.assertRaisesRegex(ValueError, "reserved field"):
                embed_shapefile_watermark(first, second, "second", self.key)
        report = {}
        with patch.dict("os.environ", {"GDS_STRICT_REEMBED": "0"}):
            embed_shapefile_watermark(first, second, "second", self.key, report)
        self.assertTrue(report["rewatermarked"])
        self.assertEqual(len(report["previous_marker_sha256"]), 64)
        self.assertEqual(extract_shapefile_watermark(second, self.key), ["second"] * 2)

    def test_fast_verification_rejects_tampered_rows(self):
        source = self.write_source()
        destination = self.output / source.name
        embed_shapefile_watermark(source, destination, "marker", self.key)
        with destination.with_suffix(".dbf").open("r+b") as stream:
            _, header, width = struct.unpack_from("<IHH", stream.read(32), 4)
            stream.seek(header + width - 254 + 4)
            original = stream.read(1)
            stream.seek(-1, 1)
            stream.write(b"B" if original == b"A" else b"A")
        with self.assertRaises(InvalidTag):
            extract_shapefile_watermark(destination, self.key)

    def test_padded_legacy_marker_name_preserves_strict_policy(self):
        source = self.write_source()
        marked = self.output / source.name
        embed_shapefile_watermark(source, marked, "marker", self.key)
        with marked.with_suffix(".dbf").open("r+b") as stream:
            _, header, _ = struct.unpack_from("<IHH", stream.read(32), 4)
            stream.seek(header - 33)
            stream.write(b" _wm".ljust(11, b"\0"))
        with patch.dict("os.environ", {"GDS_STRICT_REEMBED": "1"}):
            with self.assertRaisesRegex(ValueError, "reserved field"):
                embed_shapefile_watermark(marked, self.output / "duplicate.shp", "second", self.key)
        self.assertEqual(extract_shapefile_watermark(marked, self.key), ["marker"] * 2)
