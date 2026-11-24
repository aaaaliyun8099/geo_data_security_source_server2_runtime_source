"""Bounded-memory attribute watermarking for ordinary dBASE III shapefiles."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import mmap
import os
from pathlib import Path
import shutil
import struct
from tempfile import mkstemp
from typing import BinaryIO, Iterator

from .crypto import TextCipher


_DBF_HEADER_LENGTH = 32
_FIELD_LENGTH = 32
_MARKER_NAME = b"_wm"
_MARKER_WIDTH = 254
_HEADER_END = b"\r"
_EOF = b"\x1a"
_BUFFER_SIZE = 4 * 1024 * 1024
_GEOMETRY_HEADER_LENGTH = 100
_SUPPORTED_GEOMETRIES = {1, 3, 5, 8}  # Point, PolyLine, Polygon, MultiPoint.
_INT = struct.Struct("<i")
_INT_PAIR = struct.Struct("<ii")
_INDEX_ENTRY = struct.Struct(">ii")


class _UnsupportedLayout(Exception):
    """Leave supported legacy variants to PyShp."""


@dataclass(frozen=True)
class _DbfLayout:
    header: bytes
    descriptors: bytes
    rows: int
    header_size: int
    row_size: int
    marker: tuple[int, int] | None
    trailer: bytes


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated shapefile or DBF data")
    return data


def _read_dbf(stream: BinaryIO) -> _DbfLayout:
    header = _read_exact(stream, _DBF_HEADER_LENGTH)
    if header[0] != 3 or header[14] or header[15] or header[28]:
        raise _UnsupportedLayout
    rows, header_size, row_size = struct.unpack_from("<IHH", header, 4)
    if header_size <= _DBF_HEADER_LENGTH or row_size == 0:
        raise ValueError("invalid DBF header size or record size")
    descriptors = _read_exact(stream, header_size - _DBF_HEADER_LENGTH)
    if descriptors[-1:] != _HEADER_END or (len(descriptors) - 1) % _FIELD_LENGTH:
        # Other DBF layouts can place additional metadata after the descriptors.
        raise _UnsupportedLayout

    marker = None
    offset = 1  # Deletion flag precedes the attribute values in each row.
    for start in range(0, len(descriptors) - 1, _FIELD_LENGTH):
        field = descriptors[start:start + _FIELD_LENGTH]
        # Match the legacy reader's name handling before checking for an existing
        # marker. In particular, a left-padded name must still enforce reembedding policy.
        raw_name = field[:11]
        end = raw_name.find(b"\0")
        name = raw_name[:end if end != -1 else 10].decode("gbk", errors="replace").lstrip()
        kind, width, decimals = field[11], field[16], field[17]
        if kind not in b"CNFDL" or (kind == ord("C") and decimals):
            raise _UnsupportedLayout
        if not name or not width:
            raise ValueError("invalid DBF field descriptor")
        if name.lower() == "_wm":
            if marker is not None:
                raise ValueError("duplicate shapefile watermark fields")
            if kind != ord("C"):
                raise _UnsupportedLayout
            marker = (offset, width)
        offset += width
    if offset != row_size:
        raise ValueError("DBF field widths do not match its record size")

    content_end = header_size + rows * row_size
    trailer_size = os.fstat(stream.fileno()).st_size - content_end
    if trailer_size not in (0, 1):
        raise ValueError("DBF record count does not match its file size")
    stream.seek(content_end)
    trailer = _read_exact(stream, trailer_size)
    if trailer and trailer != _EOF:
        raise ValueError("invalid DBF end marker")
    stream.seek(header_size)
    return _DbfLayout(header, descriptors, rows, header_size, row_size, marker, trailer)


def _read_geometry_header(stream: BinaryIO) -> tuple[int, int]:
    header = _read_exact(stream, _GEOMETRY_HEADER_LENGTH)
    file_size = os.fstat(stream.fileno()).st_size
    if (struct.unpack_from(">i", header)[0] != 9994
            or struct.unpack_from(">i", header, 24)[0] * 2 != file_size
            or _INT.unpack_from(header, 28)[0] != 1000):
        raise ValueError("invalid shapefile or index header")
    return file_size, _INT.unpack_from(header, 32)[0]


def _validate_geometry(source: Path, index: Path, rows: int) -> None:
    # Match every index entry to its main record. Offsets and lengths are in
    # 16-bit words, as described by the ESRI Shapefile Technical Description.
    with source.open("rb", buffering=_BUFFER_SIZE) as shp, index.open("rb", buffering=_BUFFER_SIZE) as shx:
        file_size, shape_type = _read_geometry_header(shp)
        index_size, index_type = _read_geometry_header(shx)
        if shape_type != index_type or index_size != _GEOMETRY_HEADER_LENGTH + rows * 8:
            raise ValueError("shapefile index and DBF records do not match")
        if shape_type not in _SUPPORTED_GEOMETRIES:
            raise _UnsupportedLayout
        position = _GEOMETRY_HEADER_LENGTH
        for number in range(1, rows + 1):
            offset, size = _INDEX_ENTRY.unpack(_read_exact(shx, 8))
            size *= 2
            end = position + 8 + size
            if offset * 2 != position or size < 4 or end > file_size:
                raise ValueError("shapefile index points outside its record boundaries")
            record_number, record_words = _INDEX_ENTRY.unpack(_read_exact(shp, 8))
            if record_number != number or record_words * 2 != size:
                raise ValueError("shapefile record header does not match its index")
            record_type = _INT.unpack(_read_exact(shp, 4))[0]
            if record_type == 0:
                if size != 4:
                    raise ValueError("invalid null shapefile record")
            elif record_type != shape_type:
                raise ValueError("shapefile record has an unexpected geometry type")
            elif shape_type == 1:
                if size != 20:
                    raise ValueError("invalid shapefile point record")
            elif shape_type == 8:
                if size < 40:
                    raise ValueError("truncated shapefile multipoint record")
                points = _INT.unpack_from(_read_exact(shp, 36), 32)[0]
                if points < 0 or size != 40 + 16 * points:
                    raise ValueError("invalid shapefile multipoint count")
            else:
                if size < 44:
                    raise ValueError("truncated shapefile multipart record")
                parts, points = _INT_PAIR.unpack_from(_read_exact(shp, 40), 32)
                if not 0 <= parts <= points or size != 44 + 4 * parts + 16 * points:
                    raise ValueError("invalid shapefile parts or points count")
                previous = -1
                for part in range(parts):
                    start = _INT.unpack(_read_exact(shp, 4))[0]
                    if (part == 0 and start != 0) or not previous < start < points:
                        raise ValueError("invalid shapefile part index")
                    previous = start
            shp.seek(end)
            position = end
        if position != file_size:
            raise ValueError("shapefile contains geometry missing from its index")


def _rows(stream: BinaryIO, layout: _DbfLayout) -> Iterator[memoryview]:
    records_per_block = max(1, _BUFFER_SIZE // layout.row_size)
    for start in range(0, layout.rows, records_per_block):
        count = min(records_per_block, layout.rows - start)
        block = memoryview(_read_exact(stream, count * layout.row_size))
        for offset in range(0, len(block), layout.row_size):
            _require_active(block[offset])
            yield block[offset:offset + layout.row_size]


def _require_active(state: int) -> None:
    if state == ord("*"):
        # The legacy reader has separate behavior for deleted records.
        raise _UnsupportedLayout
    if state != ord(" "):
        raise ValueError("invalid DBF deletion flag")


def _write_marked_dbf(stream: BinaryIO, target: Path, layout: _DbfLayout, text: str, cipher: TextCipher) -> None:
    header = bytearray(layout.header)
    struct.pack_into("<HH", header, 8, layout.header_size + _FIELD_LENGTH, layout.row_size + _MARKER_WIDTH)
    marker = bytearray(_FIELD_LENGTH)
    marker[:len(_MARKER_NAME)] = _MARKER_NAME
    marker[11], marker[16] = ord("C"), _MARKER_WIDTH
    with target.open("wb", buffering=_BUFFER_SIZE) as output:
        output.write(header)
        output.write(layout.descriptors[:-1])
        output.write(marker)
        output.write(_HEADER_END)
        for row in _rows(stream, layout):
            token = cipher.encrypt(text).encode("ascii")
            if len(token) > _MARKER_WIDTH:
                raise ValueError("encrypted watermark exceeds the DBF field width")
            output.write(row)
            output.write(token.ljust(_MARKER_WIDTH, b" "))
        output.write(layout.trailer)


def try_embed(source: Path, destination: Path, text: str, key: bytes) -> Path | None:
    """Return the written SHP, or defer to the legacy writer for other layouts."""
    dbf, index = source.with_suffix(".dbf"), source.with_suffix(".shx")
    if not dbf.is_file() or not index.is_file():
        return None
    files = {suffix: source.with_suffix(suffix) for suffix in (".prj", ".cpg") if source.with_suffix(suffix).is_file()}
    files.update({".shp": source, ".shx": index, ".dbf": dbf})
    outputs = {suffix: destination.with_suffix(suffix) for suffix in files}
    # Each processing task owns a fresh result directory. Leave existing bundles
    # to the compatibility path, which already handles replacement.
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        return None

    try:
        with ExitStack() as cleanup, dbf.open("rb", buffering=_BUFFER_SIZE) as stream:
            layout = _read_dbf(stream)
            if (layout.marker is not None or layout.rows == 0 or layout.header_size + _FIELD_LENGTH > 0xffff
                    or layout.row_size + _MARKER_WIDTH > 0xffff):
                return None
            _validate_geometry(source, index, layout.rows)

            staged = {}
            for suffix, output in outputs.items():
                descriptor, name = mkstemp(prefix=".shp-wm-", suffix=".tmp", dir=output.parent)
                os.close(descriptor)
                temporary = Path(name)
                cleanup.callback(temporary.unlink, missing_ok=True)
                staged[suffix] = temporary
            _write_marked_dbf(stream, staged[".dbf"], layout, text, TextCipher(key))
            for suffix, original in files.items():
                if suffix != ".dbf":
                    shutil.copyfile(original, staged[suffix])

            published = []
            try:
                # Publish the main SHP last, after all validation and staging.
                for suffix in [name for name in files if name != ".shp"] + [".shp"]:
                    destination_file = outputs[suffix]
                    if destination_file.exists() or destination_file.is_symlink():
                        raise FileExistsError(destination_file)
                    os.replace(staged[suffix], destination_file)
                    published.append(destination_file)
            except Exception:
                for path in published:
                    path.unlink(missing_ok=True)
                raise
            return outputs[".shp"]
    except _UnsupportedLayout:
        return None


def try_extract(source: Path, key: bytes) -> list[str] | None:
    """Authenticate every active record in the eligible fixed-width marker layout."""
    dbf = source.with_suffix(".dbf")
    if not dbf.is_file():
        return None
    try:
        with dbf.open("rb", buffering=_BUFFER_SIZE) as stream:
            layout = _read_dbf(stream)
            if layout.marker is None:
                return None
            start, width = layout.marker
            end = start + width
            cipher = TextCipher(key)
            # Only copy the marker bytes into Python, not the other attributes.
            # A read-only mapping leaves paging to the OS and uses no full-file heap buffer.
            try:
                table = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            except (OSError, OverflowError, ValueError):
                table = None
            if table is not None:
                with table:
                    limit = layout.header_size + layout.rows * layout.row_size
                    values = []
                    for position in range(layout.header_size, limit, layout.row_size):
                        _require_active(table[position])
                        token = table[position + start:position + end].split(b"\0", 1)[0].strip()
                        values.append(cipher.decrypt(token.decode("ascii")))
                    return values
            return [cipher.decrypt(bytes(row[start:end]).split(b"\0", 1)[0].strip().decode("ascii"))
                    for row in _rows(stream, layout)]
    except _UnsupportedLayout:
        return None
