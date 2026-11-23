from __future__ import annotations

import hashlib
from pathlib import Path

import shapefile

from .crypto import TextCipher, decrypt_text
from .watermark_policy import marker_digest, reembed_allowed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_digest(path: Path) -> str:
    reader = shapefile.Reader(str(path), encoding="gbk", encodingErrors="replace")
    digest = hashlib.sha256()
    digest.update(str(reader.shapeType).encode())
    for shape in reader.iterShapes():
        digest.update(repr(shape.points).encode("utf-8"))
        digest.update(repr(shape.parts).encode("utf-8"))
    return digest.hexdigest()


def embed_shapefile_watermark(source: Path, destination: Path, text: str, key: bytes,
                              report: dict | None = None) -> tuple[str, str]:
    reader = shapefile.Reader(str(source), encoding="gbk", encodingErrors="replace")
    fields = list(reader.fields[1:])
    field_names = {field[0].upper() for field in fields}
    watermark_name = "_wm"
    existing = watermark_name.upper() in field_names
    if existing and not reembed_allowed():
        raise ValueError(f"source already contains reserved field {watermark_name}（当前为严格取证策略，禁止二次嵌入）")
    # 输入已带 _wm 时覆盖该列的值，不再追加第二列，字段结构保持原样。
    wm_index = next((i for i, field in enumerate(fields)
                     if field[0].upper() == watermark_name.upper()), None)
    previous: list[str] = []
    writer = shapefile.Writer(str(destination), shapeType=reader.shapeType)
    # reader.fields are the raw DBF field tuples ("name", type, size, decimals). Assigning them back
    # is what production does; pyshp then wraps them itself. Passing a Field object here raised
    # AttributeError inside pyshp's own close() and left an empty .dbf ("records (0) vs shapes (1)"),
    # measured 2026-10-18 on a 1.9M-feature shapefile - so the field list is normalised to plain
    # tuples first and the watermark column is appended as a tuple too.
    # pyshp 2.x 只接受普通元组；3.x 的 DbfWriter._header() 需要 Field 对象
    # （传元组会 "'tuple' object has no attribute 'name'"，1 GiB shapefile 会在
    # 第一条记录就失败并留下空 .dbf）。两版都支持的写法：看是否存在 Field 类，
    # 存在就构造对象，不存在就用元组。
    field_ctor = getattr(shapefile, "Field", None)

    def _mk_field(name, ftype, size, decimals=0):
        if field_ctor is None:
            return (name, ftype, size, decimals)
        return field_ctor(name, ftype, size, decimals)

    base_fields = []
    for field in reader.fields[1:]:
        if isinstance(field, (tuple, list)):
            base_fields.append(_mk_field(field[0], field[1], field[2],
                                         field[3] if len(field) > 3 else 0))
        else:  # 已是 Field 对象
            base_fields.append(field)
    if not existing:
        base_fields.append(_mk_field(watermark_name, "C", 254, 0))
    # pyshp 3.x needs Field objects in writer.fields: appending a plain tuple (what 2.x accepted)
    # makes DbfWriter._header() raise "'tuple' object has no attribute 'name'" on the FIRST record,
    # i.e. a 1 GiB shapefile fails immediately with an empty .dbf. Measured 2026-10-18.
    writer.fields = base_fields
    written = 0
    # 复用密钥派生：1.9M 要素逐行 encrypt_text 会重算 1.9M 次 HKDF-SHA256
    # （11.7 µs/次 ≈ 22 s 纯开销）。TextCipher 只派生一次，密文格式不变。
    cipher = TextCipher(key)
    try:
        for shape, record in zip(reader.iterShapes(), reader.iterRecords()):
            values = list(record)
            token = cipher.encrypt(text)
            if existing and wm_index is not None and wm_index < len(values):
                previous.append(values[wm_index])
                values[wm_index] = token
            else:
                values.append(token)
            writer.shape(shape)
            writer.record(*values)
            written += 1
    finally:
        # pyshp's writer.close() raises when a shape went in without its record ("number of records
        # (0) must correspond with the number of shapes (1)") - measured 2026-10-18 on a 1 GiB
        # shapefile built from real features. The cause is the source itself: on that layer
        # iterShapes() yields one more geometry than iterRecords() has rows (11,305 vs 11,304 on
        # lrdl.shp), so the zip() above ends with an unmatched shape. Swallowing the close error
        # would publish a corrupt .dbf, so the mismatch is reported with the counts and the file is
        # left un-published instead.
        shape_count = len(reader.shapes())
        record_count = len(reader.records())
        try:
            writer.close()
        except shapefile.ShapefileException as exc:
            raise ValueError(
                f"shapefile 记录数与几何数不一致（{written} 条已写，源几何 {shape_count} / 源记录 {record_count}）: {exc}"
            ) from exc
    for suffix in (".prj", ".cpg"):
        sidecar = source.with_suffix(suffix)
        if sidecar.exists():
            destination.with_suffix(suffix).write_bytes(sidecar.read_bytes())
    # shapefile.Writer 接受不带扩展名的路径并自行补 .shp/.shx/.dbf；而这里要哈希的是真正写出的
    # 主文件。旧代码对裸路径调用 sha256_file 会 FileNotFoundError，故优先用实写出的 .shp。
    written = destination if destination.is_file() else destination.with_suffix(".shp")
    if report is not None:
        report.update({"rewatermarked": existing,
                       "previous_marker_sha256": marker_digest(previous) if existing else None})
    return sha256_file(written), watermark_name


def extract_shapefile_watermark(path: Path, key: bytes) -> list[str]:
    reader = shapefile.Reader(str(path), encoding="gbk", encodingErrors="replace")
    names = [field[0] for field in reader.fields[1:]]
    try:
        index = next(i for i, name in enumerate(names) if name.lower() == "_wm")
    except StopIteration as exc:
        raise ValueError("watermark field not found") from exc
    return [decrypt_text(record[index], key) for record in reader.iterRecords()]
