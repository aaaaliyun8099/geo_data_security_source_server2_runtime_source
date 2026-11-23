from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

from .classification import parse_classification
from .trace_chain import LocalTraceChain


class ControlledImporter:
    """Copy approved external inputs into the encrypted Server2 incoming area."""

    def __init__(self, workspace: Path | str, approved_roots: list[Path | str], trace_chain: LocalTraceChain):
        self.workspace = Path(workspace).resolve()
        self.incoming = self.workspace / "incoming"
        self.incoming.mkdir(parents=True, exist_ok=True)
        self.approved_roots = [Path(root).resolve() for root in approved_roots]
        self.trace_chain = trace_chain

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def import_file(self, source: Path | str, *, project_id: str, classification: str, source_label: str = "approved-local") -> dict:
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if not any(source == root or root in source.parents for root in self.approved_roots):
            raise PermissionError(f"source is outside approved roots: {source}")
        category = parse_classification(classification)
        batch = self.incoming / project_id / uuid.uuid4().hex
        batch.mkdir(parents=True, exist_ok=False)
        destination = batch / source.name
        shutil.copy2(source, destination)
        source_hash = self._sha256(source)
        output_hash = self._sha256(destination)
        if source_hash != output_hash:
            destination.unlink(missing_ok=True)
            raise ValueError("import hash mismatch")
        record = self.trace_chain.append("DATA_IMPORTED", project_id=project_id, classification=category.value,
                                         source_label=source_label, input_hash=source_hash, output_hash=output_hash,
                                         sandbox_user="gds_sbx", destination=str(destination))
        return {"destination": str(destination), "sha256": output_hash, "classification": category.value,
                "trace_record_hash": record["record_hash"]}
