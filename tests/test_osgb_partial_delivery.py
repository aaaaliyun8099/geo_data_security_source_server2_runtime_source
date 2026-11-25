"""Review and delivery regression checks for a partial OSGB artifact."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from fastapi import HTTPException

from geo_security.api import ApproveRequest, create_app
from geo_security.processing import ProcessingArtifact


class OsGbPartialDeliveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="gds-osgb-review-")
        self.root = Path(temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(gettempdir()).resolve()))
        self.addCleanup(temporary.cleanup)
        self.token = "test-osgb-review-service-token"
        self.enterContext(patch.dict(os.environ, {
            "GEO_SECURITY_SERVER1_TOKEN": self.token,
            "GDS_BESU_ANCHOR_ENABLED": "0",
            "GDS_SANDBOX_SKIP_STARTUP_MOUNT": "1",
            "GDS_SUPERADMIN_PASSWORD": "test-only-osgb-review",
        }))
        self.enterContext(patch("geo_security.api.PARTIAL_ACCEPT_MIN_COVERAGE", 0.95))
        self.app = create_app(self.root, key=b"f" * 32)
        for store in ("task_store", "loan_store", "circulation_store", "sandbox_policy", "sandbox_pool"):
            self.addCleanup(getattr(self.app.state, store).close)
        self.addCleanup(self.app.state.admin_auth.db.close)
        self.addCleanup(self.app.state.task_executor.shutdown)

    def call(self, method, route, **arguments):
        endpoint = next(
            route_entry.endpoint for route_entry in self.app.routes
            if route_entry.path == route and method in route_entry.methods
        )
        return endpoint(**arguments)

    def create_result(self, result_id="osgb-test", *, reason="OSGB_UNSUPPORTED_TEXTURE_LAYOUT",
                      data_type="OSGB", status="PENDING", verification_status="VERIFIED_PARTIAL",
                      matched=96, metadata_matched=0, supported_verified=True):
        output = self.root / "results" / result_id
        output.mkdir(parents=True)
        (output / "test.osgb").write_bytes(b"synthetic artifact bytes")
        manifest = {
            "data_type": data_type,
            "verification_status": verification_status,
            "approval_policy": "MANUAL_REVIEW_REQUIRED",
            "verification": {
                "verified": False,
                "detected": True,
                "matched": matched,
                "metadata_matched": metadata_matched,
                "supported_subset_verified": supported_verified,
            },
            "partial_processing": {
                "reason": reason,
                "carriers_total": 100,
                "carriers_embedded": 96,
                "carriers_skipped": 4,
            },
        }
        path = output / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        artifact = ProcessingArtifact(result_id, data_type, "test-input", str(output), 100, status)
        self.app.state.task_store.save_artifact(artifact, project_id="project", user_id="user")
        return path

    def assert_rejected(self, code, method, route, **arguments):
        with self.assertRaises(HTTPException) as error:
            self.call(method, route, **arguments)
        self.assertEqual(error.exception.status_code, code)

    def test_layout_partial_needs_review_even_with_high_coverage_and_a_cached_export(self):
        self.create_result()
        result_id = "osgb-test"
        queue = self.call("GET", "/api/v1/results/pending-review")
        self.assertEqual(queue["results"][0]["result_id"], result_id)
        self.assertTrue(queue["results"][0]["requires_manual_review"])
        for method in ("GET", "POST"):
            self.assert_rejected(404, method, "/api/v1/results/{result_id}/export", result_id=result_id)
        self.assert_rejected(404, "GET", "/api/v1/results/{result_id}/files/{filename}",
                             result_id=result_id, filename="test.osgb")
        self.assert_rejected(403, "GET", "/readonly/results/{result_id}/file",
                             result_id=result_id, filename="test.osgb", project_id="project", user_id="user")
        archive = self.root / "previous.zip"
        archive.write_bytes(b"test cached archive")
        self.app.state.task_store.save_export_job({
            "export_id": "old-export", "result_id": result_id, "project_id": "project",
            "user_id": "user", "status": "COMPLETED", "archive_path": str(archive),
        })
        self.assert_rejected(404, "GET", "/api/v1/exports/{export_id}/download", export_id="old-export")

    def test_both_approval_routes_require_confirmation_and_export_as_partial(self):
        for index, (route, arguments) in enumerate((
            ("/api/v1/results/{result_id}/approve", {}),
            ("/internal/results/{result_id}/approve", {"service_token": self.token}),
        )):
            with self.subTest(route=route):
                result_id = f"osgb-test-{index}"
                manifest_path = self.create_result(result_id, matched=0, metadata_matched=1)
                self.assert_rejected(409, "POST", route, result_id=result_id,
                                     request=ApproveRequest(reviewer_id="reviewer"), **arguments)
                self.assert_rejected(422, "POST", route, result_id=result_id,
                                     request=ApproveRequest(reviewer_id="reviewer", manual_review_confirmed=True,
                                                            review_conclusion="   "), **arguments)
                approved = self.call(
                    "POST", route, result_id=result_id, **arguments,
                    request=ApproveRequest(reviewer_id="reviewer", manual_review_confirmed=True,
                                           review_conclusion="Reviewed the preserved tiles"),
                )
                self.assertEqual(approved["approval_status"], "PARTIALLY_APPROVED")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertTrue(manifest["manual_review"]["confirmed"])
                self.assertFalse(manifest["verification"]["verified"])
                response = self.call("GET", "/api/v1/results/{result_id}/export", result_id=result_id)
                with ZipFile(response.path) as archive:
                    self.assertEqual(archive.read("test.osgb"), b"synthetic artifact bytes")
                    self.assertEqual(json.loads(archive.read("manifest.json"))["status"], "PARTIALLY_APPROVED")
                with patch.object(self.app.state.task_executor, "submit", side_effect=lambda fn, *args: fn(*args)):
                    job = self.call("POST", "/api/v1/results/{result_id}/export", result_id=result_id)
                response = self.call("GET", "/api/v1/exports/{export_id}/download", export_id=job["export_id"])
                with ZipFile(response.path) as archive:
                    self.assertEqual(archive.read("test.osgb"), b"synthetic artifact bytes")

    def test_manual_approval_cannot_replace_watermark_verification(self):
        approve = "/api/v1/results/{result_id}/approve"
        request = ApproveRequest(reviewer_id="reviewer", manual_review_confirmed=True, review_conclusion="Reviewed")
        for result_id, overrides in (
            ("no-mark-confirmed", {"matched": 0}),
            ("subset-not-verified", {"supported_verified": False}),
            ("verification-failed", {"verification_status": "VERIFICATION_FAILED"}),
            ("rejected", {"status": "REJECTED"}),
        ):
            with self.subTest(result_id=result_id):
                self.create_result(result_id, **overrides)
                self.assert_rejected(409, "POST", approve, result_id=result_id, request=request)

    def test_dlg_auto_delivery_and_normal_approved_osgb_are_preserved(self):
        self.create_result("dlg-partial", reason="SOURCE_TRUNCATED", data_type="SHP")
        self.create_result("normal-osgb", reason=None, status="APPROVED", verification_status="VERIFIED")
        for result_id in ("dlg-partial", "normal-osgb"):
            with self.subTest(result_id=result_id):
                response = self.call("GET", "/api/v1/results/{result_id}/export", result_id=result_id)
                with ZipFile(response.path) as archive:
                    self.assertIn("test.osgb", archive.namelist())


if __name__ == "__main__":
    unittest.main()
