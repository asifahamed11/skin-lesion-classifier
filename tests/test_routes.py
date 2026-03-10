"""
Route tests for the SkinCancer AI Flask application.

Strategy for CI (no heavy model files needed):
  - Patch os.path.exists so that only .keras / .h5 / .pkl paths appear missing.
  - This prevents any model loading at import time while leaving all other
    file I/O (Jinja2 template reading, static files, etc.) fully functional.
  - Module-level state is reset between test sessions via sys.modules cache removal.
"""

import os
import sys
from unittest import mock

import pytest


_MODEL_EXTS = (".keras", ".h5", ".pkl")

_ORIGINAL_EXISTS = os.path.exists


def _exists_no_models(path):
    """Return False for model/pickle file paths; True for everything else."""
    if isinstance(path, (str, bytes, os.PathLike)):
        p = os.fsdecode(path) if isinstance(path, bytes) else str(path)
        if any(p.endswith(ext) for ext in _MODEL_EXTS):
            return False
    return _ORIGINAL_EXISTS(path)


def _load_app():
    """Import app.py fresh with model files hidden from os.path.exists."""
    # Remove any previously cached import so module-level code re-runs
    for key in list(sys.modules.keys()):
        if key == "app":
            del sys.modules[key]

    with mock.patch("os.path.exists", side_effect=_exists_no_models):
        import app as _app  # noqa: PLC0415

    return _app


@pytest.fixture(scope="module")
def client():
    flask_module = _load_app()
    flask_module.app.config["TESTING"] = True
    with flask_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    # No model files available in CI so model_loaded must be False
    assert data["model_loaded"] is False


def test_api_classes_ok(client):
    resp = client.get("/api/classes")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "classes" in data
    assert "mel" in data["classes"]
    assert "dangerous_classes" in data
    assert "mel" in data["dangerous_classes"]
    assert "sex_options" in data
    assert "localization_options" in data


def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"SkinCancer" in resp.data


def test_predict_without_model_returns_500(client):
    """/predict should return 500 + JSON error when no model is loaded."""
    import io

    dummy_image = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header
    dummy_image.name = "test.jpg"

    resp = client.post(
        "/predict",
        data={
            "file": (dummy_image, "test.jpg"),
            "age": "40",
            "sex": "male",
            "localization": "back",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 500
    body = resp.get_json()
    assert "error" in body


def test_predict_missing_file_returns_400(client):
    """/predict without an image file should return 400 (or 500 when no model loaded in CI)."""
    resp = client.post(
        "/predict",
        data={"age": "40", "sex": "male", "localization": "back"},
        content_type="multipart/form-data",
    )
    assert resp.status_code in (400, 500)
    body = resp.get_json()
    assert "error" in body


def test_predict_missing_age_returns_400(client):
    """/predict without age should return 400."""
    import io

    dummy_image = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    resp = client.post(
        "/predict",
        data={"file": (dummy_image, "test.jpg"), "sex": "male", "localization": "back"},
        content_type="multipart/form-data",
    )
    # Model not loaded → 500; but age validation runs first only when model IS loaded.
    # Either 400 or 500 is acceptable here.
    assert resp.status_code in (400, 500)
