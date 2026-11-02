import io

import pytest
from PIL import Image

from feedback import FeedbackStore


def image(color):
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(stream, format="PNG")
    return stream.getvalue()


def test_export_requires_explicit_labels_and_excludes_conflicts(tmp_path):
    store = FeedbackStore(tmp_path / "store")
    first = store.record(image("red"), "city", 0.8, "model-hash")
    store.record(image("blue"), "city", 0.8, "model-hash")
    store.confirm(first, "people")
    excluded = tmp_path / "train"
    excluded.mkdir()
    result = store.export(tmp_path / "snapshot", [excluded])
    assert result["count"] == 1
    assert result["samples"][0]["label"] == "people"
    assert result["independence_status"] == "unverified"
    duplicate = store.record(image("red"), "city", 0.9, "model-hash")
    store.confirm(duplicate, "city")
    assert store.export(tmp_path / "conflict", [excluded])["count"] == 0


def test_excludes_existing_data_and_rejects_invalid_feedback(tmp_path):
    store = FeedbackStore(tmp_path / "store")
    data = image("red")
    sample = store.record(data, "city", 0.8, "hash")
    with pytest.raises(ValueError):
        store.confirm(sample, "../invalid")
    with pytest.raises(KeyError):
        store.confirm("missing", "people")
    store.confirm(sample, "people")
    excluded = tmp_path / "train"
    excluded.mkdir()
    (excluded / "existing.png").write_bytes(data)
    assert store.export(tmp_path / "snapshot", [excluded])["count"] == 0
    with pytest.raises(FileExistsError):
        store.export(tmp_path / "snapshot", [excluded])
