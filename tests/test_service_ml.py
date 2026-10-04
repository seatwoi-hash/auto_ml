import io

import pytest
import torch
from PIL import Image
from torch import nn

from service_ml import PhotoClassifier


class FixedModel:
    def __call__(self, input_tensor):
        assert input_tensor.shape == (1, 3, 2, 2)
        return torch.tensor([[0.0, 1.0, 5.0, 2.0, -1.0]])


def make_classifier():
    classifier = PhotoClassifier.__new__(PhotoClassifier)
    classifier.model = FixedModel()
    classifier.transform_eval = lambda image: torch.zeros(3, 2, 2)
    classifier.class_names = [
        "animals",
        "city",
        "documents",
        "nature",
        "people",
    ]
    return classifier


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_predict_from_bytes_returns_best_class_and_confidence():
    result = await make_classifier().predict_from_bytes(
        image_bytes(), filename="document.png"
    )

    expected_confidence = torch.softmax(
        torch.tensor([0.0, 1.0, 5.0, 2.0, -1.0]), dim=0
    )[2].item()

    assert result["success"] is True
    assert result["class_idx"] == 2
    assert result["class_name"] == "documents"
    assert result["confidence"] == pytest.approx(expected_confidence)


@pytest.mark.asyncio
async def test_predict_from_bytes_rejects_invalid_image():
    result = await make_classifier().predict_from_bytes(
        b"not an image", filename="broken.jpg"
    )

    assert result["success"] is False
    assert result["error"]


def test_classifier_loads_versioned_checkpoint_with_weights_only(monkeypatch, tmp_path):
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    checkpoint = {
        "format_version": 1,
        "architecture": "resnet18",
        "model_state_dict": model.state_dict(),
        "classifier_head": {"type": "linear"},
        "class_names": ["animals", "nature"],
        "input_size": 2,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    checkpoint_path = tmp_path / "model.pth"
    torch.save(checkpoint, checkpoint_path)

    monkeypatch.setattr(
        PhotoClassifier,
        "_build_model",
        staticmethod(lambda architecture, num_classes, classifier_head: model),
    )
    original_load = torch.load
    load_arguments = {}

    def recording_load(*args, **kwargs):
        load_arguments.update(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    classifier = PhotoClassifier(checkpoint_path)

    assert classifier.class_names == ["animals", "nature"]
    assert load_arguments["weights_only"] is True


def test_classifier_rejects_incomplete_checkpoint(monkeypatch):
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="отсутствуют поля"):
        PhotoClassifier("missing-fields.pth")
