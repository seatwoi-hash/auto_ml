from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Subset

from auto_ml import (
    TrainingConfig,
    calculate_class_weights,
    create_datasets,
    save_checkpoint,
)


def create_dataset(root: Path) -> None:
    for class_name in ("animals", "nature"):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(5):
            Image.new("RGB", (8, 8), color=(index * 20, 0, 0)).save(
                class_dir / f"{index}.png"
            )


def test_dataset_split_is_reproducible_and_uses_separate_sources(tmp_path):
    create_dataset(tmp_path)
    config = TrainingConfig(data_dir=tmp_path, seed=7)

    first = create_datasets(config)
    second = create_datasets(config)
    train_dataset, validation_dataset = first[:2]

    assert isinstance(train_dataset, Subset)
    assert train_dataset.indices == second[0].indices
    assert validation_dataset.indices == second[1].indices
    assert train_dataset.dataset is not validation_dataset.dataset
    assert len(train_dataset) == 8
    assert len(validation_dataset) == 2


def test_class_weights_compensate_for_imbalance():
    weights = calculate_class_weights(
        targets=[0, 0, 0, 1],
        train_indices=[0, 1, 2, 3],
        num_classes=2,
    )

    assert weights[1] > weights[0]
    assert weights.tolist() == pytest.approx([2 / 3, 2.0])


def test_checkpoint_contains_safe_runtime_metadata(tmp_path):
    model = torch.nn.Linear(2, 2)
    output_path = tmp_path / "model.pth"

    save_checkpoint(
        model,
        ["animals", "nature"],
        output_path,
        {"batch_size": 16},
        0.75,
    )
    checkpoint = torch.load(output_path, weights_only=True)

    assert checkpoint["format_version"] == 1
    assert checkpoint["architecture"] == "resnet18"
    assert checkpoint["classifier_head"] == {"type": "linear"}
    assert checkpoint["class_names"] == ["animals", "nature"]
    assert checkpoint["training"]["validation_accuracy"] == 0.75
