import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from service_ml import CHECKPOINT_VERSION

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
NORMALIZATION = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path = Path("my_dataset/train")
    output_path: Path = Path("model_checkpoint.pth")
    validation_fraction: float = 0.2
    seed: int = 42
    trials: int = 15
    tuning_epochs: int = 5
    final_epochs: int = 10
    num_workers: int = 0


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_datasets(config: TrainingConfig):
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZATION),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZATION),
        ]
    )

    train_source = datasets.ImageFolder(config.data_dir, transform=train_transform)
    validation_source = datasets.ImageFolder(
        config.data_dir,
        transform=validation_transform,
    )
    if train_source.classes != validation_source.classes:
        raise RuntimeError("Train и validation содержат разные классы")

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(train_source), generator=generator).tolist()
    validation_size = int(len(indices) * config.validation_fraction)
    if validation_size == 0 or validation_size == len(indices):
        raise ValueError("Недостаточно изображений для train/validation split")

    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return (
        Subset(train_source, train_indices),
        Subset(validation_source, validation_indices),
        train_source.classes,
        train_source.targets,
        train_indices,
    )


def calculate_class_weights(
    targets: list[int],
    train_indices: list[int],
    num_classes: int,
) -> torch.Tensor:
    train_targets = torch.tensor([targets[index] for index in train_indices])
    counts = torch.bincount(train_targets, minlength=num_classes).float()
    if torch.any(counts == 0):
        raise ValueError("После разделения один из классов отсутствует в train")
    return counts.sum() / (num_classes * counts)


def build_model(num_classes: int, *, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
):
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay,
    )


def train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


def evaluate(model, loader, device) -> tuple[float, list[int], list[int]]:
    model.eval()
    correct = 0
    total = 0
    labels_result: list[int] = []
    predictions_result: list[int] = []

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            labels_result.extend(labels.cpu().tolist())
            predictions_result.extend(predictions.cpu().tolist())

    return correct / total, labels_result, predictions_result


def make_loaders(train_dataset, validation_dataset, batch_size, num_workers):
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(validation_dataset, shuffle=False, **common),
    )


def tune_hyperparameters(
    config,
    train_dataset,
    validation_dataset,
    class_weights,
    num_classes,
    device,
):
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        train_loader, validation_loader = make_loaders(
            train_dataset,
            validation_dataset,
            batch_size,
            config.num_workers,
        )
        model = build_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = create_optimizer(
            model,
            optimizer_name,
            learning_rate,
            weight_decay,
        )

        for epoch in range(config.tuning_epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
            accuracy, _, _ = evaluate(model, validation_loader, device)
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned
        return accuracy

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.seed),
        study_name="image_classifier_hpo",
    )
    study.optimize(objective, n_trials=config.trials, show_progress_bar=True)
    return study.best_params, study.best_value


def save_checkpoint(model, class_names, output_path, best_params, accuracy):
    checkpoint = {
        "format_version": CHECKPOINT_VERSION,
        "architecture": "resnet18",
        "model_state_dict": model.cpu().state_dict(),
        "classifier_head": {"type": "linear"},
        "class_names": class_names,
        "input_size": IMAGE_SIZE,
        "normalization": NORMALIZATION,
        "training": {
            "best_params": best_params,
            "validation_accuracy": accuracy,
        },
    }
    torch.save(checkpoint, output_path)


def main(config: TrainingConfig | None = None) -> None:
    config = config or TrainingConfig()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    set_reproducible_seed(config.seed)
    device = select_device()
    train_dataset, validation_dataset, class_names, targets, train_indices = (
        create_datasets(config)
    )
    class_weights = calculate_class_weights(
        targets,
        train_indices,
        len(class_names),
    )
    logger.info(
        "Датасет: train=%d, validation=%d, классы=%s, устройство=%s",
        len(train_dataset),
        len(validation_dataset),
        class_names,
        device,
    )

    best_params, best_accuracy = tune_hyperparameters(
        config,
        train_dataset,
        validation_dataset,
        class_weights,
        len(class_names),
        device,
    )
    train_loader, validation_loader = make_loaders(
        train_dataset,
        validation_dataset,
        best_params["batch_size"],
        config.num_workers,
    )
    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = create_optimizer(
        model,
        best_params["optimizer"],
        best_params["learning_rate"],
        best_params["weight_decay"],
    )

    for epoch in range(config.final_epochs):
        loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        validation_accuracy, labels, predictions = evaluate(
            model,
            validation_loader,
            device,
        )
        logger.info(
            "Эпоха %d/%d: loss=%.4f train=%.2f%% validation=%.2f%%",
            epoch + 1,
            config.final_epochs,
            loss,
            train_accuracy * 100,
            validation_accuracy * 100,
        )

    logger.info(
        "Classification report:\n%s",
        classification_report(
            labels,
            predictions,
            target_names=class_names,
            zero_division=0,
        ),
    )
    logger.info("Confusion matrix:\n%s", confusion_matrix(labels, predictions))
    save_checkpoint(
        model,
        class_names,
        config.output_path,
        best_params,
        validation_accuracy,
    )
    logger.info("Checkpoint сохранён: %s", config.output_path)


if __name__ == "__main__":
    main()
