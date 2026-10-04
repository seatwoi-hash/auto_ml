import asyncio
import io
import logging
from pathlib import Path
from typing import TypedDict

import torch
import torch.nn.functional as functional
from PIL import Image
from torch import nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1
SUPPORTED_ARCHITECTURES = {"resnet18"}


class Prediction(TypedDict, total=False):
    success: bool
    class_idx: int
    class_name: str
    confidence: float
    error: str


class PhotoClassifier:
    def __init__(self, model_path: str | Path = "model.pth") -> None:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
        self._validate_checkpoint(checkpoint)

        self.class_names: list[str] = checkpoint["class_names"]
        self.model = self._build_model(
            checkpoint["architecture"],
            len(self.class_names),
            checkpoint["classifier_head"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        input_size = checkpoint["input_size"]
        normalization = checkpoint["normalization"]
        self.transform_eval = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalization["mean"],
                    std=normalization["std"],
                ),
            ]
        )
        logger.info(
            "Модель загружена из %s, классов: %d",
            model_path,
            len(self.class_names),
        )

    @staticmethod
    def _build_model(
        architecture: str,
        num_classes: int,
        classifier_head: dict,
    ) -> nn.Module:
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Неподдерживаемая архитектура: {architecture}")

        model = models.resnet18(weights=None)
        if classifier_head["type"] == "dropout_linear":
            model.fc = nn.Sequential(
                nn.Dropout(float(classifier_head["dropout"])),
                nn.Linear(model.fc.in_features, num_classes),
            )
        elif classifier_head["type"] == "linear":
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError("Неподдерживаемый classifier head")
        return model

    @staticmethod
    def _validate_checkpoint(checkpoint: object) -> None:
        if not isinstance(checkpoint, dict):
            raise ValueError("Некорректный формат checkpoint")

        required_fields = {
            "format_version",
            "architecture",
            "model_state_dict",
            "classifier_head",
            "class_names",
            "input_size",
            "normalization",
        }
        missing_fields = required_fields - checkpoint.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"В checkpoint отсутствуют поля: {missing}")
        if checkpoint["format_version"] != CHECKPOINT_VERSION:
            raise ValueError("Неподдерживаемая версия checkpoint")
        if checkpoint["architecture"] not in SUPPORTED_ARCHITECTURES:
            raise ValueError("Неподдерживаемая архитектура checkpoint")
        if not checkpoint["class_names"]:
            raise ValueError("Checkpoint не содержит классы")
        if not isinstance(checkpoint["classifier_head"], dict):
            raise ValueError("Некорректное описание classifier head")

    async def predict_from_bytes(
        self,
        file_bytes: bytes,
        filename: str | None = None,
    ) -> Prediction:
        return await asyncio.to_thread(self._predict, file_bytes, filename)

    def _predict(
        self,
        file_bytes: bytes,
        filename: str | None = None,
    ) -> Prediction:
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            input_tensor = self.transform_eval(image).unsqueeze(0)

            with torch.inference_mode():
                output = self.model(input_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()
                confidence = functional.softmax(output, dim=1)[0, predicted_idx].item()

            class_name = self.class_names[predicted_idx]
            logger.info(
                "Классифицирован %s: %s (уверенность: %.2f%%)",
                filename or "файл",
                class_name,
                confidence * 100,
            )
            return {
                "success": True,
                "class_idx": predicted_idx,
                "class_name": class_name,
                "confidence": confidence,
            }
        except (OSError, ValueError, IndexError, RuntimeError) as error:
            logger.warning(
                "Ошибка классификации %s: %s",
                filename or "файла",
                error,
            )
            return {"success": False, "error": str(error)}
