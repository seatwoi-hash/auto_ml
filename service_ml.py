import torch
from torchvision import datasets, transforms, models
from PIL import Image
import torch.nn.functional as F
import io
import logging



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Eval_photo:

    def __init__(self, model_path: str = 'model.pth'):
        """
        Инициализация классификатора

        Args:
            model_path: путь к файлу модели
        """
        # Загружаем модель
        self.model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        self.model.eval()

        # Трансформации для инференса
        self.transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ✅ ДОБАВЛЕН атрибут class_names
        self.class_names = ['animals', 'city', 'documents', 'nature', 'people']

        logger.info(f"✅ Модель загружена из {model_path}, классов: {len(self.class_names)}")

    async def predict_from_bytes(self, file_bytes: bytes, filename: str = None) -> dict:
        """
        Классифицирует изображение из байтов

        Args:
            file_bytes: байты изображения
            filename: имя файла (опционально, для логирования)

        Returns:
            dict: {
                "class_idx": int,
                "class_name": str,
                "confidence": float,
                "success": bool
            }
        """
        try:
            # Преобразуем байты в изображение
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')

            # Применяем трансформации
            input_tensor = self.transform_eval(image).unsqueeze(0)

            # Инференс
            with torch.no_grad():
                output = self.model(input_tensor)
                predicted_idx = torch.argmax(output, dim=1).item()
                probabilities = F.softmax(output, dim=1)
                confidence = probabilities[0][predicted_idx].item()

            class_name = self.class_names[predicted_idx]

            logger.info(f"📸 Классифицирован {filename or 'файл'}: {class_name} (уверенность: {confidence:.2%})")

            return {
                "success": True,
                "class_idx": predicted_idx,
                "class_name": class_name,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Ошибка классификации {filename or 'файла'}: {e}")
            return {
                "success": False,
                "error": str(e)
            }


    # image = Image.open('photo_path/img.png').convert('RGB')
    # input_tensor = transform_eval(image).unsqueeze(0)
    # output = model(input_tensor)
    #
    # with torch.no_grad():
    #     output = model(input_tensor)
    #     predicted_class = torch.argmax(output, dim=1)
    #     probabilities = F.softmax(output, dim=1)
    #     confidence = probabilities[0][predicted_class].item()
    #
    #
    # logging.info(f"  Класс: {predicted_class.item()}")
    # logging.info(f"  Уверенность: {confidence:.2%}")
    # print(f"  Класс: {predicted_class.item()}")
    # print(f"  Уверенность: {confidence:.2%}")


if __name__ == "__main__":
    # Создаём экземпляр классификатора
    classifier = Eval_photo('model.pth')