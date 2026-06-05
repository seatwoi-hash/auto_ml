import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import ssl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ssl._create_default_https_context = ssl._create_unverified_context

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_dir = "my_dataset/train"

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # все фото к одному размеру
    transforms.RandomHorizontalFlip(p=0.5),  # случайное отражение (увеличивает данные)
    transforms.RandomRotation(degrees=15),  # случайный поворот
    transforms.ToTensor(),  # превращаем в числа
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(data_dir, transform=transform_train)

class_names = full_dataset.classes
logger.info(f"Найденные классы: {class_names}")
logger.info(f"Всего изображений: {len(full_dataset)}")

# Разделяем на train (80%) и val (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

logger.info(f"Train: {train_size} фото | Val: {val_size} фото")

model = models.resnet18(weights='IMAGENET1K_V1')
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    # Обучение
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)

    train_acc = 100 * correct_train / total_train

    # Валидация
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = 100 * correct_val / total_val

    logger.info(
        f"Эпоха {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

# --- 7. Сохраняем модель ---
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names
}, 'photo_classifier.pth')

torch.save(model, 'model.pth')

logger.info("Модель сохранена как 'photo_classifier.pth'")
logger.info(f"Классы: {class_names}")
