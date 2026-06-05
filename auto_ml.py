import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import optuna
from optuna.samplers import TPESampler
import ssl
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ssl._create_default_https_context = ssl._create_unverified_context

# Определяем устройство
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

data_dir = "my_dataset/train"

# Трансформации (предобработка)
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Загружаем датасет
full_dataset = datasets.ImageFolder(data_dir, transform=transform_train)
class_names = full_dataset.classes
logger.info(f"Найденные классы: {class_names}")
logger.info(f"Всего изображений: {len(full_dataset)}")

# Разделяем на train (80%) и val (20%)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


# ============================================
# ФУНКЦИЯ ДЛЯ БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ
# ============================================
def objective(trial):
    """
    Целевая функция для байесовской оптимизации.
    """

    # Шаг 2: Генерация набора гиперпараметров
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    dropout_rate = trial.suggest_float('dropout', 0.0, 0.5)

    logger.info(f"Пробуем: batch={batch_size}, lr={lr:.5f}, opt={optimizer_name}, dropout={dropout_rate:.2f}")

    # Шаг 3: Создание загрузчиков
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Шаг 4: Создание пайплайна (модель с новыми параметрами)
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_classes = len(class_names)

    # Добавляем dropout для регуляризации
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)

    # Выбор оптимизатора
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    # Шаг 5: Обучение на валидации (кросс-валидация)
    num_epochs = 5  # Быстрая оценка для HPO

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Шаг 6: Оценка метрики качества
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

    val_acc = correct_val / total_val

    return val_acc


# ============================================
# ЗАПУСК БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ
# ============================================
logger.info("=" * 60)
logger.info("ЗАПУСК БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ ГИПЕРПАРАМЕТРОВ (HPO)")
logger.info("=" * 60)

sampler = TPESampler(seed=42)

study = optuna.create_study(
    direction='maximize',
    sampler=sampler,
    study_name='image_classifier_hpo'
)

# Запускаем оптимизацию (N итераций)
study.optimize(objective, n_trials=15, show_progress_bar=True)

# Получаем лучшие параметры
best_params = study.best_params
best_accuracy = study.best_value

logger.info("=" * 60)
logger.info("РЕЗУЛЬТАТЫ БАЙЕСОВСКОЙ ОПТИМИЗАЦИИ")
logger.info("=" * 60)
logger.info(f"Лучшая точность: {best_accuracy:.4f}")
logger.info(f"Лучшие гиперпараметры:")
for key, value in best_params.items():
    logger.info(f"  {key}: {value}")


logger.info("\n" + "=" * 60)
logger.info("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ С ЛУЧШИМИ ПАРАМЕТРАМИ")
logger.info("=" * 60)

batch_size = best_params['batch_size']
lr = best_params['lr']
optimizer_name = best_params['optimizer']
dropout_rate = best_params['dropout']

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet18(weights='IMAGENET1K_V1')
num_classes = len(class_names)
model.fc = nn.Sequential(
    nn.Dropout(dropout_rate),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

if optimizer_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

criterion = nn.CrossEntropyLoss()
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
        f"Эпоха {epoch + 1}/{num_epochs} | Loss: {running_loss / len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

torch.save(model, 'model.pth')
logger.info("Модель сохранена как 'model.pth'")