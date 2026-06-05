# Dockerfile
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Создаем папку для базы данных
RUN mkdir -p /data

# Открываем порт
EXPOSE 8877

# Команда для запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8877"]