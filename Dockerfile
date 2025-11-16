# Минимальный оптимизированный Dockerfile для Digital Inspector
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование и установка зависимостей Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi uvicorn[standard] python-multipart pydantic PyMuPDF slowapi && \
    pip install --no-cache-dir -r requirements.txt
# Примечание: Для GPU версии замените cpu на cu121 в строке выше

# Копирование кода приложения
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY training_results/ ./training_results/
COPY docker-entrypoint.sh ./

# Установка прав на скрипт запуска
RUN chmod +x /app/docker-entrypoint.sh

# Открытие портов
EXPOSE 8000 8080

# Переменные окружения
ENV MODEL_PATH=/app/training_results/run_default_15ep2/weights/best.pt
ENV PYTHONUNBUFFERED=1
ENV UVICORN_WORKERS=4
ENV MAX_CONCURRENT_REQUESTS=10
ENV MAX_BATCH_SIZE=20

# Запуск приложения
ENTRYPOINT ["/app/docker-entrypoint.sh"]
