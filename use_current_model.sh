#!/bin/bash
# Скрипт для использования текущей модели (last.pt) во время обучения

cd "$(dirname "$0")"

# Используем last.pt (последний чекпоинт) вместо best.pt
export MODEL_PATH="$(pwd)/training_results/run_default_15ep2/weights/last.pt"

echo "Используется модель: $MODEL_PATH"
echo "Запуск backend с текущей моделью..."

source venv/bin/activate
cd backend
python main.py


