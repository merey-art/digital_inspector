#!/bin/bash
# Скрипт для запуска backend с лучшей моделью (best.pt)
# Используется продакшн-модель: run_default_15ep2/weights/best.pt

cd "$(dirname "$0")"

# Используем best.pt (лучшая модель) - продакшн-модель
export MODEL_PATH="$(pwd)/training_results/run_default_15ep2/weights/best.pt"

echo "Используется продакшн-модель: $MODEL_PATH"
echo "Запуск backend..."

source venv/bin/activate
cd backend
python main.py


