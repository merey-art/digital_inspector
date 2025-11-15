#!/bin/bash
# Скрипт для запуска backend
# Использует основное venv с torch и CUDA

cd "$(dirname "$0")"

# Проверка наличия venv
if [ ! -d "venv" ]; then
    echo "Ошибка: venv не найден. Создайте виртуальное окружение и установите зависимости."
    exit 1
fi

# Активация venv (где установлен torch с CUDA)
source venv/bin/activate

# Проверка установки зависимостей backend
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Установка зависимостей backend..."
    cd backend
    pip install -r requirements.txt
    cd ..
fi

# Запуск backend
cd backend
python main.py

