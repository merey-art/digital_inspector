#!/bin/bash
# Скрипт для переключения между моделями

cd "$(dirname "$0")"

RUN_DIR="training_results/run_doc_v12/weights"

echo "Доступные модели:"
echo "1) last.pt - Последний чекпоинт (текущая эпоха)"
echo "2) best.pt - Лучшая модель по метрикам"
echo "3) epoch10.pt - Чекпоинт 10 эпохи"
echo "4) epoch0.pt - Чекпоинт 0 эпохи"
echo ""
read -p "Выберите модель (1-4) или введите путь: " choice

case $choice in
    1)
        MODEL_FILE="$RUN_DIR/last.pt"
        ;;
    2)
        MODEL_FILE="$RUN_DIR/best.pt"
        ;;
    3)
        MODEL_FILE="$RUN_DIR/epoch10.pt"
        ;;
    4)
        MODEL_FILE="$RUN_DIR/epoch0.pt"
        ;;
    *)
        MODEL_FILE="$choice"
        ;;
esac

if [ ! -f "$MODEL_FILE" ]; then
    echo "Ошибка: Модель не найдена: $MODEL_FILE"
    exit 1
fi

export MODEL_PATH="$(pwd)/$MODEL_FILE"
echo "Используется модель: $MODEL_PATH"
echo ""
echo "Запуск backend:"
echo "export MODEL_PATH=\"$MODEL_PATH\""
echo "cd backend && python main.py"
echo ""
echo "Или используйте скрипт:"
echo "./use_current_model.sh"


