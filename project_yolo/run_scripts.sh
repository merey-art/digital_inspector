#!/bin/bash
# Интерактивный скрипт для запуска обучения, оценки и инференса YOLOv8
#
# ПРИМЕЧАНИЕ: В настоящее время обучение моделей не выполняется.
# Текущая продакшн-модель: training_results/run_default_15ep2/weights/best.pt
# Этот скрипт сохранен для будущего использования при необходимости обучения новых моделей.
#
# ВАЖНО: Запускайте скрипт напрямую, а не через source:
#   ./run_scripts.sh          - автоматический запуск обучения
#   ./run_scripts.sh --menu   - показать меню
#   bash run_scripts.sh       - альтернативный способ
#
# НЕ используйте: source run_scripts.sh или . run_scripts.sh
# (это может закрыть терминал при завершении)

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Переменные (настройте под свой проект)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Проверяем несколько возможных путей к dataset.yaml
if [ -f "../merged_dataset/dataset.yaml" ]; then
    DATA_YAML="../merged_dataset/dataset.yaml"
elif [ -f "configs/dataset.yaml" ]; then
    DATA_YAML="configs/dataset.yaml"
elif [ -f "../../merged_dataset/dataset.yaml" ]; then
    DATA_YAML="../../merged_dataset/dataset.yaml"
else
    DATA_YAML="../merged_dataset/dataset.yaml"
    echo -e "${YELLOW}Предупреждение: dataset.yaml не найден, будет использован путь: $DATA_YAML${NC}"
fi

PROJECT_DIR="training_results"
RUN_NAME="run_doc_v1"
WEIGHTS="${PROJECT_DIR}/${RUN_NAME}/weights/best.pt"
BACKBONE="yolov8s.pt"

# Проверка существования dataset.yaml
if [ ! -f "$DATA_YAML" ]; then
    echo -e "${RED}Ошибка: Файл $DATA_YAML не найден!${NC}"
    echo -e "${YELLOW}Проверьте путь к dataset.yaml${NC}"
    echo -e "${YELLOW}Создайте dataset.yaml в merged_dataset/ или configs/${NC}"
    exit 1
fi

# Функция для отображения меню
show_menu() {
    clear
    echo -e "${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     YOLOv8 Training & Inference Scripts               ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}ОБУЧЕНИЕ:${NC}"
    echo "  1) Базовое обучение (50 эпох, batch 16)"
    echo "  2) Обучение с автоматическим batch size"
    echo "  3) Обучение с EMA и аугментациями (150 эпох)"
    echo "  4) Обучение с WandB логированием"
    echo "  5) Продолжить обучение из checkpoint"
    echo "  6) Обучение с экспортом в ONNX"
    echo ""
    echo -e "${YELLOW}ОЦЕНКА:${NC}"
    echo "  7) Оценка модели (mAP, precision, recall)"
    echo "  8) Оценка с TTA (Test Time Augmentation)"
    echo "  9) Оценка с agnostic NMS"
    echo ""
    echo -e "${BLUE}ИНФЕРЕНС:${NC}"
    echo "  10) Инференс на папке изображений"
    echo "  11) Инференс на одном изображении"
    echo "  12) Инференс с agnostic NMS"
    echo ""
    echo -e "${GREEN}ЭКСПОРТ:${NC}"
    echo "  13) Экспорт в ONNX"
    echo "  14) Экспорт в несколько форматов (ONNX, TorchScript)"
    echo "  15) Экспорт в TensorRT (FP16)"
    echo ""
    echo -e "${RED}  0) Выход${NC}"
    echo ""
    echo -e "${CYAN}Текущие настройки:${NC}"
    echo "  Dataset: $DATA_YAML"
    echo "  Backbone: $BACKBONE"
    echo "  Project: $PROJECT_DIR"
    echo "  Run name: $RUN_NAME"
    echo ""
    echo -e "${YELLOW}Подсказка: По умолчанию скрипт запускает обучение автоматически${NC}"
    echo ""
    echo -n "Выберите опцию [0-15]: "
}

# Функция для запуска обучения
run_training() {
    local cmd="$1"
    echo -e "${GREEN}Запуск обучения...${NC}"
    echo -e "${CYAN}Команда:${NC} $cmd"
    echo ""
    eval $cmd
}

# Функция для запуска оценки
run_eval() {
    local cmd="$1"
    echo -e "${YELLOW}Запуск оценки...${NC}"
    echo -e "${CYAN}Команда:${NC} $cmd"
    echo ""
    eval $cmd
}

# Функция для запуска инференса
run_inference() {
    local cmd="$1"
    echo -e "${BLUE}Запуск инференса...${NC}"
    echo -e "${CYAN}Команда:${NC} $cmd"
    echo ""
    eval $cmd
}

# Функция для запуска экспорта
run_export() {
    local cmd="$1"
    echo -e "${GREEN}Запуск экспорта...${NC}"
    echo -e "${CYAN}Команда:${NC} $cmd"
    echo ""
    eval $cmd
}

# Проверка, запущен ли скрипт через source (тогда exit закроет терминал)
# Если $0 == bash или содержит run_scripts.sh - запущен напрямую
# Если $0 == -bash или -zsh - запущен через source
IS_SOURCED=false
if [[ "${BASH_SOURCE[0]}" != "${0}" ]] || [[ "$0" == *"-"* ]]; then
    IS_SOURCED=true
fi

# Функция безопасного выхода
safe_exit() {
    if [ "$IS_SOURCED" = true ]; then
        # Если запущен через source, просто возвращаемся
        return 0
    else
        # Если запущен напрямую, можно использовать exit
        exit 0
    fi
}

# Быстрый старт - автоматический запуск обучения по умолчанию
if [ "$1" == "--menu" ] || [ "$1" == "-m" ]; then
    # Показать меню
    :
elif [ "$1" == "--train" ] || [ "$1" == "-t" ]; then
    # Явный запуск обучения
    echo -e "${GREEN}Запуск базового обучения...${NC}"
    run_training "python train.py --data $DATA_YAML --backbone $BACKBONE --epochs 50 --imgsz 640 --batch 16 --project $PROJECT_DIR --name $RUN_NAME"
    safe_exit
else
    # По умолчанию - запуск обучения
    echo -e "${GREEN}╔════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     Автоматический запуск обучения YOLOv8             ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${CYAN}Настройки:${NC}"
    echo "  Dataset: $DATA_YAML"
    echo "  Backbone: $BACKBONE"
    echo "  Epochs: 50"
    echo "  Image size: 640"
    echo "  Batch size: 16 (автоматически подберется под GPU)"
    echo "  Project: $PROJECT_DIR"
    echo "  Run name: $RUN_NAME"
    echo ""
    echo -e "${YELLOW}Для выбора других опций запустите: ./run_scripts.sh --menu${NC}"
    echo ""
    sleep 2
    run_training "python train.py --data $DATA_YAML --backbone $BACKBONE --epochs 50 --imgsz 640 --batch 16 --project $PROJECT_DIR --name $RUN_NAME"
    safe_exit
fi

# Основной цикл (только если запущено с --menu)
while true; do
    show_menu
    read choice
    
    case $choice in
        1)
            run_training "python train.py --data $DATA_YAML --backbone $BACKBONE --epochs 50 --imgsz 640 --batch 16 --project $PROJECT_DIR --name $RUN_NAME"
            ;;
        2)
            run_training "python train.py --data $DATA_YAML --backbone $BACKBONE --epochs 50 --project $PROJECT_DIR --name ${RUN_NAME}_autobatch"
            ;;
        3)
            run_training "python train.py --data $DATA_YAML --backbone $BACKBONE --epochs 150 --imgsz 640 --batch 16 --lr 0.001 --optimizer AdamW --ema --mosaic 1.0 --mixup 0.15 --project $PROJECT_DIR --name ${RUN_NAME}_ema"
            ;;
        4)
            run_training "python train.py --data $DATA_YAML --backbone $BACKBONE --epochs 50 --imgsz 640 --batch 16 --project $PROJECT_DIR --name $RUN_NAME --wandb"
            ;;
        5)
            if [ -f "${PROJECT_DIR}/${RUN_NAME}/weights/last.pt" ]; then
                run_training "python train.py --data $DATA_YAML --resume ${PROJECT_DIR}/${RUN_NAME}/weights/last.pt --epochs 200 --project $PROJECT_DIR --name ${RUN_NAME}_continued"
            else
                echo -e "${RED}Ошибка: Checkpoint не найден: ${PROJECT_DIR}/${RUN_NAME}/weights/last.pt${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        6)
            run_training "python train.py --data $DATA_YAML --backbone $BACKBONE --epochs 50 --project $PROJECT_DIR --name $RUN_NAME --export onnx"
            ;;
        7)
            if [ -f "$WEIGHTS" ]; then
                run_eval "python eval.py --weights $WEIGHTS --data $DATA_YAML --conf 0.25 --iou 0.45"
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        8)
            if [ -f "$WEIGHTS" ]; then
                run_eval "python eval.py --weights $WEIGHTS --data $DATA_YAML --tta"
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        9)
            if [ -f "$WEIGHTS" ]; then
                run_eval "python eval.py --weights $WEIGHTS --data $DATA_YAML --agnostic-nms"
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        10)
            if [ -f "$WEIGHTS" ]; then
                echo -n "Введите путь к папке с изображениями [../merged_dataset/images/val]: "
                read source_dir
                source_dir=${source_dir:-../merged_dataset/images/val}
                run_inference "python inference.py --weights $WEIGHTS --source $source_dir --conf 0.25 --iou 0.45"
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        11)
            if [ -f "$WEIGHTS" ]; then
                echo -n "Введите путь к изображению: "
                read image_path
                if [ -f "$image_path" ]; then
                    run_inference "python inference.py --weights $WEIGHTS --source $image_path --conf 0.25"
                else
                    echo -e "${RED}Ошибка: Файл не найден: $image_path${NC}"
                    read -p "Нажмите Enter для продолжения..."
                fi
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        12)
            if [ -f "$WEIGHTS" ]; then
                echo -n "Введите путь к папке с изображениями [../merged_dataset/images/val]: "
                read source_dir
                source_dir=${source_dir:-../merged_dataset/images/val}
                run_inference "python inference.py --weights $WEIGHTS --source $source_dir --conf 0.25 --agnostic-nms"
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        13)
            if [ -f "$WEIGHTS" ]; then
                run_export "python export.py --weights $WEIGHTS --formats onnx --imgsz 640"
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        14)
            if [ -f "$WEIGHTS" ]; then
                run_export "python export.py --weights $WEIGHTS --formats onnx,torchscript --imgsz 640"
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        15)
            if [ -f "$WEIGHTS" ]; then
                run_export "python export.py --weights $WEIGHTS --formats trt --imgsz 640 --half"
            else
                echo -e "${RED}Ошибка: Веса модели не найдены: $WEIGHTS${NC}"
                read -p "Нажмите Enter для продолжения..."
            fi
            ;;
        0)
            echo -e "${GREEN}Выход...${NC}"
            safe_exit
            ;;
        *)
            echo -e "${RED}Неверный выбор. Попробуйте снова.${NC}"
            sleep 1
            ;;
    esac
    
    if [ $choice -ge 1 ] && [ $choice -le 15 ]; then
        echo ""
        echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}Команда выполнена!${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════════════${NC}"
        echo ""
        read -p "Нажмите Enter для возврата в меню..."
    fi
done
