#!/bin/bash
# Примеры команд для запуска обучения, оценки и инференса YOLOv8

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== YOLOv8 Training Scripts ===${NC}\n"

# Переменные (настройте под свой проект)
DATA_YAML="merged_dataset/dataset.yaml"
PROJECT_DIR="training_results"
RUN_NAME="run_doc_v1"
WEIGHTS="${PROJECT_DIR}/${RUN_NAME}/weights/best.pt"

# ============================================
# 1. ОБУЧЕНИЕ
# ============================================

echo -e "${GREEN}1. Базовое обучение:${NC}"
echo "python train.py \\"
echo "    --data ${DATA_YAML} \\"
echo "    --backbone yolov8s.pt \\"
echo "    --epochs 100 \\"
echo "    --imgsz 640 \\"
echo "    --batch 16 \\"
echo "    --project ${PROJECT_DIR} \\"
echo "    --name ${RUN_NAME}"
echo ""

echo -e "${GREEN}2. Обучение с WandB:${NC}"
echo "python train.py \\"
echo "    --data ${DATA_YAML} \\"
echo "    --backbone yolov8s.pt \\"
echo "    --epochs 100 \\"
echo "    --imgsz 640 \\"
echo "    --batch 16 \\"
echo "    --project ${PROJECT_DIR} \\"
echo "    --name ${RUN_NAME} \\"
echo "    --wandb"
echo ""

echo -e "${GREEN}3. Обучение с EMA и аугментациями:${NC}"
echo "python train.py \\"
echo "    --data ${DATA_YAML} \\"
echo "    --backbone yolov8s.pt \\"
echo "    --epochs 150 \\"
echo "    --imgsz 640 \\"
echo "    --batch 16 \\"
echo "    --lr 0.001 \\"
echo "    --optimizer AdamW \\"
echo "    --ema \\"
echo "    --mosaic 1.0 \\"
echo "    --mixup 0.15 \\"
echo "    --project ${PROJECT_DIR} \\"
echo "    --name ${RUN_NAME}_ema"
echo ""

echo -e "${GREEN}4. Обучение с автоматическим batch size:${NC}"
echo "python train.py \\"
echo "    --data ${DATA_YAML} \\"
echo "    --backbone yolov8s.pt \\"
echo "    --epochs 100 \\"
echo "    --project ${PROJECT_DIR} \\"
echo "    --name ${RUN_NAME}_autobatch"
echo ""

echo -e "${GREEN}5. Продолжить обучение из checkpoint:${NC}"
echo "python train.py \\"
echo "    --data ${DATA_YAML} \\"
echo "    --resume ${PROJECT_DIR}/${RUN_NAME}/weights/last.pt \\"
echo "    --epochs 200 \\"
echo "    --project ${PROJECT_DIR} \\"
echo "    --name ${RUN_NAME}_continued"
echo ""

echo -e "${GREEN}6. Обучение с экспортом в ONNX:${NC}"
echo "python train.py \\"
echo "    --data ${DATA_YAML} \\"
echo "    --backbone yolov8s.pt \\"
echo "    --epochs 100 \\"
echo "    --project ${PROJECT_DIR} \\"
echo "    --name ${RUN_NAME} \\"
echo "    --export onnx"
echo ""

# ============================================
# 2. ОЦЕНКА
# ============================================

echo -e "${YELLOW}7. Оценка модели (с поддержкой overlapping classes):${NC}"
echo "python eval.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --data ${DATA_YAML} \\"
echo "    --conf 0.25 \\"
echo "    --iou 0.45"
echo ""

echo -e "${YELLOW}8. Оценка с TTA:${NC}"
echo "python eval.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --data ${DATA_YAML} \\"
echo "    --tta"
echo ""

echo -e "${YELLOW}8a. Оценка с подавлением overlapping детекций:${NC}"
echo "python eval.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --data ${DATA_YAML} \\"
echo "    --agnostic-nms"
echo ""

# ============================================
# 3. ИНФЕРЕНС
# ============================================

echo -e "${BLUE}9. Инференс на папке изображений (с поддержкой overlapping classes):${NC}"
echo "python inference.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --source merged_dataset/images/val \\"
echo "    --conf 0.25 \\"
echo "    --iou 0.45"
echo ""

echo -e "${BLUE}10. Инференс на одном изображении:${NC}"
echo "python inference.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --source path/to/image.jpg \\"
echo "    --conf 0.25"
echo ""

echo -e "${BLUE}10a. Инференс с подавлением overlapping детекций:${NC}"
echo "python inference.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --source merged_dataset/images/val \\"
echo "    --conf 0.25 \\"
echo "    --agnostic-nms"
echo ""

# ============================================
# 4. ЭКСПОРТ
# ============================================

echo -e "${GREEN}11. Экспорт в ONNX:${NC}"
echo "python export.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --formats onnx \\"
echo "    --imgsz 640"
echo ""

echo -e "${GREEN}12. Экспорт в несколько форматов:${NC}"
echo "python export.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --formats onnx,torchscript \\"
echo "    --imgsz 640"
echo ""

echo -e "${GREEN}13. Экспорт в TensorRT (FP16):${NC}"
echo "python export.py \\"
echo "    --weights ${WEIGHTS} \\"
echo "    --formats trt \\"
echo "    --imgsz 640 \\"
echo "    --half"
echo ""

echo -e "${BLUE}=== Конец примеров ===${NC}"
echo ""
echo "Примечание: Замените пути и параметры на свои значения."

