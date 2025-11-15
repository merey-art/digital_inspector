# YOLOv8 Training Project

Полнофункциональный проект для обучения и использования YOLOv8 (Ultralytics) с поддержкой CUDA, автоматическим подбором batch size, EMA, early stopping и комплексным логированием.

## Структура проекта

```
project_yolo/
├── train.py              # Скрипт обучения модели
├── eval.py               # Оценка модели (mAP, precision, recall)
├── inference.py          # Батч-инференс на изображениях
├── export.py             # Экспорт модели в различные форматы
├── utils/                # Вспомогательные модули
│   ├── autosize.py      # Автоматический подбор batch size
│   ├── ema.py           # Exponential Moving Average
│   ├── metrics.py       # Метрики и визуализация
│   └── tta.py           # Test Time Augmentation
├── configs/              # Конфигурационные файлы
│   ├── model.yaml       # Конфигурация модели
│   └── training.yaml    # Конфигурация обучения
├── requirements.txt      # Зависимости Python
└── README.md            # Этот файл
```

## Установка

### 1. Создание виртуального окружения

**Linux/Mac:**
```bash
python -m venv yolovenv
source yolovenv/bin/activate
```

**Windows:**
```cmd
python -m venv yolovenv
yolovenv\Scripts\activate
```

### 2. Установка PyTorch с CUDA

**Важно:** Сначала установите PyTorch с нужной версией CUDA.

Проверьте версию CUDA на вашей системе:
```bash
nvidia-smi
```

Затем установите PyTorch с соответствующей версией CUDA:
- Для CUDA 11.8: https://pytorch.org/get-started/locally/
- Для CUDA 12.1: https://pytorch.org/get-started/locally/

Пример для CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**⚠️ ВАЖНО:** Если torch с CUDA уже установлен, НЕ запускайте `pip install -r requirements.txt` без проверки, так как это может переустановить CPU версию!

### 3. Установка зависимостей проекта

**Если torch с CUDA уже установлен:**
```bash
# Установите только недостающие зависимости (torch и torchvision закомментированы)
pip install -r requirements.txt
```

**Если torch еще не установлен:**
1. Сначала установите torch с CUDA (см. шаг 2)
2. Затем установите остальные зависимости:
```bash
pip install -r requirements.txt
```

**Проверка установки:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Опциональные зависимости

**Weights & Biases (для логирования экспериментов):**
```bash
pip install wandb
wandb login
```

**TensorRT (для экспорта в TensorRT):**
Установите TensorRT согласно официальной документации NVIDIA.

## Быстрый старт

### Обучение модели

Базовый пример:
```bash
python train.py \
    --data merged_dataset/dataset.yaml \
    --backbone yolov8s.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --project training_results \
    --name run_doc_v1
```

С WandB логированием:
```bash
python train.py \
    --data merged_dataset/dataset.yaml \
    --backbone yolov8s.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16 \
    --project training_results \
    --name run_doc_v1 \
    --wandb
```

С EMA и дополнительными опциями:
```bash
python train.py \
    --data merged_dataset/dataset.yaml \
    --backbone yolov8s.pt \
    --epochs 150 \
    --imgsz 640 \
    --batch 16 \
    --lr 0.001 \
    --optimizer AdamW \
    --ema \
    --mosaic 1.0 \
    --mixup 0.15 \
    --project training_results \
    --name run_doc_v2
```

Автоматический подбор batch size (без указания --batch):
```bash
python train.py \
    --data merged_dataset/dataset.yaml \
    --backbone yolov8s.pt \
    --epochs 100 \
    --project training_results \
    --name run_auto_batch
```

### Оценка модели

```bash
python eval.py \
    --weights training_results/run_doc_v1/weights/best.pt \
    --data merged_dataset/dataset.yaml \
    --conf 0.25 \
    --iou 0.45
```

С TTA (Test Time Augmentation):
```bash
python eval.py \
    --weights training_results/run_doc_v1/weights/best.pt \
    --data merged_dataset/dataset.yaml \
    --tta
```

### Инференс на изображениях

На папке с изображениями:
```bash
python inference.py \
    --weights training_results/run_doc_v1/weights/best.pt \
    --source merged_dataset/images/val \
    --conf 0.25 \
    --iou 0.45
```

На одном изображении:
```bash
python inference.py \
    --weights training_results/run_doc_v1/weights/best.pt \
    --source path/to/image.jpg \
    --conf 0.25
```

**Поддержка перекрывающихся классов (overlapping classes):**

По умолчанию модель настроена для детекции перекрывающихся объектов разных классов (например, подпись под печатью). Это достигается использованием per-class NMS (`agnostic_nms=False`), что позволяет детектировать объекты разных классов даже если они перекрываются.

Если нужно подавить перекрывающиеся детекции:
```bash
python inference.py \
    --weights training_results/run_doc_v1/weights/best.pt \
    --source merged_dataset/images/val \
    --agnostic-nms  # Подавляет перекрывающиеся детекции
```

### Экспорт модели

Экспорт в ONNX:
```bash
python export.py \
    --weights training_results/run_doc_v1/weights/best.pt \
    --formats onnx \
    --imgsz 640
```

Экспорт в несколько форматов:
```bash
python export.py \
    --weights training_results/run_doc_v1/weights/best.pt \
    --formats onnx,torchscript,trt \
    --imgsz 640 \
    --half
```

## Параметры обучения

### Основные параметры

- `--data`: Путь к dataset.yaml (обязательно)
- `--backbone`: Предобученная модель (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
- `--epochs`: Количество эпох (по умолчанию 150)
- `--imgsz`: Размер изображения (по умолчанию 640)
- `--batch`: Размер батча (автоматически подбирается, если не указан)
- `--lr`: Начальный learning rate (по умолчанию 0.001)
- `--optimizer`: Оптимизатор (SGD, Adam, AdamW, NAdam, RAdam, RMSProp)
- `--project`: Директория проекта (по умолчанию training_results)
- `--name`: Имя запуска

### Дополнительные опции

- `--ema`: Использовать Exponential Moving Average
- `--focal`: Использовать Focal Loss (для несбалансированных классов)
- `--mosaic`: Вероятность mosaic аугментации (0.0-1.0)
- `--mixup`: Вероятность mixup аугментации (0.0-1.0)
- `--wandb`: Использовать Weights & Biases для логирования
- `--resume`: Продолжить обучение из checkpoint
- `--patience`: Patience для early stopping (по умолчанию 20)
- `--export`: Экспортировать модель после обучения (onnx,torchscript,trt)
- `--config`: Путь к конфигурационному файлу training.yaml

## Особенности

### Поддержка перекрывающихся классов (Overlapping Classes)

Проект поддерживает детекцию перекрывающихся объектов разных классов (например, подпись под печатью, QR-код на документе). Это реализовано через **per-class NMS** (`agnostic_nms=False`), что означает:

- NMS (Non-Maximum Suppression) применяется **отдельно для каждого класса**
- Объекты разных классов могут перекрываться и будут детектироваться независимо
- Например, если подпись находится под печатью, оба объекта будут обнаружены

**Настройка:**
- По умолчанию: `agnostic_nms=False` (перекрывающиеся классы разрешены)
- Для подавления перекрытий: используйте флаг `--agnostic-nms` в inference/eval
- Максимальное количество детекций: `--max-det 300` (по умолчанию)

**Пример использования:**
```bash
# Детекция с поддержкой overlapping classes (по умолчанию)
python inference.py --weights best.pt --source images/

# Подавление overlapping детекций
python inference.py --weights best.pt --source images/ --agnostic-nms
```

### Автоматический подбор batch size

Скрипт автоматически определяет оптимальный batch size на основе:
- Доступной VRAM GPU
- Размера изображения
- Размера модели

Если `pynvml` установлен, используется точное определение VRAM. Иначе используются консервативные оценки.

### Early Stopping

Автоматическая остановка обучения, если метрика mAP50 не улучшается в течение заданного количества эпох (по умолчанию 20).

### EMA (Exponential Moving Average)

Опциональное использование EMA весов для улучшения обобщения модели. EMA веса сохраняются в `ema_best.pt`.

### Логирование

- Локальные логи: CSV файлы с метриками, графики, confusion matrix
- WandB: Интеграция с Weights & Biases (опционально)
- TensorBoard: Логи автоматически сохраняются ultralytics

### Экспорт моделей

Поддерживаемые форматы:
- ONNX
- TorchScript
- TensorRT (требует установки TensorRT)
- CoreML
- TFLite
- TensorFlow (SavedModel)
- OpenVINO

## Структура результатов обучения

```
training_results/
└── run_name/
    ├── weights/
    │   ├── best.pt          # Лучшие веса
    │   ├── last.pt          # Последние веса
    │   └── ema_best.pt      # EMA веса (если используется)
    ├── results.csv          # Метрики по эпохам
    ├── metrics.json         # Финальные метрики
    ├── confusion_matrix.png # Confusion matrix
    ├── results.png          # Графики обучения
    └── args.yaml            # Параметры обучения
```

## Метрики и визуализация

### Описание метрик

#### Метрики качества детекции

**Precision (Точность)**
- **Определение**: Доля правильно обнаруженных объектов среди всех предсказанных
- **Формула**: `Precision = TP / (TP + FP)`, где:
  - `TP` (True Positives) - правильно обнаруженные объекты
  - `FP` (False Positives) - ложные срабатывания (объекты, которых нет)
- **Интерпретация**: Высокая precision означает, что модель редко ошибается, когда говорит, что объект найден
- **Желаемое значение**: > 0.7 (70%)

**Recall (Полнота)**
- **Определение**: Доля найденных объектов среди всех реально существующих
- **Формула**: `Recall = TP / (TP + FN)`, где:
  - `TP` (True Positives) - правильно обнаруженные объекты
  - `FN` (False Negatives) - пропущенные объекты
- **Интерпретация**: Высокий recall означает, что модель находит большинство объектов
- **Желаемое значение**: > 0.7 (70%)

**mAP50 (mean Average Precision at IoU=0.5)**
- **Определение**: Средняя точность по всем классам при пороге IoU = 0.5
- **Интерпретация**: Основная метрика качества детекции. Показывает, насколько хорошо модель находит объекты с достаточной точностью локализации
- **Желаемое значение**: > 0.7 (70%)
- **Особенность**: Более мягкая метрика, чем mAP50-95

**mAP50-95 (mean Average Precision at IoU=0.5:0.95)**
- **Определение**: Средняя точность по всем классам при порогах IoU от 0.5 до 0.95 с шагом 0.05
- **Интерпретация**: Более строгая метрика, требующая высокой точности локализации. Учитывает не только обнаружение, но и точность границ bounding box
- **Желаемое значение**: > 0.5 (50%)
- **Особенность**: Более строгая метрика, чем mAP50

#### Потери (Losses)

**train/box_loss** - Потеря локализации (bounding box regression)
- Измеряет, насколько точно модель предсказывает координаты bounding box
- Должна уменьшаться в процессе обучения
- Типичные значения: 0.5 - 2.0

**train/cls_loss** - Потеря классификации
- Измеряет, насколько точно модель определяет класс объекта
- Должна уменьшаться в процессе обучения
- Типичные значения: 0.3 - 1.5

**train/dfl_loss** - Distribution Focal Loss
- Специальная потеря для точной локализации в YOLOv8
- Должна уменьшаться в процессе обучения
- Типичные значения: 0.8 - 1.5

**val/box_loss, val/cls_loss, val/dfl_loss** - Валидационные потери
- Аналогичны train losses, но вычисляются на валидационном наборе
- Показывают, насколько хорошо модель обобщается на новых данных
- Если val loss значительно выше train loss - возможен overfitting

#### Learning Rate (LR)

**lr/pg0, lr/pg1, lr/pg2** - Learning rate для разных групп параметров
- `pg0`: Learning rate для backbone (основная сеть)
- `pg1`: Learning rate для neck (промежуточные слои)
- `pg2`: Learning rate для head (головы детекции)
- Обычно используется cosine annealing или другой scheduler для уменьшения LR в процессе обучения

### Визуализация метрик

#### Автоматическая визуализация результатов обучения

Используйте скрипт `visualize_training.py` для создания графиков из CSV файла с результатами:

```bash
# Базовое использование (создает комплексный график)
python visualize_training.py --csv training_results/run_name/results.csv

# Создать все типы графиков
python visualize_training.py --csv training_results/run_name/results.csv --all

# Сохранить в другую директорию
python visualize_training.py --csv results.csv --output my_plots/

# Без сглаживания графиков
python visualize_training.py --csv results.csv --no-smooth
```

#### Типы создаваемых графиков

1. **losses.png** - График всех потерь (train и val)
   - Показывает динамику уменьшения потерь в процессе обучения
   - Помогает выявить overfitting (когда val loss перестает уменьшаться или растет)

2. **metrics.png** - График метрик качества
   - Precision, Recall, mAP50, mAP50-95 по эпохам
   - Показывает улучшение качества детекции

3. **learning_rate.png** - График изменения learning rate
   - Показывает, как изменяется learning rate в процессе обучения
   - Полезно для отладки learning rate scheduler

4. **train_val_comparison.png** - Сравнение train и val потерь
   - Прямое сравнение train и validation потерь на одном графике
   - Помогает выявить overfitting

5. **comprehensive.png** - Комплексный график
   - Все метрики и потери на одной странице
   - Удобно для общего обзора процесса обучения

#### Визуализация метрик по классам

Используйте функции из `utils/metrics.py` для детального анализа:

```python
from project_yolo.utils.metrics import plot_per_class_metrics, plot_detection_statistics

# Визуализация метрик по классам (precision, recall, mAP)
plot_per_class_metrics(per_class_metrics, save_path='per_class_metrics.png')

# Статистика детекций (распределение по классам, confidence, площади bbox)
plot_detection_statistics(detections, class_names, save_path='detection_stats.png')
```

#### Confusion Matrix

Confusion Matrix показывает, какие классы путает модель:
- По диагонали - правильные предсказания
- Вне диагонали - ошибки классификации
- Создается автоматически при валидации (сохраняется в `confusion_matrix.png`)

### Интерпретация результатов

#### Хорошие признаки обучения:
- ✅ Потери (losses) стабильно уменьшаются
- ✅ Метрики (precision, recall, mAP) увеличиваются
- ✅ Val loss близок к train loss (нет overfitting)
- ✅ mAP50 > 0.7, mAP50-95 > 0.5

#### Проблемы и их решения:

**Overfitting (переобучение)**
- Признаки: val loss перестает уменьшаться или растет, train loss продолжает падать
- Решение: увеличить аугментацию, добавить dropout, уменьшить learning rate, использовать early stopping

**Underfitting (недообучение)**
- Признаки: и train, и val loss высокие, метрики низкие
- Решение: увеличить количество эпох, увеличить learning rate, использовать более сложную модель

**Низкий Precision**
- Признаки: много ложных срабатываний
- Решение: увеличить confidence threshold, улучшить качество данных, добавить больше негативных примеров

**Низкий Recall**
- Признаки: много пропущенных объектов
- Решение: уменьшить confidence threshold, улучшить аугментацию, добавить больше примеров редких классов

**Нестабильное обучение**
- Признаки: резкие скачки в метриках и потерях
- Решение: уменьшить learning rate, использовать learning rate scheduler, увеличить batch size

## Решение проблем

### CUDA out of memory

1. Уменьшите batch size: `--batch 8`
2. Уменьшите размер изображения: `--imgsz 512`
3. Используйте меньшую модель: `--backbone yolov8n.pt`

### Медленное обучение на CPU

Обучение на CPU очень медленное. Рекомендуется использовать GPU с CUDA.

### TensorRT экспорт не работает

Убедитесь, что:
1. TensorRT установлен
2. CUDA доступна
3. Версии TensorRT и CUDA совместимы

### WandB не работает

Если WandB не установлен или не настроен, скрипт продолжит работу с локальным логированием.

## Лицензия

Этот проект использует Ultralytics YOLOv8, который распространяется под лицензией AGPL-3.0.

## Поддержка

Для вопросов и проблем:
- Документация Ultralytics: https://docs.ultralytics.com/
- GitHub Issues: создайте issue в репозитории проекта

