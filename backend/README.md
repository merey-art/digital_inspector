# Digital Inspector Backend

FastAPI backend для детекции печатей, подписей и QR-кодов на документах.

## Установка

**Важно:** Убедитесь, что вы используете то же виртуальное окружение, где установлен torch с CUDA.

```bash
# Активируйте основное venv (где установлен torch с CUDA)
source ../venv/bin/activate

# Установите только недостающие зависимости для backend
cd backend
pip install -r requirements.txt
```

**Примечание:** `torch`, `torchvision` и `ultralytics` уже должны быть установлены в основном venv. В `requirements.txt` они закомментированы, чтобы избежать переустановки.

## Запуск

```bash
# Простой запуск
python main.py

# Или через uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Переменные окружения

- `MODEL_PATH` - путь к модели (по умолчанию: `../training_results/run_doc_v12/weights/best.pt`)

## API Endpoints

### GET `/`
Информация о API

### GET `/health`
Проверка здоровья сервиса

### POST `/api/inference`
Инференс на одном изображении

**Параметры:**
- `file` (file): Изображение
- `conf` (float, default=0.25): Порог уверенности
- `iou` (float, default=0.45): Порог IoU
- `agnostic_nms` (bool, default=False): Использовать agnostic NMS
- `max_det` (int, default=300): Максимальное количество детекций
- `draw_boxes_flag` (bool, default=True): Рисовать bounding boxes

**Ответ:**
```json
{
  "success": true,
  "detections": [...],
  "num_detections": 3,
  "image_size": {"width": 1920, "height": 1080},
  "inference_time": 0.123,
  "model_info": {...},
  "annotated_image": "data:image/png;base64,..."
}
```

### POST `/api/batch-inference`
Батч-инференс на нескольких изображениях

## Пример использования

```python
import requests

with open('document.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/inference',
        files={'file': f},
        data={
            'conf': 0.25,
            'iou': 0.45,
            'agnostic_nms': False
        }
    )
    result = response.json()
    print(f"Найдено объектов: {result['num_detections']}")
```

