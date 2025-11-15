# Digital Inspector - Веб-приложение

Веб-интерфейс для детекции печатей, подписей и QR-кодов на документах.

## Структура проекта

```
digital-inspector/
├── backend/          # FastAPI backend
│   ├── main.py      # Основной файл API
│   └── requirements.txt
├── frontend/         # Frontend (HTML/JS)
│   └── index.html   # Веб-интерфейс
└── project_yolo/    # YOLO тренировка
```

## Быстрый старт

### 1. Запуск Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend будет доступен на `http://localhost:8000`

### 2. Запуск Frontend

Просто откройте `frontend/index.html` в браузере или используйте простой HTTP сервер:

```bash
cd frontend
python -m http.server 8080
```

Затем откройте `http://localhost:8080` в браузере.

## Функционал

- ✅ Загрузка изображений (drag & drop или выбор файла)
- ✅ Детекция печатей, подписей и QR-кодов
- ✅ Визуализация bounding boxes на изображении
- ✅ Настройка параметров детекции (conf, iou, max_det)
- ✅ Поддержка overlapping classes
- ✅ Отображение статистики и метрик
- ✅ Батч-обработка нескольких изображений

## API Документация

После запуска backend, документация доступна по адресу:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Настройка модели

По умолчанию используется модель из `training_results/run_doc_v12/weights/best.pt`.

Чтобы использовать другую модель, установите переменную окружения:

```bash
export MODEL_PATH=/path/to/your/model.pt
python main.py
```

## Классы детекции

- **stamp** (печать) - красный цвет
- **signature** (подпись) - зеленый цвет
- **QR-Code** (QR-код) - синий цвет

## Примеры использования

### Через веб-интерфейс

1. Откройте `frontend/index.html` в браузере
2. Загрузите изображение документа
3. Настройте параметры детекции (опционально)
4. Получите результаты с визуализацией

### Через API

```bash
curl -X POST "http://localhost:8000/api/inference" \
  -F "file=@document.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45"
```

## Решение проблем

### Модель не загружается

Проверьте путь к модели в переменной окружения `MODEL_PATH` или в коде `backend/main.py`.

### CORS ошибки

Если frontend на другом порту, обновите CORS настройки в `backend/main.py`:

```python
allow_origins=["http://localhost:8080"]  # Укажите ваш frontend URL
```

### Медленная обработка

- Убедитесь, что CUDA доступна (проверьте `/health` endpoint)
- Уменьшите `max_det` параметр
- Используйте меньшие изображения


