# 🚀 Запуск Backend

## Способ 1: Через скрипт (рекомендуется)

```bash
./start_backend.sh
```

Скрипт автоматически:
- Проверит наличие venv
- Активирует venv
- Установит недостающие зависимости (FastAPI и т.д.)
- Запустит backend

## Способ 2: Вручную

### Шаг 1: Активируйте venv

```bash
source venv/bin/activate
```

### Шаг 2: Установите зависимости backend (если еще не установлены)

```bash
cd backend
pip install -r requirements.txt
```

Это установит только FastAPI и другие недостающие библиотеки. 
**torch с CUDA не будет переустановлен**, так как он закомментирован в requirements.txt.

### Шаг 3: Запустите backend

```bash
# Из директории backend
python main.py

# Или из корня проекта
cd backend && python main.py
```

Backend будет доступен на `http://localhost:8000`

## Проверка работы

### 1. Проверка здоровья API

```bash
curl http://localhost:8000/health
```

Должен вернуть:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "cuda_available": true
}
```

### 2. Откройте документацию API

В браузере откройте:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Проверка загрузки модели

```bash
curl http://localhost:8000/
```

## Решение проблем

### Ошибка: "ModuleNotFoundError: No module named 'fastapi'"

```bash
source venv/bin/activate
cd backend
pip install -r requirements.txt
```

### Ошибка: "Model not found"

Проверьте путь к модели:
```bash
ls -lh training_results/run_doc_v12/weights/best.pt
```

Или установите переменную окружения:
```bash
export MODEL_PATH=/path/to/your/model.pt
python backend/main.py
```

### Ошибка: "CUDA not available"

Проверьте установку torch с CUDA:
```bash
source venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"
```

Если False, переустановите torch с CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Запуск в фоне

```bash
# Запуск в фоне с логированием
nohup python backend/main.py > backend.log 2>&1 &

# Просмотр логов
tail -f backend.log

# Остановка
pkill -f "python backend/main.py"
```

## Изменение порта

По умолчанию backend запускается на порту 8000.

Чтобы изменить порт, отредактируйте `backend/main.py`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Измените порт здесь
```

Или запустите через uvicorn напрямую:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```


