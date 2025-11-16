# 🐳 Docker инструкции для Digital Inspector

Этот документ описывает, как собрать и запустить Digital Inspector в Docker контейнере.

## 📋 Требования

- Docker 20.10+
- Docker Compose 2.0+ (опционально, для удобного запуска)
- NVIDIA Docker (для GPU версии, опционально)

## 🚀 Быстрый старт

### Вариант 1: CPU версия (рекомендуется для начала)

```bash
# Сборка образа
docker build -t digital-inspector:latest --target cpu .

# Запуск контейнера
docker run -d \
  --name digital-inspector \
  -p 8000:8000 \
  -v $(pwd)/training_results:/app/training_results:ro \
  -e MODEL_PATH=/app/training_results/run_default_15ep2/weights/best.pt \
  digital-inspector:latest
```

### Вариант 2: CUDA версия (для GPU)

```bash
# Сборка образа
docker build -t digital-inspector:gpu --target cuda .

# Запуск контейнера с GPU
docker run -d \
  --name digital-inspector-gpu \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/training_results:/app/training_results:ro \
  -e MODEL_PATH=/app/training_results/run_default_15ep2/weights/best.pt \
  digital-inspector:gpu
```

### Вариант 3: Использование Docker Compose

```bash
# CPU версия
docker-compose up -d digital-inspector-cpu

# GPU версия (требует NVIDIA Docker)
docker-compose up -d digital-inspector-gpu
```

## 📦 Сборка образов

### CPU версия

```bash
docker build -t digital-inspector:cpu --target cpu .
```

### CUDA версия (GPU)

```bash
docker build -t digital-inspector:gpu --target cuda .
```

## 🔧 Настройка

### Переменные окружения

- `MODEL_PATH` - путь к модели YOLO (по умолчанию: `/app/training_results/run_default_15ep2/weights/best.pt`)
- `PYTHONUNBUFFERED` - отключение буферизации Python (установлено в 1)

### Монтирование томов

Рекомендуется монтировать директорию с моделью:

```bash
-v $(pwd)/training_results:/app/training_results:ro
```

Для разработки можно также монтировать frontend:

```bash
-v $(pwd)/frontend:/app/frontend:ro
```

## 🌐 Доступ к приложению

После запуска контейнера:

- **Backend API**: http://localhost:8000
- **API документация**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

Для доступа к frontend откройте `frontend/index.html` в браузере или используйте простой HTTP сервер:

```bash
cd frontend
python -m http.server 8080
```

Затем откройте http://localhost:8080 в браузере.

## 🔍 Проверка работы

```bash
# Проверка здоровья сервиса
curl http://localhost:8000/health

# Проверка информации о модели
curl http://localhost:8000/

# Просмотр логов
docker logs digital-inspector

# Просмотр логов в реальном времени
docker logs -f digital-inspector
```

## 🛠️ Устранение неполадок

### Проблема: Модель не найдена

**Решение**: Убедитесь, что модель монтирована правильно:

```bash
# Проверка содержимого контейнера
docker exec digital-inspector ls -lh /app/training_results/run_default_15ep2/weights/

# Если модель отсутствует, проверьте локальный путь
ls -lh training_results/run_default_15ep2/weights/best.pt
```

### Проблема: CUDA не работает в GPU версии

**Решение**: 

1. Убедитесь, что установлен NVIDIA Docker:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

2. Проверьте, что контейнер видит GPU:
```bash
docker exec digital-inspector-gpu python -c "import torch; print(torch.cuda.is_available())"
```

### Проблема: Порт уже занят

**Решение**: Измените порт при запуске:

```bash
docker run -d \
  --name digital-inspector \
  -p 8080:8000 \
  ...
```

Или в `docker-compose.yml` измените `8000:8000` на `8080:8000`.

### Проблема: Контейнер не запускается

**Решение**: Проверьте логи:

```bash
docker logs digital-inspector
```

Убедитесь, что:
- Модель существует по указанному пути
- Все зависимости установлены
- Порты не заняты

## 📝 Полезные команды

```bash
# Остановка контейнера
docker stop digital-inspector

# Запуск остановленного контейнера
docker start digital-inspector

# Удаление контейнера
docker rm digital-inspector

# Удаление образа
docker rmi digital-inspector:latest

# Просмотр запущенных контейнеров
docker ps

# Вход в контейнер
docker exec -it digital-inspector bash

# Пересборка образа без кэша
docker build --no-cache -t digital-inspector:latest --target cpu .
```

## 🎯 Production рекомендации

Для production окружения:

1. **Используйте конкретные теги версий** вместо `latest`
2. **Настройте CORS** в `backend/main.py` для конкретных доменов
3. **Используйте reverse proxy** (nginx, traefik) перед контейнером
4. **Настройте логирование** в файлы или централизованную систему
5. **Используйте secrets** для чувствительных данных
6. **Настройте health checks** и мониторинг
7. **Ограничьте ресурсы** контейнера (CPU, память)

Пример с ограничением ресурсов:

```yaml
services:
  digital-inspector:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## 📚 Дополнительная информация

- [Docker документация](https://docs.docker.com/)
- [Docker Compose документация](https://docs.docker.com/compose/)
- [NVIDIA Docker документация](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

