#!/bin/bash
set -e

# Функция для корректного завершения
cleanup() {
    echo "Остановка сервисов..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Запуск Backend с несколькими воркерами
cd /app/backend
# Используем переменные окружения для настройки
UVICORN_WORKERS=${UVICORN_WORKERS:-4}
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-10}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-20}

export UVICORN_WORKERS
export MAX_CONCURRENT_REQUESTS
export MAX_BATCH_SIZE

# Запуск с uvicorn напрямую для поддержки workers
uvicorn main:app --host 0.0.0.0 --port 8000 --workers $UVICORN_WORKERS &
BACKEND_PID=$!

# Небольшая задержка для запуска backend
sleep 2

# Запуск Frontend
cd /app/frontend
python -m http.server 8080 &
FRONTEND_PID=$!

echo "Backend запущен (PID: $BACKEND_PID)"
echo "Frontend запущен (PID: $FRONTEND_PID)"
echo "Backend API: http://localhost:8000"
echo "Frontend: http://localhost:8080"

# Ожидание завершения
wait

