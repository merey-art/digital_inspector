#!/bin/bash
set -e

# Функция для корректного завершения
cleanup() {
    echo "Остановка сервисов..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Запуск Backend
cd /app/backend
python main.py &
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

