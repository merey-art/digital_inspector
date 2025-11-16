#!/bin/bash
# Скрипт для запуска Frontend и Backend сервисов

set -e

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Переход в корневую директорию проекта
cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

# Порты
BACKEND_PORT=8000
FRONTEND_PORT=8080

# PID файлы для отслеживания процессов
BACKEND_PID_FILE="/tmp/digital_inspector_backend.pid"
FRONTEND_PID_FILE="/tmp/digital_inspector_frontend.pid"

# Функция для очистки при выходе
cleanup() {
    echo -e "\n${YELLOW}Остановка сервисов...${NC}"
    
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo -e "${YELLOW}Остановка Backend (PID: $BACKEND_PID)...${NC}"
            kill "$BACKEND_PID" 2>/dev/null || true
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            echo -e "${YELLOW}Остановка Frontend (PID: $FRONTEND_PID)...${NC}"
            kill "$FRONTEND_PID" 2>/dev/null || true
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    echo -e "${GREEN}Сервисы остановлены.${NC}"
    exit 0
}

# Установка обработчика сигналов для корректного завершения
trap cleanup SIGINT SIGTERM

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo -e "${RED}Ошибка: Виртуальное окружение не найдено!${NC}"
    echo -e "${YELLOW}Создайте виртуальное окружение: python -m venv venv${NC}"
    exit 1
fi

# Активация виртуального окружения
echo -e "${CYAN}Активация виртуального окружения...${NC}"
source venv/bin/activate

# Проверка модели
MODEL_PATH="$PROJECT_ROOT/training_results/run_default_15ep2/weights/best.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Предупреждение: Модель не найдена по пути: $MODEL_PATH${NC}"
    echo -e "${YELLOW}Используйте переменную окружения MODEL_PATH для указания другого пути${NC}"
else
    echo -e "${GREEN}Модель найдена: $MODEL_PATH${NC}"
    export MODEL_PATH="$MODEL_PATH"
fi

# Проверка занятости портов
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 1  # Порт занят
    else
        return 0  # Порт свободен
    fi
}

if ! check_port $BACKEND_PORT; then
    echo -e "${RED}Ошибка: Порт $BACKEND_PORT уже занят!${NC}"
    echo -e "${YELLOW}Остановите другой процесс на этом порту или измените BACKEND_PORT в скрипте${NC}"
    exit 1
fi

if ! check_port $FRONTEND_PORT; then
    echo -e "${RED}Ошибка: Порт $FRONTEND_PORT уже занят!${NC}"
    echo -e "${YELLOW}Остановите другой процесс на этом порту или измените FRONTEND_PORT в скрипте${NC}"
    exit 1
fi

# Запуск Backend
echo -e "\n${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Запуск Digital Inspector Services                 ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${GREEN}[1/2] Запуск Backend API на порту $BACKEND_PORT...${NC}"
cd "$PROJECT_ROOT/backend"

# Настройки для оптимизации производительности
UVICORN_WORKERS=${UVICORN_WORKERS:-4}
MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS:-10}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-20}

export UVICORN_WORKERS
export MAX_CONCURRENT_REQUESTS
export MAX_BATCH_SIZE

# Запуск с uvicorn напрямую для поддержки workers
uvicorn main:app --host 0.0.0.0 --port $BACKEND_PORT --workers $UVICORN_WORKERS > /tmp/digital_inspector_backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$BACKEND_PID_FILE"

# Небольшая задержка для запуска backend
sleep 2

# Проверка, что backend запустился
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo -e "${RED}Ошибка: Backend не запустился!${NC}"
    echo -e "${YELLOW}Проверьте логи: cat /tmp/digital_inspector_backend.log${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}✓ Backend запущен (PID: $BACKEND_PID)${NC}"

# Запуск Frontend
echo -e "${GREEN}[2/2] Запуск Frontend на порту $FRONTEND_PORT...${NC}"
cd "$PROJECT_ROOT/frontend"
python -m http.server $FRONTEND_PORT > /tmp/digital_inspector_frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$FRONTEND_PID_FILE"

# Небольшая задержка для запуска frontend
sleep 1

# Проверка, что frontend запустился
if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo -e "${RED}Ошибка: Frontend не запустился!${NC}"
    echo -e "${YELLOW}Проверьте логи: cat /tmp/digital_inspector_frontend.log${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}✓ Frontend запущен (PID: $FRONTEND_PID)${NC}"

# Информация о сервисах
echo -e "\n${CYAN}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Сервисы успешно запущены!                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${GREEN}Backend API:${NC}"
echo -e "  URL: ${BLUE}http://localhost:$BACKEND_PORT${NC}"
echo -e "  Health: ${BLUE}http://localhost:$BACKEND_PORT/health${NC}"
echo -e "  Docs: ${BLUE}http://localhost:$BACKEND_PORT/docs${NC}"
echo -e "  Логи: ${YELLOW}/tmp/digital_inspector_backend.log${NC}\n"

echo -e "${GREEN}Frontend:${NC}"
echo -e "  URL: ${BLUE}http://localhost:$FRONTEND_PORT${NC}"
echo -e "  Логи: ${YELLOW}/tmp/digital_inspector_frontend.log${NC}\n"

echo -e "${YELLOW}Для остановки сервисов нажмите Ctrl+C${NC}\n"

# Ожидание завершения
wait

