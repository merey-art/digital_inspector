#!/bin/bash
# Скрипт для запуска frontend сервера

cd "$(dirname "$0")/frontend"
python3 -m http.server 8080


