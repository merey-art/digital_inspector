#!/bin/bash
# Скрипт для проверки установки torch с CUDA

echo "Проверка установки PyTorch..."
python -c "
import torch
print(f'PyTorch версия: {torch.__version__}')
print(f'CUDA доступна: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA версия: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  ВНИМАНИЕ: CUDA недоступна!')
    print('Если вы установили torch через pip install -r requirements.txt,')
    print('возможно была установлена CPU версия вместо CUDA версии.')
    print('')
    print('Для исправления установите torch с CUDA:')
    print('pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
"
