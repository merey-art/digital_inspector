#!/usr/bin/env python3
"""
Скрипт для визуализации результатов обучения YOLO модели
Создает графики метрик, потерь и других показателей из CSV файла

Использование:
    python visualize_training.py --csv path/to/results.csv
    python visualize_training.py --csv path/to/results.csv --all
    python visualize_training.py --csv path/to/results.csv --output output_dir

Примеры:
    # Базовое использование (создает комплексный график)
    python visualize_training.py --csv training_results/run_default_15ep2/results.csv
    
    # Создать все типы графиков
    python visualize_training.py --csv training_results/run_default_15ep2/results.csv --all
    
    # Сохранить в другую директорию
    python visualize_training.py --csv results.csv --output my_plots/
    
    # Без сглаживания
    python visualize_training.py --csv results.csv --no-smooth
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Настройка matplotlib для работы без GUI (если нужно)
matplotlib.use('Agg')  # Используем backend без GUI

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results_csv(csv_path: Path) -> pd.DataFrame:
    """Загрузка CSV файла с результатами обучения."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} эпох из {csv_path}")
    return df


def plot_losses(df: pd.DataFrame, save_dir: Path, smooth: bool = True):
    """Визуализация потерь (losses)."""
    loss_columns = [col for col in df.columns if 'loss' in col.lower()]
    
    if not loss_columns:
        print("Не найдены колонки с потерями")
        return
    
    n_losses = len(loss_columns)
    fig, axes = plt.subplots(1, n_losses, figsize=(6 * n_losses, 5))
    
    if n_losses == 1:
        axes = [axes]
    
    for idx, loss_col in enumerate(loss_columns):
        values = df[loss_col].values
        epochs = df['epoch'].values if 'epoch' in df.columns else range(len(values))
        
        axes[idx].plot(epochs, values, 'o-', linewidth=2, markersize=4, 
                      label=loss_col, alpha=0.7)
        
        if smooth and len(values) > 3:
            smoothed = gaussian_filter1d(values, sigma=1.0)
            axes[idx].plot(epochs, smoothed, '--', linewidth=2, 
                          label=f'{loss_col} (сглаженная)', alpha=0.8)
        
        axes[idx].set_title(f'{loss_col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Эпоха', fontsize=10)
        axes[idx].set_ylabel('Loss', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    save_path = save_dir / 'losses.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График потерь сохранен: {save_path}")
    plt.close()


def plot_metrics(df: pd.DataFrame, save_dir: Path, smooth: bool = True):
    """Визуализация метрик (precision, recall, mAP)."""
    metric_columns = [col for col in df.columns if 'metric' in col.lower() or 
                     'precision' in col.lower() or 'recall' in col.lower() or 
                     'map' in col.lower()]
    
    if not metric_columns:
        print("Не найдены колонки с метриками")
        return
    
    n_metrics = len(metric_columns)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric_col in enumerate(metric_columns):
        values = df[metric_col].values
        epochs = df['epoch'].values if 'epoch' in df.columns else range(len(values))
        
        axes[idx].plot(epochs, values, 'o-', linewidth=2, markersize=4, 
                      label=metric_col, alpha=0.7, color=sns.color_palette()[idx % len(sns.color_palette())])
        
        if smooth and len(values) > 3:
            smoothed = gaussian_filter1d(values, sigma=1.0)
            axes[idx].plot(epochs, smoothed, '--', linewidth=2, 
                          label=f'{metric_col} (сглаженная)', alpha=0.8)
        
        axes[idx].set_title(f'{metric_col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Эпоха', fontsize=10)
        axes[idx].set_ylabel('Значение', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)
        axes[idx].set_ylim(bottom=0)
    
    # Скрываем лишние subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / 'metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График метрик сохранен: {save_path}")
    plt.close()


def plot_learning_rate(df: pd.DataFrame, save_dir: Path):
    """Визуализация learning rate."""
    lr_columns = [col for col in df.columns if 'lr' in col.lower()]
    
    if not lr_columns:
        print("Не найдены колонки с learning rate")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    epochs = df['epoch'].values if 'epoch' in df.columns else range(len(df))
    
    for lr_col in lr_columns:
        values = df[lr_col].values
        ax.plot(epochs, values, 'o-', linewidth=2, markersize=4, 
               label=lr_col, alpha=0.7)
    
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.set_xlabel('Эпоха', fontsize=10)
    ax.set_ylabel('Learning Rate', fontsize=10)
    ax.set_yscale('log')  # Логарифмическая шкала для LR
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    save_path = save_dir / 'learning_rate.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График learning rate сохранен: {save_path}")
    plt.close()


def plot_comprehensive(df: pd.DataFrame, save_dir: Path, smooth: bool = True):
    """Комплексная визуализация всех метрик и потерь."""
    # Разделяем на train и val
    train_losses = [col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower()]
    val_losses = [col for col in df.columns if 'val' in col.lower() and 'loss' in col.lower()]
    metrics = [col for col in df.columns if 'metric' in col.lower() or 
              ('precision' in col.lower() or 'recall' in col.lower() or 'map' in col.lower())]
    
    epochs = df['epoch'].values if 'epoch' in df.columns else range(len(df))
    
    # Создаем фигуру с несколькими subplots
    n_plots = len(train_losses) + len(val_losses) + len(metrics)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    
    if rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Train losses
    for loss_col in train_losses:
        values = df[loss_col].values
        axes[plot_idx].plot(epochs, values, 'o-', linewidth=2, markersize=3, 
                           label=loss_col, alpha=0.7, color='blue')
        if smooth and len(values) > 3:
            smoothed = gaussian_filter1d(values, sigma=1.0)
            axes[plot_idx].plot(epochs, smoothed, '--', linewidth=2, 
                               label='сглаженная', alpha=0.8, color='darkblue')
        axes[plot_idx].set_title(loss_col, fontsize=11, fontweight='bold')
        axes[plot_idx].set_xlabel('Эпоха', fontsize=9)
        axes[plot_idx].set_ylabel('Loss', fontsize=9)
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend(fontsize=8)
        plot_idx += 1
    
    # Val losses
    for loss_col in val_losses:
        values = df[loss_col].values
        axes[plot_idx].plot(epochs, values, 'o-', linewidth=2, markersize=3, 
                           label=loss_col, alpha=0.7, color='red')
        if smooth and len(values) > 3:
            smoothed = gaussian_filter1d(values, sigma=1.0)
            axes[plot_idx].plot(epochs, smoothed, '--', linewidth=2, 
                               label='сглаженная', alpha=0.8, color='darkred')
        axes[plot_idx].set_title(loss_col, fontsize=11, fontweight='bold')
        axes[plot_idx].set_xlabel('Эпоха', fontsize=9)
        axes[plot_idx].set_ylabel('Loss', fontsize=9)
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend(fontsize=8)
        plot_idx += 1
    
    # Metrics
    for metric_col in metrics:
        values = df[metric_col].values
        axes[plot_idx].plot(epochs, values, 'o-', linewidth=2, markersize=3, 
                           label=metric_col, alpha=0.7, color='green')
        if smooth and len(values) > 3:
            smoothed = gaussian_filter1d(values, sigma=1.0)
            axes[plot_idx].plot(epochs, smoothed, '--', linewidth=2, 
                               label='сглаженная', alpha=0.8, color='darkgreen')
        axes[plot_idx].set_title(metric_col, fontsize=11, fontweight='bold')
        axes[plot_idx].set_xlabel('Эпоха', fontsize=9)
        axes[plot_idx].set_ylabel('Значение', fontsize=9)
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].legend(fontsize=8)
        axes[plot_idx].set_ylim(bottom=0)
        plot_idx += 1
    
    # Скрываем лишние subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / 'comprehensive.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Комплексный график сохранен: {save_path}")
    plt.close()


def plot_comparison(df: pd.DataFrame, save_dir: Path):
    """Сравнение train и val потерь на одном графике."""
    train_losses = [col for col in df.columns if 'train' in col.lower() and 'loss' in col.lower()]
    val_losses = [col for col in df.columns if 'val' in col.lower() and 'loss' in col.lower()]
    
    if not train_losses or not val_losses:
        print("Недостаточно данных для сравнения")
        return
    
    # Находим соответствующие пары train/val
    loss_pairs = []
    for train_col in train_losses:
        loss_name = train_col.replace('train/', '').replace('train_', '')
        for val_col in val_losses:
            if loss_name in val_col:
                loss_pairs.append((train_col, val_col, loss_name))
                break
    
    if not loss_pairs:
        return
    
    n_pairs = len(loss_pairs)
    cols = min(3, n_pairs)
    rows = (n_pairs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    
    if n_pairs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    epochs = df['epoch'].values if 'epoch' in df.columns else range(len(df))
    
    for idx, (train_col, val_col, loss_name) in enumerate(loss_pairs):
        train_values = df[train_col].values
        val_values = df[val_col].values
        
        axes[idx].plot(epochs, train_values, 'o-', linewidth=2, markersize=3, 
                      label='Train', alpha=0.7, color='blue')
        axes[idx].plot(epochs, val_values, 's-', linewidth=2, markersize=3, 
                      label='Validation', alpha=0.7, color='red')
        
        axes[idx].set_title(f'{loss_name} - Train vs Validation', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Эпоха', fontsize=10)
        axes[idx].set_ylabel('Loss', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    # Скрываем лишние subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = save_dir / 'train_val_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сравнения train/val сохранен: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Визуализация результатов обучения YOLO')
    parser.add_argument('--csv', type=str, 
                       default='training_results/run_default_15ep2/results.csv',
                       help='Путь к CSV файлу с результатами')
    parser.add_argument('--output', type=str, 
                       help='Директория для сохранения графиков (по умолчанию - та же, где CSV)')
    parser.add_argument('--no-smooth', action='store_true',
                       help='Отключить сглаживание графиков')
    parser.add_argument('--all', action='store_true',
                       help='Создать все типы графиков')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = csv_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка данных
    df = load_results_csv(csv_path)
    
    print(f"\nКолонки в CSV: {list(df.columns)}")
    print(f"Количество эпох: {len(df)}")
    
    smooth = not args.no_smooth
    
    # Создание графиков
    if args.all:
        print("\nСоздание всех графиков...")
        plot_losses(df, output_dir, smooth)
        plot_metrics(df, output_dir, smooth)
        plot_learning_rate(df, output_dir)
        plot_comparison(df, output_dir)
        plot_comprehensive(df, output_dir, smooth)
    else:
        # По умолчанию создаем комплексный график
        print("\nСоздание комплексного графика...")
        plot_comprehensive(df, output_dir, smooth)
        plot_comparison(df, output_dir)
    
    print(f"\nГрафики сохранены в: {output_dir}")


if __name__ == '__main__':
    main()

