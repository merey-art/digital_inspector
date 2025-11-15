"""
Metrics calculation and visualization utilities.
Includes confusion matrix, per-class metrics, and plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: Optional[Path] = None, normalize: bool = True):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (n_classes, n_classes)
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the matrix
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized' if normalize else 'Count'})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(metrics: Dict[str, List[float]], save_path: Optional[Path] = None):
    """
    Plot training curves (loss, mAP, etc.).
    
    Args:
        metrics: Dictionary with metric names as keys and lists of values as values
        save_path: Path to save the plot
    """
    n_metrics = len(metrics)
    if n_metrics == 0:
        return
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        axes[idx].plot(values, linewidth=2)
        axes[idx].set_title(metric_name)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric_name)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_per_class_metrics(results: Dict, class_names: List[str]) -> Dict:
    """
    Calculate per-class metrics from YOLO results.
    
    Args:
        results: Results dictionary from YOLO validation
        class_names: List of class names
        
    Returns:
        Dictionary with per-class metrics
    """
    per_class_metrics = {}
    
    # Extract metrics from results
    if hasattr(results, 'results_dict'):
        metrics_dict = results.results_dict
    elif isinstance(results, dict):
        metrics_dict = results
    else:
        logger.warning("Unknown results format")
        return {}
    
    # Get per-class AP if available
    if 'metrics' in metrics_dict:
        metrics = metrics_dict['metrics']
        for i, class_name in enumerate(class_names):
            class_metrics = {
                'precision': metrics.get(f'precision/{i}', 0.0),
                'recall': metrics.get(f'recall/{i}', 0.0),
                'mAP50': metrics.get(f'mAP50/{i}', 0.0),
                'mAP50-95': metrics.get(f'mAP50-95/{i}', 0.0),
            }
            per_class_metrics[class_name] = class_metrics
    
    return per_class_metrics


def save_metrics_json(metrics: Dict, save_path: Path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        save_path: Path to save JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_metrics = convert_to_serializable(metrics)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {save_path}")


def save_metrics_csv(metrics: Dict, save_path: Path):
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Metrics dictionary (should be epoch-wise)
        save_path: Path to save CSV file
    """
    import pandas as pd
    
    # Flatten metrics if needed
    if isinstance(metrics, dict) and all(isinstance(v, list) for v in metrics.values()):
        df = pd.DataFrame(metrics)
    else:
        # Try to convert to DataFrame
        df = pd.DataFrame([metrics])
    
    df.to_csv(save_path, index=False)
    logger.info(f"Metrics CSV saved to {save_path}")


def plot_per_class_metrics(per_class_metrics: Dict, save_path: Optional[Path] = None):
    """
    Визуализация метрик по классам (precision, recall, mAP).
    
    Args:
        per_class_metrics: Словарь с метриками по классам
        save_path: Путь для сохранения графика
    """
    if not per_class_metrics:
        logger.warning("No per-class metrics to plot")
        return
    
    classes = list(per_class_metrics.keys())
    metrics_names = ['precision', 'recall', 'mAP50', 'mAP50-95']
    
    # Подготовка данных
    data = {metric: [per_class_metrics[cls].get(metric, 0.0) for cls in classes] 
            for metric in metrics_names}
    
    # Создание графика
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    x = np.arange(len(classes))
    width = 0.6
    
    for idx, metric in enumerate(metrics_names):
        values = data[metric]
        bars = axes[idx].bar(x, values, width, alpha=0.7, 
                            color=sns.color_palette("husl", len(classes)))
        axes[idx].set_title(f'{metric.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Класс', fontsize=10)
        axes[idx].set_ylabel('Значение', fontsize=10)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(classes, rotation=45, ha='right')
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Per-class metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_detection_statistics(detections: List[Dict], class_names: Dict[int, str],
                              save_path: Optional[Path] = None):
    """
    Визуализация статистики детекций: распределение по классам, confidence и т.д.
    
    Args:
        detections: Список детекций с полями class_id, confidence, bbox
        class_names: Словарь соответствия class_id -> class_name
        save_path: Путь для сохранения графика
    """
    if not detections:
        logger.warning("No detections to plot")
        return
    
    # Подготовка данных
    class_counts = {}
    confidences = {name: [] for name in class_names.values()}
    bbox_areas = {name: [] for name in class_names.values()}
    
    for det in detections:
        class_id = det.get('class_id', det.get('class'))
        class_name = class_names.get(class_id, f'class_{class_id}')
        confidence = det.get('confidence', 0.0)
        bbox = det.get('bbox', {})
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidences[class_name].append(confidence)
        
        if 'width' in bbox and 'height' in bbox:
            area = bbox['width'] * bbox['height']
            bbox_areas[class_name].append(area)
    
    # Создание графиков
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Распределение по классам
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = sns.color_palette("husl", len(classes))
    
    axes[0, 0].bar(classes, counts, color=colors, alpha=0.7)
    axes[0, 0].set_title('Количество детекций по классам', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Класс', fontsize=10)
    axes[0, 0].set_ylabel('Количество', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for i, (cls, count) in enumerate(zip(classes, counts)):
        axes[0, 0].text(i, count, str(count), ha='center', va='bottom', fontsize=9)
    
    # 2. Распределение confidence по классам
    for class_name in classes:
        if confidences[class_name]:
            axes[0, 1].hist(confidences[class_name], bins=20, alpha=0.6, 
                           label=class_name, density=True)
    axes[0, 1].set_title('Распределение confidence по классам', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Confidence', fontsize=10)
    axes[0, 1].set_ylabel('Плотность', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot confidence по классам
    conf_data = [confidences[cls] for cls in classes if confidences[cls]]
    conf_labels = [cls for cls in classes if confidences[cls]]
    if conf_data:
        bp = axes[1, 0].boxplot(conf_data, labels=conf_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1, 0].set_title('Box plot: Confidence по классам', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Класс', fontsize=10)
        axes[1, 0].set_ylabel('Confidence', fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Распределение площадей bbox по классам
    for class_name in classes:
        if bbox_areas[class_name]:
            axes[1, 1].hist(bbox_areas[class_name], bins=20, alpha=0.6, 
                           label=class_name, density=True)
    axes[1, 1].set_title('Распределение площадей bbox по классам', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Площадь (px²)', fontsize=10)
    axes[1, 1].set_ylabel('Плотность', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Detection statistics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

