#!/usr/bin/env python3
"""
Evaluation script for YOLOv8 models.
Computes mAP, precision, recall, confusion matrix, and per-class metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import yaml

import torch
import numpy as np
from ultralytics import YOLO

# Import utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.metrics import (
    plot_confusion_matrix, 
    calculate_per_class_metrics,
    save_metrics_json,
    save_metrics_csv
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_class_names(data_yaml: str) -> list:
    """Load class names from dataset YAML."""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return list(data.get('names', {}).values())


def evaluate_model(weights: str, data: str, imgsz: int = 640, 
                  conf: float = 0.25, iou: float = 0.45,
                  device: str = 'cuda:0', save_dir: Path = None,
                  tta: bool = False, agnostic_nms: bool = False, max_det: int = 300):
    """
    Evaluate YOLOv8 model on validation set.
    
    Args:
        weights: Path to model weights
        data: Path to dataset.yaml
        imgsz: Image size
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: Device to use
        save_dir: Directory to save results
        tta: Use Test Time Augmentation
        agnostic_nms: If False, NMS is applied per-class (allows overlapping classes)
        max_det: Maximum number of detections per image
    """
    logger.info(f"Loading model from: {weights}")
    model = YOLO(weights)
    
    # Check device
    if device == 'cuda:0' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"Evaluating on device: {device}")
    if not agnostic_nms:
        logger.info("Per-class NMS enabled: overlapping classes will be evaluated")
    else:
        logger.info("Agnostic NMS enabled: overlapping detections will be suppressed")
    
    # Run validation
    logger.info("Running validation...")
    results = model.val(
        data=data,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        plots=True,
        save_json=True,
        save_hybrid=False,
        half=False,  # Use FP32 for accurate metrics
        verbose=True
    )
    
    # Extract metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
    else:
        metrics = {}
    
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    
    # Print key metrics
    key_metrics = [
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)',
        'metrics/precision(B)',
        'metrics/recall(B)',
    ]
    
    for key in key_metrics:
        if key in metrics:
            logger.info(f"{key}: {metrics[key]:.4f}")
    
    # Load class names
    try:
        class_names = load_class_names(data)
        logger.info(f"Classes: {class_names}")
    except Exception as e:
        logger.warning(f"Could not load class names: {e}")
        class_names = []
    
    # Calculate per-class metrics
    if class_names:
        per_class = calculate_per_class_metrics(results, class_names)
        if per_class:
            logger.info("\nPer-class metrics:")
            for class_name, class_metrics in per_class.items():
                logger.info(f"  {class_name}:")
                for metric_name, value in class_metrics.items():
                    logger.info(f"    {metric_name}: {value:.4f}")
    
    # Save results
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics JSON
        metrics_path = save_dir / 'eval_metrics.json'
        save_metrics_json(metrics, metrics_path)
        
        # Save per-class metrics
        if per_class:
            per_class_path = save_dir / 'per_class_metrics.json'
            save_metrics_json(per_class, per_class_path)
        
        # Plot confusion matrix if available
        try:
            if hasattr(results, 'confusion_matrix'):
                cm = results.confusion_matrix.matrix
                cm_path = save_dir / 'confusion_matrix.png'
                plot_confusion_matrix(cm, class_names, cm_path)
        except Exception as e:
            logger.warning(f"Could not plot confusion matrix: {e}")
        
        logger.info(f"\nResults saved to: {save_dir}")
    
    # TTA evaluation if requested
    if tta:
        logger.info("\nRunning TTA evaluation...")
        # Note: TTA is complex and may require custom implementation
        # For now, we'll use ultralytics built-in if available
        try:
            tta_results = model.val(
                data=data,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                agnostic_nms=agnostic_nms,
                max_det=max_det,
                plots=True,
                save_json=True,
                augment=True,  # Enable augmentation during validation
            )
            logger.info("TTA evaluation completed")
            if save_dir:
                tta_metrics_path = save_dir / 'tta_metrics.json'
                if hasattr(tta_results, 'results_dict'):
                    save_metrics_json(tta_results.results_dict, tta_metrics_path)
        except Exception as e:
            logger.warning(f"TTA evaluation failed: {e}")
    
    return metrics, per_class


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv8 model')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--save-dir', type=str, help='Directory to save results')
    parser.add_argument('--tta', action='store_true', help='Use Test Time Augmentation')
    parser.add_argument('--agnostic-nms', action='store_true',
                       help='Use agnostic NMS (suppresses overlapping detections across classes). '
                            'By default, per-class NMS is used (allows overlapping classes)')
    parser.add_argument('--max-det', type=int, default=300,
                       help='Maximum number of detections per image (default: 300)')
    
    args = parser.parse_args()
    
    # Setup save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        # Default: save in weights directory
        weights_path = Path(args.weights)
        save_dir = weights_path.parent.parent / 'eval_results'
    
    evaluate_model(
        weights=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=save_dir,
        tta=args.tta,
        agnostic_nms=args.agnostic_nms,
        max_det=args.max_det
    )


if __name__ == '__main__':
    main()

