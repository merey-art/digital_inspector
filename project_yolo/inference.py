#!/usr/bin/env python3
"""
Batch inference script for YOLOv8 models.
Processes images from a directory and saves visualizations and JSON results.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_image(model, image_path: Path, conf: float, iou: float, 
                 agnostic_nms: bool = False, max_det: int = 300) -> Dict[str, Any]:
    """
    Process a single image and return results.
    
    Args:
        model: YOLO model
        image_path: Path to image
        conf: Confidence threshold
        iou: IoU threshold for NMS
        agnostic_nms: If False, NMS is applied per-class (allows overlapping classes)
        max_det: Maximum number of detections per image
        
    Returns:
        Dictionary with detection results
    """
    try:
        # Run inference with per-class NMS to allow overlapping classes
        # agnostic_nms=False means NMS is applied separately for each class,
        # allowing objects of different classes to overlap (e.g., signature under stamp)
        results = model(str(image_path), conf=conf, iou=iou, 
                       agnostic_nms=agnostic_nms, max_det=max_det, verbose=False)
        
        # Extract detections
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                conf_score = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                cls_name = model.names[cls]
                
                detections.append({
                    'class': cls_name,
                    'class_id': cls,
                    'confidence': conf_score,
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3]),
                        'width': float(box[2] - box[0]),
                        'height': float(box[3] - box[1])
                    }
                })
        
        return {
            'image': str(image_path.name),
            'path': str(image_path),
            'detections': detections,
            'num_detections': len(detections),
            'success': True
        }
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return {
            'image': str(image_path.name),
            'path': str(image_path),
            'detections': [],
            'num_detections': 0,
            'success': False,
            'error': str(e)
        }


def batch_inference(weights: str, source: str, conf: float = 0.25, 
                   iou: float = 0.45, device: str = 'cuda:0',
                   save_dir: Path = None, save_images: bool = True,
                   agnostic_nms: bool = False, max_det: int = 300):
    """
    Run batch inference on images in a directory.
    
    Args:
        weights: Path to model weights
        source: Path to image directory or single image
        conf: Confidence threshold
        iou: IoU threshold for NMS
        device: Device to use
        save_dir: Directory to save results
        save_images: Whether to save annotated images
        agnostic_nms: If False, NMS is applied per-class (allows overlapping classes)
                     Set to True to apply NMS across all classes (suppresses overlapping)
        max_det: Maximum number of detections per image
    """
    logger.info(f"Loading model from: {weights}")
    model = YOLO(weights)
    
    # Check device
    if device == 'cuda:0' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    
    model.to(device)
    logger.info(f"Using device: {device}")
    
    # Find images
    source_path = Path(source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [p for p in source_path.rglob('*') 
                      if p.suffix.lower() in image_extensions]
        logger.info(f"Found {len(image_paths)} images")
    else:
        raise ValueError(f"Source path does not exist: {source}")
    
    if len(image_paths) == 0:
        logger.error("No images found")
        return
    
    # Setup output directory
    if save_dir is None:
        save_dir = Path('runs/inference')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = save_dir / 'images'
    if save_images:
        images_dir.mkdir(exist_ok=True)
    
    # Process images
    all_results = []
    total_time = 0.0
    successful = 0
    failed = 0
    
    logger.info("Running inference...")
    if not agnostic_nms:
        logger.info("Per-class NMS enabled: overlapping classes will be detected (e.g., signature under stamp)")
    else:
        logger.info("Agnostic NMS enabled: overlapping detections will be suppressed")
    logger.info(f"Max detections per image: {max_det}")
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        start_time = time.time()
        
        # Process image
        result = process_image(model, image_path, conf, iou, agnostic_nms, max_det)
        
        # Save annotated image if requested
        if save_images and result['success']:
            try:
                # Run inference with visualization (with overlapping classes support)
                vis_results = model(str(image_path), conf=conf, iou=iou, 
                                  agnostic_nms=agnostic_nms, max_det=max_det,
                                  save=True, project=str(save_dir), name='images', exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to save visualization for {image_path}: {e}")
        
        elapsed = time.time() - start_time
        total_time += elapsed
        result['inference_time'] = elapsed
        result['fps'] = 1.0 / elapsed if elapsed > 0 else 0.0
        
        if result['success']:
            successful += 1
        else:
            failed += 1
        
        all_results.append(result)
    
    # Calculate statistics
    avg_time = total_time / len(image_paths) if len(image_paths) > 0 else 0.0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0.0
    total_detections = sum(r['num_detections'] for r in all_results)
    
    # Summary
    summary = {
        'total_images': len(image_paths),
        'successful': successful,
        'failed': failed,
        'total_detections': total_detections,
        'avg_inference_time': avg_time,
        'avg_fps': avg_fps,
        'total_time': total_time,
        'conf_threshold': conf,
        'iou_threshold': iou,
        'agnostic_nms': agnostic_nms,
        'max_det': max_det,
        'model': str(weights),
        'device': device
    }
    
    # Save results
    results_json = save_dir / 'results.json'
    with open(results_json, 'w') as f:
        json.dump({
            'summary': summary,
            'results': all_results
        }, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Inference Summary")
    logger.info("=" * 60)
    logger.info(f"Total images: {summary['total_images']}")
    logger.info(f"Successful: {summary['successful']}")
    logger.info(f"Failed: {summary['failed']}")
    logger.info(f"Total detections: {summary['total_detections']}")
    logger.info(f"Average inference time: {avg_time:.3f} s")
    logger.info(f"Average FPS: {avg_fps:.2f}")
    logger.info(f"Total time: {total_time:.2f} s")
    logger.info(f"\nResults saved to: {save_dir}")
    logger.info(f"JSON results: {results_json}")
    if save_images:
        logger.info(f"Annotated images: {images_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run batch inference with YOLOv8')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, 
                       help='Path to image directory or single image')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--save-dir', type=str, help='Directory to save results')
    parser.add_argument('--no-save-images', action='store_true', 
                       help='Do not save annotated images')
    parser.add_argument('--agnostic-nms', action='store_true',
                       help='Use agnostic NMS (suppresses overlapping detections across classes). '
                            'By default, per-class NMS is used (allows overlapping classes)')
    parser.add_argument('--max-det', type=int, default=300,
                       help='Maximum number of detections per image (default: 300)')
    
    args = parser.parse_args()
    
    batch_inference(
        weights=args.weights,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=Path(args.save_dir) if args.save_dir else None,
        save_images=not args.no_save_images,
        agnostic_nms=args.agnostic_nms,
        max_det=args.max_det
    )


if __name__ == '__main__':
    main()

