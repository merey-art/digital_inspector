#!/usr/bin/env python3
"""
Advanced YOLOv8 training script with CUDA support, automatic batch sizing,
EMA, early stopping, and comprehensive logging.
"""

import argparse
import logging
import os
import sys
import shutil
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import json

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ultralytics_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Import utilities
sys.path.insert(0, str(Path(__file__).parent))
from utils.autosize import auto_batch_size
from utils.metrics import save_metrics_json, save_metrics_csv, plot_training_curves


def check_environment():
    """Check CUDA availability and GPU info."""
    logger.info("=" * 60)
    logger.info("Environment Check")
    logger.info("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        device_count = torch.cuda.device_count()
        logger.info(f"Number of GPUs: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        return 'cuda:0'
    else:
        logger.warning("CUDA not available, using CPU (training will be slow)")
        return 'cpu'


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    default_config = {
        'epochs': 150,
        'imgsz': 640,
        'batch': 16,
        'lr0': 0.001,
        'lrf': 0.01,
        'optimizer': 'AdamW',
        'weight_decay': 0.0005,
        'momentum': 0.937,
        'warmup_epochs': 3.0,
        'lr_scheduler': 'cosine',
        'patience': 20,
        'ema': False,
        'ema_decay': 0.9999,
        'mosaic': 1.0,
        'mixup': 0.0,
        'focal': False,
        'amp': True,
    }
    
    if config_path:
        config_path_obj = Path(config_path)
        if config_path_obj.exists():
            with open(config_path_obj, 'r') as f:
                config = yaml.safe_load(f)
            default_config.update(config)
            logger.info(f"Loaded config from {config_path}")
    
    return default_config


def determine_batch_size(args, config, device, model_size='s'):
    """Determine optimal batch size."""
    target_batch = args.batch or config.get('batch', 16)
    
    if device == 'cpu':
        batch_size = min(target_batch, 4)
        logger.info(f"CPU mode: using batch size {batch_size}")
        return batch_size
    
    # Auto-adjust batch size based on VRAM
    imgsz = args.imgsz or config.get('imgsz', 640)
    auto_batch = auto_batch_size(imgsz, model_size, device, target_batch)
    
    return auto_batch


def setup_wandb(args, config):
    """Setup Weights & Biases logging."""
    if not args.wandb:
        return None
    
    try:
        import wandb
        wandb.init(
            project=args.project or 'yolov8-training',
            name=args.name or 'run',
            config={
                **config,
                'data': args.data,
                'backbone': args.backbone,
            },
            resume='allow'
        )
        logger.info("WandB initialized")
        return wandb
    except ImportError:
        logger.warning("wandb not installed, skipping WandB logging")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        return None


# Note: Early stopping is handled by Ultralytics built-in functionality
# This class is kept for reference but not actively used
class EarlyStopping:
    """Early stopping callback (deprecated - using Ultralytics built-in)."""
    
    def __init__(self, patience=20, min_delta=0.0, metric='metrics/mAP50(B)'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, metrics: Dict) -> bool:
        """Check if training should stop."""
        if self.metric not in metrics:
            return False
        
        current_score = metrics[self.metric]
        
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        else:
            self.best_score = current_score
            self.counter = 0
        
        return self.early_stop


def train(args):
    """Main training function."""
    # Check environment
    device = check_environment()
    
    # Load configs
    config = load_config(args.config)
    
    # Determine batch size
    model_size = args.backbone.replace('yolov8', '').replace('.pt', '') if args.backbone else 's'
    batch_size = determine_batch_size(args, config, device, model_size)
    
    # Setup output directory
    project_dir = Path(args.project or 'training_results')
    run_name = args.name or 'run'
    run_dir = project_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    weights_dir = run_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {run_dir}")
    
    # Setup WandB
    wandb_run = setup_wandb(args, config)
    
    # Load model
    backbone = args.backbone or 'yolov8s.pt'
    logger.info(f"Loading model: {backbone}")
    
    if args.resume and Path(args.resume).exists():
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(backbone)
    
    # Prepare training arguments
    train_args = {
        'data': args.data,
        'epochs': args.epochs or config.get('epochs', 150),
        'imgsz': args.imgsz or config.get('imgsz', 640),
        'batch': batch_size,
        'lr0': args.lr or config.get('lr0', 0.001),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3.0),
        'optimizer': args.optimizer or config.get('optimizer', 'AdamW'),
        'project': str(project_dir),
        'name': run_name,
        'device': device,
        'amp': config.get('amp', True),
        'mosaic': args.mosaic if args.mosaic is not None else config.get('mosaic', 1.0),
        'mixup': args.mixup if args.mixup is not None else config.get('mixup', 0.0),
        'copy_paste': config.get('copy_paste', 0.0),  # Copy-paste augmentation для minority классов
        'save': True,
        'save_period': config.get('save_period', 10),
        'val': True,
        'plots': True,
        'verbose': True,
    }
    
    # Увеличение веса classification loss для несбалансированных классов
    # Датасет: stamp 90.7%, signature 5.1%, QR-Code 4.2%
    cls_weight = config.get('cls', 0.5)  # По умолчанию 0.5, увеличиваем до 1.5 для minority классов
    if cls_weight != 0.5:
        train_args['cls'] = cls_weight
        logger.info(f"Classification loss weight increased to {cls_weight} for better minority class learning")
    
    # Learning rate scheduler
    if args.lr_scheduler or config.get('lr_scheduler'):
        lr_scheduler = args.lr_scheduler or config.get('lr_scheduler')
        if lr_scheduler == 'cosine':
            train_args['cos_lr'] = True
        elif lr_scheduler == 'onecycle':
            train_args['onecycle'] = True
    
    # EMA (Ultralytics has built-in EMA support)
    if args.ema or config.get('ema', False):
        logger.info("Using EMA (built-in Ultralytics support)")
        train_args['ema'] = True
        train_args['ema_decay'] = config.get('ema_decay', 0.9999)
    
    # Overlapping classes support (per-class NMS)
    # By default, agnostic_nms=False allows overlapping classes (e.g., signature under stamp)
    train_args['agnostic_nms'] = config.get('agnostic_nms', False)
    train_args['max_det'] = config.get('max_det', 300)
    if not train_args['agnostic_nms']:
        logger.info("Per-class NMS enabled: overlapping classes will be detected during training/validation")
    
    # Early stopping (Ultralytics has built-in early stopping)
    if config.get('early_stopping', True):
        patience = args.patience or config.get('patience', 20)
        train_args['patience'] = patience
        logger.info(f"Early stopping enabled with patience={patience}")
    
    # Focal loss (Note: may require custom implementation in ultralytics)
    if args.focal or config.get('focal', False):
        logger.warning("Focal loss may require custom implementation in ultralytics")
        # Ultralytics doesn't have direct focal loss flag, would need custom loss
    
    logger.info("Starting training...")
    logger.info(f"Training arguments: {train_args}")
    
    try:
        # Train model
        results = model.train(**train_args)
        
        # Save final metrics
        final_metrics = {}
        if hasattr(results, 'results_dict'):
            final_metrics = results.results_dict
        elif hasattr(results, 'results'):
            # Try alternative attribute
            final_metrics = results.results if isinstance(results.results, dict) else {}
        
        # Also try to get metrics from results object directly
        if not final_metrics:
            try:
                final_metrics = {
                    'metrics/mAP50(B)': getattr(results, 'metrics', {}).get('map50', 0.0),
                    'metrics/mAP50-95(B)': getattr(results, 'metrics', {}).get('map', 0.0),
                }
            except:
                pass
        
        # Save metrics to JSON
        metrics_path = run_dir / 'metrics.json'
        save_metrics_json(final_metrics, metrics_path)
        
        # Save metrics CSV if available (ultralytics saves this automatically)
        results_csv = run_dir / 'results.csv'
        if results_csv.exists():
            logger.info(f"Results CSV saved to {results_csv}")
        
        logger.info("Training completed successfully")
        
        # Export model if requested
        if args.export:
            export_formats = args.export.split(',')
            logger.info(f"Exporting model to: {export_formats}")
            
            best_weights = run_dir / 'weights' / 'best.pt'
            if best_weights.exists():
                export_model = YOLO(str(best_weights))
                for fmt in export_formats:
                    try:
                        if fmt == 'onnx':
                            export_model.export(format='onnx', imgsz=train_args['imgsz'])
                        elif fmt == 'torchscript':
                            export_model.export(format='torchscript', imgsz=train_args['imgsz'])
                        elif fmt == 'trt' or fmt == 'tensorrt':
                            try:
                                export_model.export(format='engine', imgsz=train_args['imgsz'])
                            except Exception as e:
                                logger.warning(f"TensorRT export failed: {e}")
                        logger.info(f"Exported to {fmt}")
                    except Exception as e:
                        logger.error(f"Failed to export to {fmt}: {e}")
        
        # TTA evaluation if requested
        if args.tta:
            logger.info("Running TTA evaluation...")
            # TTA will be handled in eval.py script
            logger.info("Use eval.py with --tta flag for TTA evaluation")
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            logger.error("CUDA out of memory! Try reducing batch size or image size")
            logger.error(f"Current batch size: {batch_size}")
            raise
        else:
            raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if wandb_run:
            try:
                wandb_run.finish()
            except:
                pass
    
    logger.info(f"Training results saved to: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model')
    
    # Data and model
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    parser.add_argument('--weights', type=str, help='Initial weights (deprecated, use --backbone)')
    parser.add_argument('--backbone', type=str, default='yolov8s.pt', 
                       help='Pretrained model (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, help='Image size')
    parser.add_argument('--batch', type=int, help='Batch size (auto-adjusted if not specified)')
    parser.add_argument('--lr', type=float, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'],
                       help='Optimizer')
    parser.add_argument('--lr-scheduler', type=str, choices=['linear', 'cosine', 'onecycle'],
                       help='Learning rate scheduler')
    
    # Features
    parser.add_argument('--ema', action='store_true', help='Use EMA')
    parser.add_argument('--focal', action='store_true', help='Use Focal Loss')
    parser.add_argument('--mosaic', type=float, help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, help='Mixup augmentation probability')
    parser.add_argument('--tta', action='store_true', help='Run TTA evaluation after training')
    
    # Early stopping
    parser.add_argument('--patience', type=int, help='Early stopping patience')
    
    # Output
    parser.add_argument('--project', type=str, default='training_results', help='Project directory')
    parser.add_argument('--name', type=str, help='Run name')
    parser.add_argument('--config', type=str, help='Path to training config YAML')
    
    # Logging
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases')
    
    # Export
    parser.add_argument('--export', type=str, help='Export formats (comma-separated: onnx,torchscript,trt)')
    
    args = parser.parse_args()
    
    # Handle deprecated --weights
    if args.weights and not args.backbone:
        args.backbone = args.weights
        logger.warning("--weights is deprecated, use --backbone instead")
    
    train(args)


if __name__ == '__main__':
    main()

