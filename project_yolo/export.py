#!/usr/bin/env python3
"""
Model export script for YOLOv8.
Exports models to ONNX, TorchScript, TensorRT, and other formats.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_tensorrt():
    """Check if TensorRT is available."""
    try:
        import tensorrt
        logger.info(f"TensorRT version: {tensorrt.__version__}")
        return True
    except ImportError:
        logger.warning("TensorRT not available")
        return False


def export_model(weights: str, formats: list, imgsz: int = 640,
                device: str = 'cuda:0', half: bool = False,
                simplify: bool = True, workspace: int = 4):
    """
    Export YOLOv8 model to various formats.
    
    Args:
        weights: Path to model weights
        formats: List of export formats
        imgsz: Image size
        device: Device to use
        half: Use FP16 precision
        simplify: Simplify ONNX model
        workspace: TensorRT workspace size in GB
    """
    logger.info(f"Loading model from: {weights}")
    model = YOLO(weights)
    
    # Check device
    if device == 'cuda:0' and not torch.cuda.is_available():
        logger.warning("CUDA not available, some exports may fail")
        device = 'cpu'
    
    logger.info(f"Exporting to formats: {formats}")
    
    # Export to each format
    for fmt in formats:
        fmt = fmt.lower().strip()
        logger.info(f"\nExporting to {fmt.upper()}...")
        
        try:
            if fmt == 'onnx':
                model.export(
                    format='onnx',
                    imgsz=imgsz,
                    half=half,
                    simplify=simplify,
                    device=device
                )
                logger.info(f"✓ ONNX export successful")
            
            elif fmt == 'torchscript' or fmt == 'torch':
                model.export(
                    format='torchscript',
                    imgsz=imgsz,
                    device=device
                )
                logger.info(f"✓ TorchScript export successful")
            
            elif fmt == 'tensorrt' or fmt == 'trt' or fmt == 'engine':
                if not check_tensorrt():
                    logger.warning("Skipping TensorRT export (not available)")
                    continue
                
                if device == 'cpu':
                    logger.warning("TensorRT requires CUDA, skipping")
                    continue
                
                try:
                    model.export(
                        format='engine',
                        imgsz=imgsz,
                        half=half,
                        device=device,
                        workspace=workspace
                    )
                    logger.info(f"✓ TensorRT export successful")
                except Exception as e:
                    logger.error(f"TensorRT export failed: {e}")
                    logger.info("Make sure TensorRT is properly installed and CUDA is available")
            
            elif fmt == 'coreml':
                model.export(
                    format='coreml',
                    imgsz=imgsz,
                    half=half
                )
                logger.info(f"✓ CoreML export successful")
            
            elif fmt == 'tflite':
                model.export(
                    format='tflite',
                    imgsz=imgsz
                )
                logger.info(f"✓ TFLite export successful")
            
            elif fmt == 'pb' or fmt == 'tensorflow':
                model.export(
                    format='pb',
                    imgsz=imgsz
                )
                logger.info(f"✓ TensorFlow export successful")
            
            elif fmt == 'openvino':
                model.export(
                    format='openvino',
                    imgsz=imgsz,
                    half=half
                )
                logger.info(f"✓ OpenVINO export successful")
            
            else:
                logger.warning(f"Unknown format: {fmt}, skipping")
        
        except Exception as e:
            logger.error(f"Failed to export to {fmt}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    logger.info("\nExport completed!")
    
    # List exported files
    weights_path = Path(weights)
    export_dir = weights_path.parent
    logger.info(f"\nExported files in: {export_dir}")
    
    exported_files = list(export_dir.glob(f"{weights_path.stem}.*"))
    for f in exported_files:
        if f.suffix != '.pt':
            logger.info(f"  - {f.name}")


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8 model to various formats')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--formats', type=str, required=True,
                       help='Comma-separated list of formats (onnx,torchscript,trt,coreml,tflite,pb,openvino)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device (cuda:0 or cpu)')
    parser.add_argument('--half', action='store_true', help='Use FP16 precision')
    parser.add_argument('--no-simplify', action='store_true', help='Do not simplify ONNX model')
    parser.add_argument('--workspace', type=int, default=4, help='TensorRT workspace size in GB')
    
    args = parser.parse_args()
    
    formats = [f.strip() for f in args.formats.split(',')]
    
    export_model(
        weights=args.weights,
        formats=formats,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        simplify=not args.no_simplify,
        workspace=args.workspace
    )


if __name__ == '__main__':
    main()


