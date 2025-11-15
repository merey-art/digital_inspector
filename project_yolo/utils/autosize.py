"""
Automatic batch size selection based on available VRAM.
Falls back to conservative estimates if pynvml is not available.
"""

import logging
import torch

logger = logging.getLogger(__name__)

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available, using conservative batch size estimates")


def get_gpu_memory_gb(device_id=0):
    """
    Get available GPU memory in GB.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        Available memory in GB, or None if unavailable
    """
    if not PYNVML_AVAILABLE:
        return None
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gb = info.free / (1024 ** 3)
        total_gb = info.total / (1024 ** 3)
        return free_gb, total_gb
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return None


def estimate_batch_size(imgsz=640, model_size='s', available_vram_gb=None, conservative=True):
    """
    Estimate appropriate batch size based on image size, model size, and available VRAM.
    
    Args:
        imgsz: Image size (default 640)
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        available_vram_gb: Available VRAM in GB (if None, will try to detect)
        conservative: Use conservative estimates if VRAM detection fails
        
    Returns:
        Recommended batch size
    """
    # Base batch sizes per model size for 640x640 images
    # These are rough estimates based on typical VRAM usage
    base_batches = {
        'n': 32,  # nano
        's': 16,  # small
        'm': 8,   # medium
        'l': 4,   # large
        'x': 2    # xlarge
    }
    
    # Scale factor based on image size (640 is baseline)
    size_factor = (640 / imgsz) ** 2
    
    if available_vram_gb is None:
        if torch.cuda.is_available():
            mem_info = get_gpu_memory_gb(0)
            if mem_info:
                available_vram_gb, total_vram_gb = mem_info
                logger.info(f"Detected GPU: {total_vram_gb:.1f} GB total, {available_vram_gb:.1f} GB free")
            else:
                available_vram_gb = None
        else:
            available_vram_gb = None
    
    if available_vram_gb is None:
        # Conservative fallback
        if conservative:
            base_batch = base_batches.get(model_size, 8)
            recommended = max(1, int(base_batch * size_factor * 0.5))  # 50% of base for safety
            logger.warning(f"Could not detect VRAM, using conservative batch size: {recommended}")
            return recommended
        else:
            base_batch = base_batches.get(model_size, 8)
            recommended = max(1, int(base_batch * size_factor))
            return recommended
    
    # Adjust based on available VRAM
    # Rough estimate: ~2-3 GB per batch for yolov8s at 640x640
    vram_per_batch = {
        'n': 1.5,
        's': 2.5,
        'm': 4.0,
        'l': 6.0,
        'x': 8.0
    }
    
    vram_per_batch_scaled = vram_per_batch.get(model_size, 2.5) * (imgsz / 640) ** 2
    
    # Reserve 2 GB for system and overhead
    usable_vram = max(0, available_vram_gb - 2.0)
    max_batch = int(usable_vram / vram_per_batch_scaled)
    
    # Apply size factor
    recommended = max(1, int(max_batch * size_factor))
    
    # Cap at reasonable maximum
    recommended = min(recommended, 64)
    
    logger.info(f"Recommended batch size: {recommended} (based on {available_vram_gb:.1f} GB VRAM)")
    return recommended


def auto_batch_size(imgsz=640, model_size='s', device='cuda:0', target_batch=None):
    """
    Automatically determine batch size, with fallback to target if provided.
    
    Args:
        imgsz: Image size
        model_size: Model size identifier
        device: Device string ('cuda:0' or 'cpu')
        target_batch: Target batch size (will be adjusted if too large)
        
    Returns:
        Recommended batch size
    """
    if device == 'cpu' or not torch.cuda.is_available():
        logger.info("CPU mode detected, using batch size 4")
        return 4
    
    device_id = int(device.split(':')[1]) if ':' in device else 0
    
    mem_info = get_gpu_memory_gb(device_id)
    if mem_info:
        available_vram, total_vram = mem_info
        recommended = estimate_batch_size(imgsz, model_size, available_vram, conservative=False)
    else:
        recommended = estimate_batch_size(imgsz, model_size, None, conservative=True)
    
    if target_batch is not None:
        # Use the smaller of target and recommended
        recommended = min(recommended, target_batch)
        logger.info(f"Using batch size: {recommended} (target was {target_batch})")
    
    return recommended

