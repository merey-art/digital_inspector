"""
Test Time Augmentation (TTA) routines for YOLO inference.
Includes flip and multi-scale augmentations.
"""

import torch
import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def flip_image(image: np.ndarray, flip_code: int = 1) -> np.ndarray:
    """
    Flip image horizontally or vertically.
    
    Args:
        image: Input image (H, W, C)
        flip_code: 1 for horizontal, 0 for vertical, -1 for both
        
    Returns:
        Flipped image
    """
    return cv2.flip(image, flip_code)


def scale_image(image: np.ndarray, scale: float) -> Tuple[np.ndarray, float]:
    """
    Scale image by factor.
    
    Args:
        image: Input image (H, W, C)
        scale: Scale factor
        
    Returns:
        Scaled image and scale factor
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return scaled, scale


def apply_tta_transforms(image: np.ndarray, scales: List[float] = [0.83, 1.0, 1.17], 
                         flip: bool = True) -> List[Tuple[np.ndarray, dict]]:
    """
    Generate TTA transformations for an image.
    
    Args:
        image: Input image (H, W, C)
        scales: List of scale factors to apply
        flip: Whether to include horizontal flip
        
    Returns:
        List of (transformed_image, transform_info) tuples
    """
    transforms = []
    
    # Original scale
    for scale in scales:
        if scale == 1.0:
            transforms.append((image.copy(), {'scale': 1.0, 'flip': False}))
        else:
            scaled_img, _ = scale_image(image, scale)
            transforms.append((scaled_img, {'scale': scale, 'flip': False}))
    
    # Flipped versions
    if flip:
        flipped_orig = flip_image(image, 1)
        transforms.append((flipped_orig, {'scale': 1.0, 'flip': True}))
        
        for scale in scales:
            if scale != 1.0:
                scaled_img, _ = scale_image(flipped_orig, scale)
                transforms.append((scaled_img, {'scale': scale, 'flip': True}))
    
    return transforms


def flip_boxes(boxes: np.ndarray, image_width: int, flip_code: int = 1) -> np.ndarray:
    """
    Flip bounding boxes coordinates.
    
    Args:
        boxes: Bounding boxes in format (x_center, y_center, width, height) normalized
        image_width: Image width (for horizontal flip)
        flip_code: 1 for horizontal, 0 for vertical
        
    Returns:
        Flipped boxes
    """
    boxes = boxes.copy()
    if flip_code == 1:  # Horizontal flip
        boxes[:, 0] = 1.0 - boxes[:, 0]  # Flip x_center
    elif flip_code == 0:  # Vertical flip
        boxes[:, 1] = 1.0 - boxes[:, 1]  # Flip y_center
    elif flip_code == -1:  # Both
        boxes[:, 0] = 1.0 - boxes[:, 0]
        boxes[:, 1] = 1.0 - boxes[:, 1]
    return boxes


def scale_boxes(boxes: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale bounding boxes (only needed if image was scaled, but boxes are normalized).
    Actually, if boxes are normalized, scaling doesn't change them.
    This is mainly for documentation.
    
    Args:
        boxes: Normalized bounding boxes
        scale: Scale factor
        
    Returns:
        Boxes (unchanged if normalized)
    """
    # If boxes are normalized (0-1), scaling doesn't affect them
    # This function is kept for consistency
    return boxes


def aggregate_tta_predictions(predictions: List[dict], transforms: List[dict], 
                              original_size: Tuple[int, int], 
                              conf_threshold: float = 0.25,
                              iou_threshold: float = 0.45) -> dict:
    """
    Aggregate predictions from multiple TTA transformations.
    
    Args:
        predictions: List of prediction dicts from model
        transforms: List of transform info dicts
        original_size: (height, width) of original image
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Aggregated predictions
    """
    from ultralytics.utils.ops import non_max_suppression
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    orig_h, orig_w = original_size
    
    for pred, transform in zip(predictions, transforms):
        # Extract boxes, scores, classes from prediction
        # Assuming prediction format from ultralytics
        if hasattr(pred, 'boxes'):
            boxes = pred.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = pred.boxes.conf.cpu().numpy()
            classes = pred.boxes.cls.cpu().numpy()
        else:
            # Fallback if different format
            continue
        
        # Filter by confidence
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        
        if len(boxes) == 0:
            continue
        
        # Reverse transform
        scale = transform.get('scale', 1.0)
        flipped = transform.get('flip', False)
        
        # Scale boxes back
        if scale != 1.0:
            boxes = boxes / scale
        
        # Flip boxes back if needed
        if flipped:
            # Convert to normalized center format for flipping
            boxes_norm = boxes.copy()
            boxes_norm[:, 0] = boxes[:, 0] / orig_w  # x1
            boxes_norm[:, 2] = boxes[:, 2] / orig_w  # x2
            boxes_norm[:, 1] = boxes[:, 1] / orig_h  # y1
            boxes_norm[:, 3] = boxes[:, 3] / orig_h  # y2
            
            # Flip x coordinates
            x1_new = 1.0 - boxes_norm[:, 2]
            x2_new = 1.0 - boxes_norm[:, 0]
            boxes_norm[:, 0] = x1_new
            boxes_norm[:, 2] = x2_new
            
            # Convert back to pixel coordinates
            boxes[:, 0] = boxes_norm[:, 0] * orig_w
            boxes[:, 2] = boxes_norm[:, 2] * orig_w
        
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_classes.append(classes)
    
    if len(all_boxes) == 0:
        return None
    
    # Concatenate all predictions
    all_boxes = np.concatenate(all_boxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    all_classes = np.concatenate(all_classes, axis=0)
    
    # Apply NMS
    # Convert to tensor format for NMS
    boxes_tensor = torch.from_numpy(all_boxes).float()
    scores_tensor = torch.from_numpy(all_scores).float()
    
    # Apply NMS
    keep = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, iou_threshold)
    
    final_boxes = all_boxes[keep.numpy()]
    final_scores = all_scores[keep.numpy()]
    final_classes = all_classes[keep.numpy()]
    
    # Create result dict
    result = {
        'boxes': final_boxes,
        'scores': final_scores,
        'classes': final_classes
    }
    
    return result

