"""
Exponential Moving Average (EMA) for model weights.
Provides smoother training and often better generalization.
"""

import torch
import torch.nn as nn
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class EMA:
    """
    Exponential Moving Average for model weights.
    """
    
    def __init__(self, model, decay=0.9999, device=None):
        """
        Initialize EMA.
        
        Args:
            model: PyTorch model
            decay: EMA decay factor (default 0.9999)
            device: Device to store EMA weights on
        """
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.register()
    
    def register(self):
        """Register model parameters for EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    def update(self):
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA weights to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state dict."""
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)


def create_ema_model(model, decay=0.9999, device=None):
    """
    Create EMA wrapper for model.
    
    Args:
        model: PyTorch model
        decay: EMA decay factor
        device: Device to store EMA weights on
        
    Returns:
        EMA wrapper instance
    """
    return EMA(model, decay=decay, device=device)

