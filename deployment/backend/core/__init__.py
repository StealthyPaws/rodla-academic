"""Core package for model management and dependencies"""
from .model_loader import (
    load_model,
    get_model,
    get_model_classes,
    get_device
)

__all__ = [
    'load_model',
    'get_model',
    'get_model_classes',
    'get_device'
]