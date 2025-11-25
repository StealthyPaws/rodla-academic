"""JSON serialization utilities"""
import numpy as np
from typing import Any


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization
    
    Args:
        obj: Object to convert (can be nested dict/list)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {
            key: convert_to_json_serializable(value) 
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj