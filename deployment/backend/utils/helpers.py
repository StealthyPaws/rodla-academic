"""Helper utility functions"""
import os
from typing import Optional


def parse_bool_string(value: str) -> bool:
    """
    Parse string to boolean
    
    Args:
        value: String value to parse
        
    Returns:
        Boolean value
    """
    return value.lower() in ('true', '1', 'yes', 'on')


def validate_score_threshold(score: float) -> float:
    """
    Validate score threshold is in valid range
    
    Args:
        score: Score threshold value
        
    Returns:
        Validated score
        
    Raises:
        ValueError: If score is out of range
    """
    if not 0 <= score <= 1:
        raise ValueError("Score threshold must be between 0 and 1")
    return score


def cleanup_temp_file(filepath: Optional[str]) -> None:
    """
    Safely cleanup temporary file
    
    Args:
        filepath: Path to temporary file
    """
    if filepath and os.path.exists(filepath):
        try:
            os.unlink(filepath)
        except Exception as e:
            print(f"Warning: Failed to cleanup temp file {filepath}: {e}")