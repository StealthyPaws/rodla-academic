"""RoDLA-specific metrics (mPE and mRD estimation)"""
from typing import List, Dict
import numpy as np


def calculate_rodla_metrics(detections: List[Dict], core_metrics: Dict) -> Dict:
    """
    Calculate RoDLA-specific metrics (mPE and mRD estimation)
    
    Note: Full mPE/mRD requires clean baseline and perturbation assessment.
    Here we provide proxy metrics based on current detection.
    
    Args:
        detections: List of detection dictionaries
        core_metrics: Core metrics dictionary
        
    Returns:
        Dictionary of RoDLA metrics
    """
    if not detections:
        return {}
    
    avg_conf = core_metrics["summary"]["average_confidence"]
    
    # Estimate perturbation effect based on confidence distribution
    conf_values = [d['confidence'] for d in detections]
    conf_std = np.std(conf_values)
    conf_range = max(conf_values) - min(conf_values)
    
    # Proxy mPE: Higher std and range suggest more perturbation effect
    estimated_mPE = round((conf_std * 100) + (conf_range * 50), 2)
    
    # Proxy mRD: Based on deviation from expected performance
    expected_performance = 0.85  # Typical clean performance
    degradation = max(0, (expected_performance - avg_conf) * 100)
    
    estimated_mRD = round(
        (degradation / max(estimated_mPE, 1)) * 100, 2
    ) if estimated_mPE > 0 else 0
    
    # Robustness score (0-100 scale)
    robustness_score = round((1 - (estimated_mRD / 200)) * 100, 2)
    
    return {
        "note": "These are estimated metrics. Full mRD/mPE require clean baseline comparison.",
        "estimated_mPE": estimated_mPE,
        "estimated_mRD": estimated_mRD,
        "confidence_std": round(conf_std, 4),
        "confidence_range": round(conf_range, 4),
        "robustness_score": robustness_score,
        "interpretation": {
            "mPE_level": _get_mpe_level(estimated_mPE),
            "mRD_level": _get_mrd_level(estimated_mRD),
            "overall_robustness": _get_robustness_level(avg_conf)
        }
    }


def calculate_robustness_indicators(
    detections: List[Dict], 
    core_metrics: Dict
) -> Dict:
    """
    Calculate indicators of model robustness and detection stability
    
    Args:
        detections: List of detection dictionaries
        core_metrics: Core metrics dictionary
        
    Returns:
        Dictionary of robustness indicators
    """
    if not detections:
        return {}
    
    confidences = [d['confidence'] for d in detections]
    avg_conf = core_metrics["summary"]["average_confidence"]
    
    # Coefficient of variation (lower = more stable)
    cv = (np.std(confidences) / avg_conf) if avg_conf > 0 else 0
    
    # Detection consistency score
    high_conf_ratio = sum(1 for c in confidences if c >= 0.8) / len(confidences)
    
    # Stability score
    stability_score = round((1 - cv) * 100, 2)
    
    return {
        "stability_score": stability_score,
        "coefficient_of_variation": round(cv, 4),
        "high_confidence_ratio": round(high_conf_ratio, 4),
        "prediction_consistency": _get_consistency_level(cv),
        "model_certainty": _get_certainty_level(avg_conf),
        "robustness_rating": _calculate_robustness_rating(
            avg_conf, cv, high_conf_ratio
        )
    }


def _get_mpe_level(mpe: float) -> str:
    """Categorize mPE level"""
    if mpe < 20:
        return "low"
    elif mpe < 40:
        return "medium"
    else:
        return "high"


def _get_mrd_level(mrd: float) -> str:
    """Categorize mRD level"""
    if mrd < 100:
        return "excellent"
    elif mrd < 150:
        return "good"
    else:
        return "needs_improvement"


def _get_robustness_level(avg_conf: float) -> str:
    """Categorize overall robustness"""
    if avg_conf > 0.8:
        return "high"
    elif avg_conf > 0.6:
        return "medium"
    else:
        return "low"


def _get_consistency_level(cv: float) -> str:
    """Categorize prediction consistency"""
    if cv < 0.15:
        return "high"
    elif cv < 0.3:
        return "medium"
    else:
        return "low"


def _get_certainty_level(avg_conf: float) -> str:
    """Categorize model certainty"""
    if avg_conf > 0.8:
        return "high"
    elif avg_conf > 0.6:
        return "medium"
    else:
        return "low"


def _calculate_robustness_rating(
    avg_conf: float, 
    cv: float, 
    high_conf_ratio: float
) -> Dict[str, float]:
    """Calculate overall robustness rating"""
    score = (avg_conf * 40) + ((1 - cv) * 30) + (high_conf_ratio * 30)
    
    if score >= 80:
        rating = "excellent"
    elif score >= 60:
        rating = "good"
    elif score >= 40:
        rating = "fair"
    else:
        rating = "poor"
    
    return {"rating": rating, "score": round(score, 2)}