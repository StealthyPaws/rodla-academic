"""Human-readable interpretation generation service"""
from typing import Dict, List


def generate_comprehensive_interpretation(
    core_metrics: Dict,
    rodla_metrics: Dict,
    class_metrics: Dict,
    layout_complexity: Dict,
    robustness_indicators: Dict
) -> Dict:
    """
    Generate detailed human-readable interpretation of results
    
    Args:
        core_metrics: Core detection metrics
        rodla_metrics: RoDLA-specific metrics
        class_metrics: Per-class analysis
        layout_complexity: Layout complexity metrics
        robustness_indicators: Robustness indicators
        
    Returns:
        Dictionary with interpretation sections
    """
    if not core_metrics.get("summary"):
        return {"overview": "No detections found in the document."}
    
    summary = core_metrics["summary"]
    
    # Generate overview
    overview = _generate_overview(summary, robustness_indicators)
    
    # Most common elements
    top_elements = _generate_top_elements(class_metrics)
    
    # RoDLA analysis
    rodla_desc = _generate_rodla_description(rodla_metrics)
    
    # Layout complexity
    complexity_desc = _generate_complexity_description(layout_complexity)
    
    # Key findings
    key_findings = _generate_key_findings(
        summary, robustness_indicators, layout_complexity
    )
    
    # Perturbation assessment
    perturbation = _generate_perturbation_assessment(
        rodla_metrics, robustness_indicators
    )
    
    # Recommendations
    recommendations = _generate_recommendations(
        summary, rodla_metrics, layout_complexity
    )
    
    return {
        "overview": overview,
        "top_elements": top_elements,
        "rodla_analysis": rodla_desc,
        "layout_complexity": complexity_desc,
        "key_findings": key_findings,
        "perturbation_assessment": perturbation,
        "recommendations": recommendations,
        "confidence_summary": {
            "level": robustness_indicators.get('model_certainty', 'unknown'),
            "stability": robustness_indicators.get('prediction_consistency', 'unknown'),
            "rating": robustness_indicators.get('robustness_rating', {}).get('rating', 'unknown')
        }
    }


def _generate_overview(summary: Dict, robustness: Dict) -> str:
    """Generate overview text"""
    return f"""Document Analysis Summary:
Detected {summary['total_detections']} layout elements across {summary['unique_classes']} different classes. 
The model achieved an average confidence of {summary['average_confidence']:.1%}, indicating 
{robustness.get('model_certainty', 'unknown')} certainty in predictions. 
The detected elements cover {summary['coverage_percentage']:.1f}% of the document area."""


def _generate_top_elements(class_metrics: Dict) -> str:
    """Generate top elements description"""
    if not class_metrics:
        return "No class data available."
    
    top_3 = sorted(
        class_metrics.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:3]
    
    descriptions = [
        f"{cls} ({data['count']} instances, {data['confidence_stats']['mean']:.1%} avg confidence)"
        for cls, data in top_3
    ]
    
    return "The most common elements are: " + ", ".join(descriptions)


def _generate_rodla_description(rodla_metrics: Dict) -> str:
    """Generate RoDLA analysis description"""
    if not rodla_metrics:
        return "RoDLA metrics not available."
    
    interp = rodla_metrics.get('interpretation', {})
    
    return f"""RoDLA Robustness Analysis:
Estimated perturbation effect (mPE): {rodla_metrics.get('estimated_mPE', 'N/A')} - {interp.get('mPE_level', 'unknown')} level
Estimated robustness degradation (mRD): {rodla_metrics.get('estimated_mRD', 'N/A')} - {interp.get('mRD_level', 'unknown')}
Overall robustness: {interp.get('overall_robustness', 'unknown')}
Robustness score: {rodla_metrics.get('robustness_score', 'N/A')}/100"""


def _generate_complexity_description(layout_complexity: Dict) -> str:
    """Generate complexity description"""
    if not layout_complexity:
        return "Complexity metrics not available."
    
    chars = layout_complexity.get('layout_characteristics', {})
    
    return f"""Layout Complexity:
Complexity level: {layout_complexity.get('complexity_level', 'unknown')} (score: {layout_complexity.get('complexity_score', 0):.1f}/100)
Class diversity: {layout_complexity.get('class_diversity', 0)} unique element types
Detection density: {layout_complexity.get('detection_density', 0):.2f} elements per megapixel
Spatial structure: {'Structured' if chars.get('is_structured') else 'Unstructured'}"""


def _generate_key_findings(
    summary: Dict,
    robustness: Dict,
    complexity: Dict
) -> List[str]:
    """Generate key findings list"""
    findings = []
    
    # Confidence findings
    avg_conf = summary.get('average_confidence', 0)
    if avg_conf > 0.85:
        findings.append("✓ Excellent detection confidence - model is highly certain about predictions")
    elif avg_conf < 0.6:
        findings.append("⚠ Lower confidence detected - document may have quality issues or unusual layout")
    
    # Coverage findings
    coverage = summary.get('coverage_percentage', 0)
    if coverage > 70:
        findings.append("✓ High document coverage - most of the page contains layout elements")
    elif coverage < 20:
        findings.append("⚠ Low document coverage - sparse layout with significant white space")
    
    # Robustness findings
    stability = robustness.get('stability_score', 0)
    if stability > 80:
        findings.append("✓ High stability - consistent predictions across detections")
    elif stability < 50:
        findings.append("⚠ Variable predictions - consider potential document perturbations")
    
    # Complexity findings
    level = complexity.get('complexity_level', '')
    if level == 'complex':
        findings.append("ℹ Complex document structure with diverse element types")
    elif level == 'simple':
        findings.append("ℹ Simple document structure with limited element diversity")
    
    return findings


def _generate_perturbation_assessment(
    rodla_metrics: Dict,
    robustness: Dict
) -> str:
    """Generate perturbation assessment"""
    assessment = "Based on RoDLA metrics:\n"
    
    mrd = rodla_metrics.get('estimated_mRD', 0)
    if mrd < 100:
        assessment += "✓ Minimal to no perturbation effects detected\n"
    elif mrd < 150:
        assessment += "⚠ Moderate perturbation effects may be present\n"
    else:
        assessment += "⚠ Significant perturbation effects detected\n"
    
    cv = robustness.get('coefficient_of_variation', 0)
    assessment += f"Confidence variability: {cv:.3f} "
    assessment += f"({'low variability - stable' if cv < 0.15 else 'high variability - potentially perturbed'})"
    
    return assessment


def _generate_recommendations(
    summary: Dict,
    rodla_metrics: Dict,
    complexity: Dict
) -> List[str]:
    """Generate recommendations"""
    recommendations = []
    
    avg_conf = summary.get('average_confidence', 0)
    if avg_conf < 0.7:
        recommendations.append(
            "Consider pre-processing the image (denoising, contrast adjustment)"
        )
    
    mrd = rodla_metrics.get('estimated_mRD', 0)
    if mrd > 150:
        recommendations.append(
            "High robustness degradation detected - document may have perturbations "
            "(blur, noise, distortions)"
        )
    
    chars = complexity.get('layout_characteristics', {})
    if chars.get('is_dense') and avg_conf < 0.75:
        recommendations.append(
            "Dense layout with lower confidence - verify detection accuracy manually"
        )
    
    if not recommendations:
        recommendations.append(
            "No specific recommendations - detection quality is good"
        )
    
    return recommendations