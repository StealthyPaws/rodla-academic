"""Core detection service logic"""
from typing import List, Dict
from pathlib import Path
from PIL import Image
from mmdet.apis import inference_detector
from core.model_loader import get_model, get_model_classes
from config.settings import OUTPUT_DIR


def run_inference(image_path: str) -> tuple:
    """
    Run inference on an image
    
    Returns:
        tuple: (result, img_width, img_height)
    """
    model = get_model()
    
    # Get image info
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Run detection
    result = inference_detector(model, image_path)
    
    return result, img_width, img_height


def process_detections(result, score_thr: float = 0.3) -> List[Dict]:
    """
    Convert detection results to detailed JSON format
    
    Args:
        result: Detection result from the model
        score_thr: Confidence threshold
        
    Returns:
        List of detection dictionaries
    """
    detections = []
    classes = get_model_classes()
    
    for class_id, class_result in enumerate(result):
        for bbox in class_result:
            if bbox[4] > score_thr:
                x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                width = x2 - x1
                height = y2 - y1
                
                detections.append({
                    'class_id': int(class_id),
                    'class_name': classes[class_id],
                    'bbox': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': width, 'height': height,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2
                    },
                    'confidence': float(bbox[4]),
                    'area': float(width * height),
                    'aspect_ratio': float(width / height) if height > 0 else 0
                })
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections


def generate_annotated_image(
    image_path: str, 
    result, 
    score_thr: float,
    filename: str
) -> Path:
    """
    Generate annotated image with detections
    
    Args:
        image_path: Path to input image
        result: Detection result
        score_thr: Confidence threshold
        filename: Original filename
        
    Returns:
        Path to annotated image
    """
    model = get_model()
    output_path = OUTPUT_DIR / f"annotated_{filename}"
    
    model.show_result(
        image_path,
        result,
        score_thr=score_thr,
        show=False,
        out_file=str(output_path)
    )
    
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("Failed to generate annotated image")
    
    return output_path