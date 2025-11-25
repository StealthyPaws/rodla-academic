import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from federated.perturbation_engine import PubLayNetPerturbationEngine
from federated.data_client import PubLayNetDataset # Re-use the dataset loader for evaluation
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataloader # To build test dataloader if desired
from mmdet.core import eval_map # For calculating mAP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_on_perturbed_data(model, cfg, data_root, annotation_file, perturbation_types, severity_levels, device):
    """
    Evaluates the model's robustness by applying perturbations to the test dataset
    and running inference.
    """
    results = {}

    # Build the clean test dataset
    test_dataset = PubLayNetDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        split='val', # Or 'test' depending on your split and data
        max_samples=None # Evaluate on full test set
    )
    
    # Define img_norm_cfg for inverse normalization before perturbing (needed if dataset outputs normalized tensors)
    # The PubLayNetDataset output is already normalized, so we need to reverse it to get PIL image for perturbation.
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    )
    
    # Prepare ground truth for mAP calculation
    gt_bboxes_list = []
    gt_labels_list = []
    gt_ignore_list = [None] * len(test_dataset) # MMDetection expects this

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        gt_bboxes_list.append(sample['gt_bboxes'].cpu().numpy())
        gt_labels_list.append(sample['gt_labels'].cpu().numpy())


    for pert_type in perturbation_types:
        for severity in severity_levels:
            logger.info(f"Evaluating with Perturbation: {pert_type}, Severity: {severity}")
            engine = PubLayNetPerturbationEngine(perturbation_type=pert_type, severity_level=severity)
            
            model_outputs = [] # To store inference results

            for i in range(len(test_dataset)):
                sample = test_dataset[i]
                
                # Extract original image tensor (normalized)
                original_img_tensor = sample['img'] # C, H, W tensor
                
                # Denormalize tensor to NumPy array (H, W, C, uint8) for PIL conversion
                img_np_denorm = original_img_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np_denorm = (img_np_denorm * np.array(img_norm_cfg['std']) + np.array(img_norm_cfg['mean'])).astype(np.uint8)
                pil_img = Image.fromarray(img_np_denorm)

                # Apply perturbation
                perturbed_pil_img, _ = engine.perturb(pil_img, perturbation_type=pert_type)
                
                # Convert perturbed PIL image back to NumPy array for MMDetection inference
                perturbed_img_np = np.array(perturbed_pil_img) # H, W, C, uint8
                
                # Perform inference
                # inference_detector handles normalization and resizing internally when given a numpy array
                result = inference_detector(model, perturbed_img_np)
                
                # MMDetection's result format is a tuple (bboxes, segms), where bboxes is a list of np.ndarrays
                # each with shape (n, 5) for (x1, y1, x2, y2, score) for each class.
                # Convert to a list of (n, 5) arrays for mAP
                formatted_result = [det for det in result if det.size > 0] # Filter empty detections
                model_outputs.append(formatted_result)

            # Calculate mAP for the current perturbation and severity
            # MMDetection's eval_map requires a specific format for results (list of lists of np.ndarray)
            # and ground truth (list of dicts, or list of np.ndarray for bboxes/labels)
            
            # Categories needs to be consistent with PubLayNetDataset's categories:
            # {'id': 1, 'name': 'text'}, {'id': 2, 'name': 'title'}, {'id': 3, 'name': 'list'},
            # {'id': 4, 'name': 'table'}, {'id': 5, 'name': 'figure'}
            class_names = ['text', 'title', 'list', 'table', 'figure'] 

            # Flatten model_outputs to list of dicts for eval_map, or ensure it matches expected format
            # This part needs careful alignment with mmdet.core.eval_map's input format.
            # Example:
            # mean_ap, _ = eval_map(model_outputs, gt_bboxes_list, gt_labels_list, class_names=class_names, ...)
            
            # For this simplified implementation, we'll just show the concept.
            # Actual mAP calculation might require more integration with MMDet's test functions.
            
            mean_ap, _ = eval_map(
                model_outputs, 
                gt_bboxes_list, 
                gt_labels_list, 
                gt_ignore_list=gt_ignore_list,
                scale_ranges=None, # Or your specific scale ranges
                iou_thr=0.5, 
                dataset=class_names, # List of class names
                logger=logger
            )
            
            logger.info(f"  --> Evaluation completed for {pert_type}-{severity}. mAP: {mean_ap:.4f}")
            results[f'{pert_type}_s{severity}_mAP'] = f"{mean_ap:.4f}"

    return results

def main():
    parser = argparse.ArgumentParser(description='Run RoDLA Robustness Evaluation')
    parser.add_argument('--config', required=True, 
                       help='Path to the model config file (e.g., configs/publaynet/rodla_internimage_xl_publaynet.py)')
    parser.add_argument('--checkpoint', required=True, 
                       help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--data-root', required=True,
                       help='Path to PubLayNet dataset root directory (for clean test images)')
    parser.add_argument('--annotation-file', required=True,
                       help='Path to PubLayNet annotations JSON file (for clean test split)')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference (e.g., cuda:0 or cpu)')
    
    args = parser.parse_args()

    # Define all perturbation types and severity levels from RoDLA paper
    all_perturbation_types = [
        'background', 'defocus', 'illumination', 'ink_bleeding', 'ink_holdout',
        'keystoning', 'rotation', 'speckle', 'texture', 'vibration', 
        'warping', 'watermark'
    ]
    all_severity_levels = [1, 2, 3]

    # Initialize model
    logger.info(f"Initializing model from config: {args.config}")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval() # Set to evaluation mode
    logger.info("Model initialized successfully.")

    # Run evaluation
    robustness_results = evaluate_model_on_perturbed_data(
        model, 
        args.config, 
        args.data_root, 
        args.annotation_file, 
        all_perturbation_types, 
        all_severity_levels,
        args.device
    )

    logger.info("\n--- Robustness Evaluation Summary ---")
    for key, value in robustness_results.items():
        logger.info(f"{key}: {value}")
    logger.info("-----------------------------------\n")

if __name__ == '__main__':
    main()