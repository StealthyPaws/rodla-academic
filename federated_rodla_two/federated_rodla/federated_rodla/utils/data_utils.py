# utils/data_utils.py

import base64
import io
import json
import numpy as np
import torch
from PIL import Image
import cv2
from typing import Dict, List, Optional, Tuple
import os
import logging
import logging
from PIL import Image
from typing import Dict, List, Optional, Tuple
import io
import base64
import numpy as np
import torch # Added for FederatedDataConverter
from mmcv import Config # Added for FederatedDataConverter
from mmdet.datasets.pipelines import Compose # Added for FederatedDataConverter


logger = logging.getLogger(__name__)

class DataUtils:
    """Utility class for handling various data processing, now focused on utility for FL and robustness."""
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """Encodes a PIL Image to a base64 string."""
        buffered = io.BytesIO()
        # Ensure image is in RGB mode for JPEG encoding
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @staticmethod
    def decode_base64_to_image(image_data: str) -> Image.Image:
        """Decodes a base64 string back to a PIL Image."""
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes))

    @staticmethod
    def tensor_to_pil(img_tensor: torch.Tensor, img_norm_cfg: Optional[Dict] = None) -> Image.Image:
        """Converts a normalized PyTorch tensor (C, H, W) to a PIL Image (RGB)."""
        img_np = img_tensor.cpu().numpy()
        
        if img_norm_cfg:
            mean = np.array(img_norm_cfg['mean']).reshape(3, 1, 1)
            std = np.array(img_norm_cfg['std']).reshape(3, 1, 1)
            img_np = (img_np * std + mean) # Denormalize
        
        img_np = img_np.transpose(1, 2, 0).astype(np.uint8) # C, H, W to H, W, C
        return Image.fromarray(img_np)

    @staticmethod
    def pil_to_tensor(image: Image.Image, img_norm_cfg: Optional[Dict] = None) -> torch.Tensor:
        """Converts a PIL Image (RGB) to a normalized PyTorch tensor (C, H, W)."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_np = np.array(image).astype(np.float32) # H, W, C
        
        if img_norm_cfg:
            mean = np.array(img_norm_cfg['mean'])
            std = np.array(img_norm_cfg['std'])
            img_np = (img_np - mean) / std # Normalize
        
        return torch.from_numpy(img_np).permute(2, 0, 1) # H, W, C to C, H, W

    @staticmethod
    def validate_annotations(annotations: List[Dict], image_width: int, image_height: int) -> bool:
        """
        Validates the structure and content of annotations.
        Ensures bounding box coordinates are within image bounds and labels are valid (for PubLayNet).
        """
        if not annotations:
            return False

        for ann in annotations:
            if not all(k in ann for k in ['bbox', 'category_id']):
                logger.warning("Annotation missing 'bbox' or 'category_id'.")
                return False
            
            bbox = ann['bbox']
            category_id = ann['category_id']

            # Bounding box format: [x, y, width, height] (COCO format)
            if len(bbox) != 4 or not all(isinstance(coord, (int, float)) for coord in bbox):
                logger.warning("Invalid bounding box format.")
                return False
            
            x, y, w, h = bbox
            # Ensure bbox coordinates are valid and within image bounds
            if not (0 <= x < image_width and 0 <= y < image_height and
                    w > 0 and h > 0 and
                    x + w <= image_width + 1 and y + h <= image_height + 1): # Allow slight overflow for robustness
                logger.warning(f"Bounding box out of image bounds or invalid size: {bbox} for image {image_width}x{image_height}")
                return False

            # Validate category_id for PubLayNet (categories 1-5)
            if not (1 <= category_id <= 5):
                logger.warning(f"Invalid category ID for PubLayNet: {category_id}. Expected 1-5.")
                return False
        return True

    @staticmethod
    def adjust_bboxes_for_transformation(bboxes: List[List[float]], img_info: Dict, transform_matrix: np.ndarray) -> List[List[float]]:
        """
        Adjusts bounding box coordinates based on a 2x3 affine transformation matrix.
        This method is now more generic for geometric transformations during augmentation.
        Note: This is typically for single transformations. Complex pipelines might need integrated augmentation handling.
        """
        if not bboxes or transform_matrix.shape != (2, 3):
            return bboxes

        adjusted_bboxes = []
        for bbox in bboxes:
            # Convert [x1, y1, x2, y2] to 4 corner points (x,y)
            x1, y1, x2, y2 = bbox
            corners = np.array([
                [x1, y1], [x2, y1], [x1, y2], [x2, y2]
            ], dtype=np.float32)

            # Apply transformation
            # Add a column of ones for affine transform: [x, y, 1]
            corners_homo = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
            transformed_corners = corners_homo @ transform_matrix.T # (N, 3) @ (3, 2) = (N, 2)

            # Find new min/max to form the new bounding box
            min_x, min_y = np.min(transformed_corners, axis=0)
            max_x, max_y = np.max(transformed_corners, axis=0)

            # Clamp to image bounds if necessary (img_info should contain original_width/height)
            orig_w = img_info.get('ori_shape')[1] if img_info.get('ori_shape') else max_x # Fallback
            orig_h = img_info.get('ori_shape')[0] if img_info.get('ori_shape') else max_y # Fallback
            
            new_x1 = max(0, min_x)
            new_y1 = max(0, min_y)
            new_x2 = min(orig_w, max_x)
            new_y2 = min(orig_h, max_y)

            # Ensure valid bbox after clamping
            if new_x2 > new_x1 and new_y2 > new_y1:
                adjusted_bboxes.append([new_x1, new_y1, new_x2, new_y2])
            else:
                logger.warning(f"Transformation resulted in invalid bbox: {bbox} -> {[new_x1, new_y1, new_x2, new_y2]}. Skipping.")

        return adjusted_bboxes

    # The following methods are no longer relevant in this FedAvg architecture
    # @staticmethod
    # def create_sample_metadata(...)
    # @staticmethod
    # def calculate_privacy_score(...)
    # @staticmethod
    # def save_federated_sample(...)
    # @staticmethod
    # def load_federated_sample(...)
    # @staticmethod
    # def create_federated_batch(...)
    # @staticmethod
    # def validate_federated_batch(...)


class FederatedDataConverter:
    """
    This class is now largely deprecated in a true FedAvg setup, as raw data isn't exchanged
    in the same 'federated sample' format. However, its methods for converting to/from
    RoDLA format might still be useful for internal client/server data handling if needed.
    Keeping for potential utility, but primary usage is for model exchange.
    """
    
    # FederatedDataConverter.federated_to_rodla and rodla_to_federated are less directly used
    # in the model-update FL setup, as clients handle their own data loading for local training.
    # The MMDetection pipeline handles transformations within the client's local training.
    
    @staticmethod
    def federated_to_rodla(federated_sample: Dict, config_path: str = None) -> Dict:
        """
        Converts a single federated sample dict (image data + annotations)
        into a format suitable for RoDLA model input (img_tensor, gt_bboxes, gt_labels, img_metas).
        This would primarily be used if you needed to process a single sample from a client's
        raw data format for *some* server-side processing, but not for direct model training.
        """
        if config_path:
            cfg = Config.fromfile(config_path)
            test_pipeline = Compose(cfg.data.test.pipeline) # Or train pipeline for augmentations
        else:
            # Default normalization if no config given, consistent with PubLayNetDataset
            img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
            test_pipeline = Compose([
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(1333, 800), # Default scale
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(type='Normalize', **img_norm_cfg),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img']),
                    ])
            ])

        image_data_b64 = federated_sample['image_data']
        image_pil = DataUtils.decode_base64_to_image(image_data_b64)
        
        # Prepare data for pipeline
        # MMDetection expects filename, img, height, width initially
        data_info = {
            'img': np.array(image_pil),
            'img_info': {'filename': federated_sample.get('filename', 'unknown.jpg'), 
                         'width': image_pil.width, 'height': image_pil.height},
            'ann_info': {'bboxes': [], 'labels': []} # No gt_bboxes/labels in input dict for test pipeline
        }
        
        # If annotations are present in federated_sample and needed for validation/augmentation
        if 'annotations' in federated_sample:
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            bboxes_xyxy = []
            labels = []
            for ann in federated_sample['annotations']:
                x, y, w, h = ann['bbox']
                bboxes_xyxy.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
            data_info['ann_info']['bboxes'] = np.array(bboxes_xyxy, dtype=np.float32)
            data_info['ann_info']['labels'] = np.array(labels, dtype=np.int64)
            # You might need to adjust the pipeline to handle gt_bboxes/labels in this case.
            # Usually, for 'test' pipelines, these are not provided.
            # For training, a different pipeline that handles gt_bboxes/labels is used.
            # This method assumes conversion to basic RoDLA input, not a full training sample.

        data = test_pipeline(data_info)
        
        # Manually add gt_bboxes and gt_labels if they were passed and are needed for a 'training' style output
        if 'annotations' in federated_sample:
            data['gt_bboxes'] = torch.tensor(bboxes_xyxy, dtype=torch.float32)
            data['gt_labels'] = torch.tensor(labels, dtype=torch.int64)

        return data

    @staticmethod
    def rodla_to_federated(rodla_batch: Dict) -> List[Dict]:
        """
        Converts a batch of RoDLA-formatted samples into a list of federated sample dictionaries.
        This method is now less directly used in FedAvg as clients exchange models, not raw data.
        """
        federated_samples = []
        imgs = rodla_batch['img']
        img_metas = rodla_batch['img_metas']
        gt_bboxes_batch = rodla_batch['gt_bboxes']
        gt_labels_batch = rodla_batch['gt_labels']
        
        # Default normalization if not in config
        img_norm_cfg = img_metas[0]['img_norm_cfg'] if 'img_norm_cfg' in img_metas[0] else dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

        for i in range(len(imgs)):
            img_tensor = imgs[i]
            img_meta = img_metas[i]
            gt_bboxes = gt_bboxes_batch[i]
            gt_labels = gt_labels_batch[i]

            image_pil = DataUtils.tensor_to_pil(img_tensor, img_norm_cfg)
            image_data_b64 = DataUtils.encode_image_to_base64(image_pil)

            annotations = []
            for bbox, label in zip(gt_bboxes, gt_labels):
                x1, y1, x2, y2 = bbox.tolist()
                annotations.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1], # Convert to [x, y, w, h]
                    'category_id': label.item()
                })
            
            federated_samples.append({
                'filename': img_meta['filename'],
                'image_data': image_data_b64,
                'annotations': annotations,
                'img_original_width': img_meta['ori_shape'][1],
                'img_original_height': img_meta['ori_shape'][0],
                'augmentation_metadata': {} # No explicit perturbation here
            })
        return federated_samples

# Utility functions for easy access (still relevant for client-side data handling)
def encode_image(image: Image.Image) -> str:
    return DataUtils.encode_image_to_base64(image)

def decode_image(image_data: str) -> Image.Image:
    return DataUtils.decode_base64_to_image(image_data)

def validate_sample(sample: Dict) -> bool:
    """Quick validation of a federated sample"""
    if 'image_data' not in sample or 'annotations' not in sample:
        return False
    
    image = decode_image(sample['image_data'])
    if image is None:
        return False
    
    return DataUtils.validate_annotations(sample['annotations'], image.size)

# Initialize logging
import time
# The original validate_sample function is removed as raw federated samples are not processed directly
# on the server in this architecture. Clients manage their own data.

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)