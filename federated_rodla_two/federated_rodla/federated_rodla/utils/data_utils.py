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

logger = logging.getLogger(__name__)

class DataUtils:
    """Utility class for handling federated data processing"""
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """
        Encode PIL Image to base64 string
        
        Args:
            image: PIL Image object
            format: Image format (JPEG, PNG)
            quality: JPEG quality (1-100)
            
        Returns:
            base64 encoded string
        """
        try:
            buffered = io.BytesIO()
            image.save(buffered, format=format, quality=quality)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            logger.error(f"Error encoding image to base64: {e}")
            return ""
    
    @staticmethod
    def decode_base64_to_image(image_data: str) -> Optional[Image.Image]:
        """
        Decode base64 string to PIL Image
        
        Args:
            image_data: base64 encoded image string
            
        Returns:
            PIL Image or None if decoding fails
        """
        try:
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
                
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert('RGB')  # Ensure RGB format
        except Exception as e:
            logger.error(f"Error decoding base64 to image: {e}")
            return None
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor, denormalize: bool = True) -> Image.Image:
        """
        Convert torch tensor to PIL Image
        
        Args:
            tensor: Image tensor [C, H, W]
            denormalize: Whether to reverse ImageNet normalization
            
        Returns:
            PIL Image
        """
        try:
            # Detach and convert to numpy
            if tensor.requires_grad:
                tensor = tensor.detach()
            
            # Move to CPU and convert to numpy
            tensor = tensor.cpu().numpy()
            
            # Handle different tensor shapes
            if tensor.shape[0] == 3:  # [C, H, W]
                img_np = tensor.transpose(1, 2, 0)
            else:  # [H, W, C]
                img_np = tensor
            
            # Denormalize if needed (reverse ImageNet normalization)
            if denormalize:
                mean = np.array([123.675, 116.28, 103.53])
                std = np.array([58.395, 57.12, 57.375])
                img_np = img_np * std + mean
            
            # Clip and convert to uint8
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_np)
        except Exception as e:
            logger.error(f"Error converting tensor to PIL: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (224, 224), color='white')
    
    @staticmethod
    def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
        """
        Convert PIL Image to normalized torch tensor
        
        Args:
            image: PIL Image
            normalize: Whether to apply ImageNet normalization
            
        Returns:
            Normalized tensor [C, H, W]
        """
        try:
            # Convert to numpy
            img_np = np.array(image).astype(np.float32)
            
            # Convert RGB to BGR if needed (OpenCV format)
            if img_np.shape[2] == 3:
                img_np = img_np[:, :, ::-1]  # RGB to BGR
            
            # Normalize
            if normalize:
                mean = np.array([123.675, 116.28, 103.53])
                std = np.array([58.395, 57.12, 57.375])
                img_np = (img_np - mean) / std
            
            # Convert to tensor and rearrange dimensions
            tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
            
            return tensor
        except Exception as e:
            logger.error(f"Error converting PIL to tensor: {e}")
            return torch.zeros(3, 224, 224)
    
    @staticmethod
    def validate_annotations(annotations: Dict, image_size: Tuple[int, int]) -> bool:
        """
        Validate annotation format and values
        
        Args:
            annotations: Annotation dictionary
            image_size: (width, height) of image
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_keys = ['bboxes', 'labels', 'image_size']
            
            # Check required keys
            for key in required_keys:
                if key not in annotations:
                    logger.warning(f"Missing required key in annotations: {key}")
                    return False
            
            # Validate bboxes
            bboxes = annotations['bboxes']
            if not isinstance(bboxes, list):
                logger.warning("Bboxes must be a list")
                return False
            
            for bbox in bboxes:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}")
                    return False
                
                # Check if bbox coordinates are within image bounds
                x1, y1, x2, y2 = bbox
                if x1 < 0 or y1 < 0 or x2 > image_size[0] or y2 > image_size[1]:
                    logger.warning(f"Bbox out of image bounds: {bbox}, image_size: {image_size}")
                    return False
            
            # Validate labels
            labels = annotations['labels']
            if not isinstance(labels, list):
                logger.warning("Labels must be a list")
                return False
            
            if len(bboxes) != len(labels):
                logger.warning("Number of bboxes and labels must match")
                return False
            
            # Validate label values (M6Doc has 75 classes)
            for label in labels:
                if not isinstance(label, int) or label < 0 or label >= 75:
                    logger.warning(f"Invalid label: {label}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating annotations: {e}")
            return False
    
    @staticmethod
    def adjust_bboxes_for_transformation(bboxes: List[List[float]], 
                                       original_size: Tuple[int, int],
                                       new_size: Tuple[int, int],
                                       transform_info: Dict) -> List[List[float]]:
        """
        Adjust bounding boxes for image transformations
        
        Args:
            bboxes: List of [x1, y1, x2, y2]
            original_size: (width, height) of original image
            new_size: (width, height) of transformed image
            transform_info: Information about applied transformations
            
        Returns:
            Adjusted bounding boxes
        """
        try:
            adjusted_bboxes = []
            orig_w, orig_h = original_size
            new_w, new_h = new_size
            
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                
                # Apply scaling
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
                
                # Apply rotation if present
                if 'rotation' in transform_info:
                    angle = transform_info['rotation']
                    # Simplified rotation adjustment (for small angles)
                    if abs(angle) > 5:
                        # For significant rotations, we'd need proper affine transformation
                        # This is a simplified version
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        # Approximate adjustment - in practice, use proper rotation matrix
                        pass
                
                adjusted_bboxes.append([x1, y1, x2, y2])
            
            return adjusted_bboxes
            
        except Exception as e:
            logger.error(f"Error adjusting bboxes: {e}")
            return bboxes
    
    @staticmethod
    def create_sample_metadata(client_id: str, 
                             privacy_level: str,
                             augmentation_info: Dict,
                             original_file: str = "") -> Dict:
        """
        Create standardized metadata for federated samples
        
        Args:
            client_id: Identifier for the client
            privacy_level: Privacy level (low/medium/high)
            augmentation_info: Information about applied augmentations
            original_file: Original filename (optional)
            
        Returns:
            Metadata dictionary
        """
        return {
            'client_id': client_id,
            'privacy_level': privacy_level,
            'augmentation_info': augmentation_info,
            'original_file': original_file,
            'timestamp': int(time.time()),
            'version': '1.0'
        }
    
    @staticmethod
    def calculate_privacy_score(augmentation_info: Dict) -> float:
        """
        Calculate a privacy score based on augmentation strength
        
        Args:
            augmentation_info: Information about applied augmentations
            
        Returns:
            Privacy score between 0 (low privacy) and 1 (high privacy)
        """
        score = 0.0
        transforms = augmentation_info.get('applied_transforms', [])
        parameters = augmentation_info.get('parameters', {})
        
        # Score based on number and strength of transformations
        if 'rotation' in transforms:
            angle = abs(parameters.get('rotation_angle', 0))
            score += min(angle / 15.0, 1.0) * 0.2
        
        if 'scaling' in transforms:
            scale = parameters.get('scale_factor', 1.0)
            deviation = abs(scale - 1.0)
            score += min(deviation / 0.3, 1.0) * 0.2
        
        if 'perspective' in transforms:
            score += 0.3
        
        if 'gaussian_blur' in transforms:
            radius = parameters.get('blur_radius', 0)
            score += min(radius / 2.0, 1.0) * 0.15
        
        if 'gaussian_noise' in transforms:
            score += 0.15
        
        return min(score, 1.0)
    
    @staticmethod
    def save_federated_sample(sample: Dict, output_dir: str, sample_id: str) -> bool:
        """
        Save federated sample to disk
        
        Args:
            sample: Sample dictionary
            output_dir: Output directory
            sample_id: Unique sample identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save image
            image = DataUtils.decode_base64_to_image(sample['image_data'])
            if image:
                image_path = os.path.join(output_dir, f"{sample_id}.jpg")
                image.save(image_path, "JPEG", quality=85)
            
            # Save annotations and metadata
            metadata_path = os.path.join(output_dir, f"{sample_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump({
                    'annotations': sample['annotations'],
                    'metadata': sample['metadata']
                }, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving federated sample: {e}")
            return False
    
    @staticmethod
    def load_federated_sample(input_dir: str, sample_id: str) -> Optional[Dict]:
        """
        Load federated sample from disk
        
        Args:
            input_dir: Input directory
            sample_id: Sample identifier
            
        Returns:
            Sample dictionary or None if loading fails
        """
        try:
            # Load image
            image_path = os.path.join(input_dir, f"{sample_id}.jpg")
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Load metadata
            metadata_path = os.path.join(input_dir, f"{sample_id}.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'image_data': image_data,
                'annotations': metadata['annotations'],
                'metadata': metadata['metadata']
            }
            
        except Exception as e:
            logger.error(f"Error loading federated sample: {e}")
            return None
    
    @staticmethod
    def create_federated_batch(samples: List[Dict]) -> Dict:
        """
        Create a batch of federated samples for transmission
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Batch dictionary
        """
        return {
            'batch_id': str(int(time.time())),
            'samples': samples,
            'batch_size': len(samples),
            'total_clients': len(set(sample['metadata']['client_id'] for sample in samples)),
            'average_privacy_score': np.mean([DataUtils.calculate_privacy_score(
                sample['metadata']['augmentation_info']) for sample in samples])
        }
    
    @staticmethod
    def validate_federated_batch(batch: Dict) -> Tuple[bool, str]:
        """
        Validate a federated batch
        
        Args:
            batch: Batch dictionary
            
        Returns:
            (is_valid, error_message)
        """
        try:
            required_keys = ['batch_id', 'samples', 'batch_size']
            for key in required_keys:
                if key not in batch:
                    return False, f"Missing required key: {key}"
            
            if not isinstance(batch['samples'], list):
                return False, "Samples must be a list"
            
            if len(batch['samples']) != batch['batch_size']:
                return False, "Batch size doesn't match number of samples"
            
            # Validate each sample
            for i, sample in enumerate(batch['samples']):
                if 'image_data' not in sample:
                    return False, f"Sample {i} missing image_data"
                
                if 'annotations' not in sample:
                    return False, f"Sample {i} missing annotations"
                
                if 'metadata' not in sample:
                    return False, f"Sample {i} missing metadata"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"


class FederatedDataConverter:
    """Convert between RoDLA format and federated format"""
    
    @staticmethod
    def rodla_to_federated(rodla_batch: Dict, client_id: str, 
                          privacy_level: str = 'medium') -> List[Dict]:
        """
        Convert RoDLA batch format to federated sample format
        
        Args:
            rodla_batch: Batch from RoDLA data loader
            client_id: Client identifier
            privacy_level: Privacy level for augmentations
            
        Returns:
            List of federated samples
        """
        samples = []
        
        try:
            # Extract batch components
            images = rodla_batch['img']
            img_metas = rodla_batch['img_metas']
            
            # Handle different batch structures
            if isinstance(rodla_batch['gt_bboxes'], list):
                bboxes_list = rodla_batch['gt_bboxes']
                labels_list = rodla_batch['gt_labels']
            else:
                # Convert tensor to list format
                bboxes_list = [bboxes for bboxes in rodla_batch['gt_bboxes']]
                labels_list = [labels for labels in rodla_batch['gt_labels']]
            
            for i in range(len(images)):
                # Convert tensor to PIL Image
                img_tensor = images[i]
                pil_img = DataUtils.tensor_to_pil(img_tensor)
                
                # Prepare annotations
                bboxes = bboxes_list[i].cpu().numpy().tolist() if hasattr(bboxes_list[i], 'cpu') else bboxes_list[i]
                labels = labels_list[i].cpu().numpy().tolist() if hasattr(labels_list[i], 'cpu') else labels_list[i]
                
                # Get original image info
                img_meta = img_metas[i].data if hasattr(img_metas[i], 'data') else img_metas[i]
                original_size = (img_meta['ori_shape'][1], img_meta['ori_shape'][0])  # (width, height)
                
                annotations = {
                    'bboxes': bboxes,
                    'labels': labels,
                    'image_size': original_size,
                    'original_filename': img_meta.get('filename', 'unknown')
                }
                
                # Create augmentation info (will be filled by augmentation engine)
                augmentation_info = {
                    'original_size': original_size,
                    'applied_transforms': [],
                    'parameters': {}
                }
                
                # Create sample
                sample = {
                    'image_data': DataUtils.encode_image_to_base64(pil_img),
                    'annotations': annotations,
                    'metadata': DataUtils.create_sample_metadata(
                        client_id, privacy_level, augmentation_info, 
                        img_meta.get('filename', 'unknown'))
                }
                
                samples.append(sample)
                
        except Exception as e:
            logger.error(f"Error converting RoDLA to federated format: {e}")
        
        return samples
    
    @staticmethod
    def federated_to_rodla(federated_sample: Dict) -> Dict:
        """
        Convert federated sample to RoDLA training format
        
        Args:
            federated_sample: Federated sample dictionary
            
        Returns:
            RoDLA format sample
        """
        try:
            # Decode image
            image = DataUtils.decode_base64_to_image(federated_sample['image_data'])
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Convert to tensor (normalized)
            img_tensor = DataUtils.pil_to_tensor(image)
            
            # Extract annotations
            annotations = federated_sample['annotations']
            bboxes = torch.tensor(annotations['bboxes'], dtype=torch.float32)
            labels = torch.tensor(annotations['labels'], dtype=torch.int64)
            
            # Create img_meta
            img_meta = {
                'filename': federated_sample['metadata'].get('original_file', 'federated_sample'),
                'ori_shape': (annotations['image_size'][1], annotations['image_size'][0], 3),
                'img_shape': (img_tensor.shape[1], img_tensor.shape[2], 3),
                'scale_factor': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                'flip': False,
                'flip_direction': None,
                'img_norm_cfg': {
                    'mean': [123.675, 116.28, 103.53],
                    'std': [58.395, 57.12, 57.375],
                    'to_rgb': True
                }
            }
            
            return {
                'img': img_tensor,
                'gt_bboxes': bboxes,
                'gt_labels': labels,
                'img_metas': img_meta
            }
            
        except Exception as e:
            logger.error(f"Error converting federated to RoDLA format: {e}")
            # Return empty sample as fallback
            return {
                'img': torch.zeros(3, 800, 1333),
                'gt_bboxes': torch.zeros(0, 4),
                'gt_labels': torch.zeros(0, dtype=torch.int64),
                'img_metas': {}
            }


# Utility functions for easy access
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)