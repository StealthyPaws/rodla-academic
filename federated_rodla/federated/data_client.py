# federated/data_client.py

import requests
import base64
import io
import numpy as np
import torch
from PIL import Image
import json
import time
import logging
from typing import List, Dict, Optional
import os
# Uses DataUtils.tensor_to_numpy() and DataUtils.create_sample()
from utils.data_utils import DataUtils, FederatedDataConverter
from augmentation_engine import AugmentationEngine

class FederatedDataClient:
    def __init__(self, client_id: str, server_url: str, data_loader, privacy_level: str = 'medium'):
        self.client_id = client_id
        self.server_url = server_url
        self.data_loader = data_loader
        self.privacy_level = privacy_level
        self.augmentation_engine = AugmentationEngine(privacy_level)
        self.registered = False
        
        logging.basicConfig(level=logging.INFO)
        
    def register_with_server(self):
        """Register this client with the federated server"""
        try:
            client_info = {
                'data_type': 'M6Doc',
                'privacy_level': self.privacy_level,
                'augmentation_capabilities': self.augmentation_engine.get_capabilities(),
                'timestamp': time.time()
            }
            
            response = requests.post(
                f"{self.server_url}/register_client",
                json={
                    'client_id': self.client_id,
                    'client_info': client_info
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    self.registered = True
                    logging.info(f"Client {self.client_id} successfully registered")
                    return True
            
            logging.error(f"Failed to register client: {response.text}")
            return False
            
        except Exception as e:
            logging.error(f"Registration failed: {e}")
            return False
    
    def generate_augmented_samples(self, num_samples: int = 50) -> List[Dict]:
        """Generate augmented samples from local data"""
        samples = []
        
        for i, batch in enumerate(self.data_loader):
            if len(samples) >= num_samples:
                break
                
            try:
                # Assume batch structure: {'img': tensor, 'gt_bboxes': list, 'gt_labels': list, 'img_metas': list}
                images = batch['img']
                img_metas = batch['img_metas']
                
                for j in range(len(images)):
                    if len(samples) >= num_samples:
                        break
                    
                    # Convert tensor to PIL Image
                    img_tensor = images[j]
                    img_np = self.tensor_to_numpy(img_tensor)
                    pil_img = Image.fromarray(img_np)
                    
                    # Apply augmentations
                    augmented_img, augmentation_info = self.augmentation_engine.augment_image(pil_img)
                    
                    # Prepare annotations
                    annotations = self.prepare_annotations(batch, j, augmentation_info)
                    
                    # Create sample
                    sample = self.create_sample(augmented_img, annotations, augmentation_info)
                    samples.append(sample)
                    
            except Exception as e:
                logging.warning(f"Error processing batch {i}: {e}")
                continue
        
        logging.info(f"Generated {len(samples)} augmented samples")
        return samples
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array for image"""
        # Denormalize and convert
        img_np = tensor.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]).astype(np.uint8)
        return img_np
    
    def prepare_annotations(self, batch: Dict, index: int, aug_info: Dict) -> Dict:
        """Prepare annotations for a sample, adjusting for augmentations"""
        bboxes = batch['gt_bboxes'][index].cpu().numpy() if hasattr(batch['gt_bboxes'][index], 'cpu') else batch['gt_bboxes'][index]
        labels = batch['gt_labels'][index].cpu().numpy() if hasattr(batch['gt_labels'][index], 'cpu') else batch['gt_labels'][index]
        
        # Adjust bounding boxes for geometric transformations
        if 'geometric' in aug_info['applied_transforms']:
            bboxes = self.adjust_bboxes_for_augmentation(bboxes, aug_info)
        
        annotations = {
            'bboxes': bboxes.tolist(),
            'labels': labels.tolist(),
            'image_size': aug_info['final_size'],
            'original_size': aug_info['original_size']
        }
        
        return annotations
    
    def adjust_bboxes_for_augmentation(self, bboxes: np.ndarray, aug_info: Dict) -> np.ndarray:
        """Adjust bounding boxes for geometric augmentations"""
        # Simplified bbox adjustment
        # In practice, you'd use the exact transformation matrices
        scale_x = aug_info['final_size'][0] / aug_info['original_size'][0]
        scale_y = aug_info['final_size'][1] / aug_info['original_size'][1]
        
        adjusted_bboxes = bboxes.copy()
        adjusted_bboxes[:, 0] *= scale_x  # x1
        adjusted_bboxes[:, 1] *= scale_y  # y1
        adjusted_bboxes[:, 2] *= scale_x  # x2
        adjusted_bboxes[:, 3] *= scale_y  # y2
        
        return adjusted_bboxes
    
    def create_sample(self, image: Image.Image, annotations: Dict, aug_info: Dict) -> Dict:
        """Create a sample for sending to server"""
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        sample = {
            'image_data': img_str,
            'annotations': annotations,
            'metadata': {
                'client_id': self.client_id,
                'augmentation_info': aug_info,
                'timestamp': time.time(),
                'privacy_level': self.privacy_level
            }
        }
        
        return sample
    
    def submit_augmented_data(self, samples: List[Dict]) -> bool:
        """Submit augmented samples to the server"""
        if not self.registered:
            logging.error("Client not registered with server")
            return False
        
        try:
            response = requests.post(
                f"{self.server_url}/submit_augmented_data",
                json={
                    'client_id': self.client_id,
                    'samples': samples
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    logging.info(f"Successfully submitted {result['received']} samples")
                    return True
            
            logging.error(f"Submission failed: {response.text}")
            return False
            
        except Exception as e:
            logging.error(f"Error submitting data: {e}")
            return False
    
    def run_data_generation(self, samples_per_batch: int = 50, interval: int = 300):
        """Continuously generate and submit augmented data"""
        if not self.register_with_server():
            return False
        
        logging.info(f"Starting continuous data generation (batch: {samples_per_batch}, interval: {interval}s)")
        
        while True:
            try:
                samples = self.generate_augmented_samples(samples_per_batch)
                if samples:
                    success = self.submit_augmented_data(samples)
                    if not success:
                        logging.warning("Failed to submit batch, will retry after interval")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logging.info("Data generation stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in data generation loop: {e}")
                time.sleep(interval)  # Wait before retrying