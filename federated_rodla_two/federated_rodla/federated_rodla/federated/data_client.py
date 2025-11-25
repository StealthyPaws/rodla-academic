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

from utils.data_utils import DataUtils, FederatedDataConverter
from augmentation_engine import PubLayNetAugmentationEngine

class FederatedDataClient:
    def __init__(self, client_id: str, server_url: str, data_loader, 
                 perturbation_type: str = 'random', severity_level: int = 2):
        self.client_id = client_id
        self.server_url = server_url
        self.data_loader = data_loader
        self.perturbation_type = perturbation_type
        self.severity_level = severity_level
        self.augmentation_engine = PubLayNetAugmentationEngine(perturbation_type, severity_level)
        self.registered = False
        
        logging.basicConfig(level=logging.INFO)
        
    def register_with_server(self):
        """Register this client with the federated server"""
        try:
            client_info = {
                'data_type': 'PubLayNet',
                'perturbation_type': self.perturbation_type,
                'severity_level': self.severity_level,
                'available_perturbations': self.augmentation_engine.get_available_perturbations(),
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
                    logging.info(f"Perturbation: {self.perturbation_type}, Severity: {self.severity_level}")
                    return True
            
            logging.error(f"Failed to register client: {response.text}")
            return False
            
        except Exception as e:
            logging.error(f"Registration failed: {e}")
            return False
    
    def generate_augmented_samples(self, num_samples: int = 50) -> List[Dict]:
        """Generate augmented samples using PubLayNet-P perturbations"""
        samples = []
        available_perturbations = self.augmentation_engine.get_available_perturbations()
        perturbation_cycle = 0
        
        for i, batch in enumerate(self.data_loader):
            if len(samples) >= num_samples:
                break
                
            try:
                images = batch['img']
                img_metas = batch['img_metas']
                
                for j in range(len(images)):
                    if len(samples) >= num_samples:
                        break
                    
                    # Convert tensor to PIL Image
                    img_tensor = images[j]
                    pil_img = DataUtils.tensor_to_pil(img_tensor)
                    
                    # Apply PubLayNet-P perturbation
                    if self.perturbation_type == 'all':
                        # Cycle through all perturbation types
                        pert_type = available_perturbations[perturbation_cycle % len(available_perturbations)]
                        perturbation_cycle += 1
                    elif self.perturbation_type == 'random':
                        pert_type = 'random'
                    else:
                        pert_type = self.perturbation_type
                    
                    augmented_img, augmentation_info = self.augmentation_engine.augment_image(
                        pil_img, pert_type
                    )
                    
                    # Prepare annotations
                    annotations = self.prepare_annotations(batch, j, augmentation_info)
                    
                    # Create sample
                    sample = self.create_sample(augmented_img, annotations, augmentation_info)
                    samples.append(sample)
                    
            except Exception as e:
                logging.warning(f"Error processing batch {i}: {e}")
                continue
        
        logging.info(f"Generated {len(samples)} augmented samples using {self.perturbation_type}")
        return samples
    
    def prepare_annotations(self, batch: Dict, index: int, aug_info: Dict) -> Dict:
        """Prepare annotations for a sample, adjusting for augmentations"""
        bboxes = batch['gt_bboxes'][index]
        labels = batch['gt_labels'][index]
        
        # Convert tensors to lists
        bboxes_list = bboxes.cpu().numpy().tolist() if hasattr(bboxes, 'cpu') else bboxes
        labels_list = labels.cpu().numpy().tolist() if hasattr(labels, 'cpu') else labels
        
        # Adjust bounding boxes for geometric transformations
        if aug_info['perturbation_type'] in ['rotation', 'keystoning', 'warping', 'scaling']:
            bboxes_list = self.adjust_bboxes_for_augmentation(bboxes_list, aug_info)
        
        annotations = {
            'bboxes': bboxes_list,
            'labels': labels_list,
            'image_size': aug_info['final_size'],
            'original_size': aug_info['original_size'],
            'categories': {
                1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'
            }
        }
        
        return annotations
    
    def adjust_bboxes_for_augmentation(self, bboxes: List, aug_info: Dict) -> List:
        """Adjust bounding boxes for geometric augmentations"""
        try:
            orig_w, orig_h = aug_info['original_size']
            new_w, new_h = aug_info['final_size']
            
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            
            adjusted_bboxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                
                # Apply scaling
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
                
                # For rotation, apply simple adjustment (in practice, use proper rotation matrix)
                if aug_info['perturbation_type'] == 'rotation' and 'rotation_angle' in aug_info.get('parameters', {}):
                    angle = aug_info['parameters']['rotation_angle']
                    if abs(angle) > 5:
                        # Simplified rotation adjustment - for production, use proper affine transformation
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        # This is a simplified version - real implementation would use rotation matrix
                        pass
                
                adjusted_bboxes.append([x1, y1, x2, y2])
            
            return adjusted_bboxes
            
        except Exception as e:
            logging.warning(f"Error adjusting bboxes: {e}")
            return bboxes
    
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
                'perturbation_type': aug_info['perturbation_type'],
                'severity_level': aug_info['severity_level'],
                'augmentation_info': aug_info,
                'timestamp': time.time(),
                'dataset': 'PubLayNet'
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
                    'samples': samples,
                    'perturbation_type': self.perturbation_type,
                    'severity_level': self.severity_level
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    logging.info(f"Successfully submitted {result['received']} samples "
                                f"(Perturbation: {self.perturbation_type}, Severity: {self.severity_level})")
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
        
        logging.info(f"Starting continuous data generation")
        logging.info(f"Batch size: {samples_per_batch}, Interval: {interval}s")
        logging.info(f"Perturbation: {self.perturbation_type}, Severity: {self.severity_level}")
        
        batch_count = 0
        while True:
            try:
                samples = self.generate_augmented_samples(samples_per_batch)
                if samples:
                    success = self.submit_augmented_data(samples)
                    batch_count += 1
                    
                    if success:
                        logging.info(f"Batch {batch_count} submitted successfully")
                    else:
                        logging.warning(f"Batch {batch_count} failed, will retry after interval")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logging.info("Data generation stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in data generation loop: {e}")
                time.sleep(interval)

# import requests
# import base64
# import io
# import numpy as np
# import torch
# from PIL import Image
# import json
# import time
# import logging
# from typing import List, Dict, Optional
# import os
# # Uses DataUtils.tensor_to_numpy() and DataUtils.create_sample()
# from utils.data_utils import DataUtils, FederatedDataConverter
# from augmentation_engine import PubLayNetAugmentationEngine  # CHANGED

# class FederatedDataClient:
#     def __init__(self, client_id: str, server_url: str, data_loader, 
#                  perturbation_type: str = 'random', severity_level: int = 2):  # CHANGED
#         self.client_id = client_id
#         self.server_url = server_url
#         self.data_loader = data_loader
#         self.perturbation_type = perturbation_type
#         self.severity_level = severity_level
#         self.augmentation_engine = PubLayNetAugmentationEngine(perturbation_type, severity_level)  # CHANGED
#         self.registered = False
        
#         logging.basicConfig(level=logging.INFO)
        
#     def register_with_server(self):
#         """Register this client with the federated server"""
#         try:
#             client_info = {
#                 'data_type': 'M6Doc',
#                 'privacy_level': self.privacy_level,
#                 'augmentation_capabilities': self.augmentation_engine.get_capabilities(),
#                 'timestamp': time.time()
#             }
            
#             response = requests.post(
#                 f"{self.server_url}/register_client",
#                 json={
#                     'client_id': self.client_id,
#                     'client_info': client_info
#                 },
#                 timeout=10
#             )
            
#             if response.status_code == 200:
#                 data = response.json()
#                 if data['status'] == 'success':
#                     self.registered = True
#                     logging.info(f"Client {self.client_id} successfully registered")
#                     return True
            
#             logging.error(f"Failed to register client: {response.text}")
#             return False
            
#         except Exception as e:
#             logging.error(f"Registration failed: {e}")
#             return False
    
#     def generate_augmented_samples(self, num_samples: int = 50) -> List[Dict]:
#         """Generate augmented samples using PubLayNet-P perturbations"""
#         samples = []
#         available_perturbations = self.augmentation_engine.get_available_perturbations()
        
#         for i, batch in enumerate(self.data_loader):
#             if len(samples) >= num_samples:
#                 break
                
#             try:
#                 images = batch['img']
#                 img_metas = batch['img_metas']
                
#                 for j in range(len(images)):
#                     if len(samples) >= num_samples:
#                         break
                    
#                     # Convert tensor to PIL Image
#                     img_tensor = images[j]
#                     img_np = self.tensor_to_numpy(img_tensor)
#                     pil_img = Image.fromarray(img_np)
                    
#                     # Apply PubLayNet-P perturbation (CHANGED)
#                     if self.perturbation_type == 'all':
#                         # Cycle through all perturbation types
#                         pert_type = available_perturbations[i % len(available_perturbations)]
#                     else:
#                         pert_type = self.perturbation_type
                    
#                     augmented_img, augmentation_info = self.augmentation_engine.augment_image(
#                         pil_img, pert_type
#                     )
                    
#                     # Prepare annotations
#                     annotations = self.prepare_annotations(batch, j, augmentation_info)
                    
#                     # Create sample
#                     sample = self.create_sample(augmented_img, annotations, augmentation_info)
#                     samples.append(sample)
                    
#             except Exception as e:
#                 logging.warning(f"Error processing batch {i}: {e}")
#                 continue
        
#         logging.info(f"Generated {len(samples)} augmented samples using {self.perturbation_type}")
#         return samples
    
#     def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
#         """Convert torch tensor to numpy array for image"""
#         # Denormalize and convert
#         img_np = tensor.cpu().numpy().transpose(1, 2, 0)
#         img_np = (img_np * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]).astype(np.uint8)
#         return img_np
    
#     def prepare_annotations(self, batch: Dict, index: int, aug_info: Dict) -> Dict:
#         """Prepare annotations for a sample, adjusting for augmentations"""
#         bboxes = batch['gt_bboxes'][index].cpu().numpy() if hasattr(batch['gt_bboxes'][index], 'cpu') else batch['gt_bboxes'][index]
#         labels = batch['gt_labels'][index].cpu().numpy() if hasattr(batch['gt_labels'][index], 'cpu') else batch['gt_labels'][index]
        
#         # Adjust bounding boxes for geometric transformations
#         if 'geometric' in aug_info['applied_transforms']:
#             bboxes = self.adjust_bboxes_for_augmentation(bboxes, aug_info)
        
#         annotations = {
#             'bboxes': bboxes.tolist(),
#             'labels': labels.tolist(),
#             'image_size': aug_info['final_size'],
#             'original_size': aug_info['original_size']
#         }
        
#         return annotations
    
#     def adjust_bboxes_for_augmentation(self, bboxes: np.ndarray, aug_info: Dict) -> np.ndarray:
#         """Adjust bounding boxes for geometric augmentations"""
#         # Simplified bbox adjustment
#         # In practice, you'd use the exact transformation matrices
#         scale_x = aug_info['final_size'][0] / aug_info['original_size'][0]
#         scale_y = aug_info['final_size'][1] / aug_info['original_size'][1]
        
#         adjusted_bboxes = bboxes.copy()
#         adjusted_bboxes[:, 0] *= scale_x  # x1
#         adjusted_bboxes[:, 1] *= scale_y  # y1
#         adjusted_bboxes[:, 2] *= scale_x  # x2
#         adjusted_bboxes[:, 3] *= scale_y  # y2
        
#         return adjusted_bboxes
    
#     def create_sample(self, image: Image.Image, annotations: Dict, aug_info: Dict) -> Dict:
#         """Create a sample for sending to server"""
#         # Convert image to base64
#         buffered = io.BytesIO()
#         image.save(buffered, format="JPEG", quality=85)
#         img_str = base64.b64encode(buffered.getvalue()).decode()
        
#         sample = {
#             'image_data': img_str,
#             'annotations': annotations,
#             'metadata': {
#                 'client_id': self.client_id,
#                 'augmentation_info': aug_info,
#                 'timestamp': time.time(),
#                 'privacy_level': self.privacy_level
#             }
#         }
        
#         return sample
    
#     def submit_augmented_data(self, samples: List[Dict]) -> bool:
#         """Submit augmented samples to the server"""
#         if not self.registered:
#             logging.error("Client not registered with server")
#             return False
        
#         try:
#             response = requests.post(
#                 f"{self.server_url}/submit_augmented_data",
#                 json={
#                     'client_id': self.client_id,
#                     'samples': samples
#                 },
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 if result['status'] == 'success':
#                     logging.info(f"Successfully submitted {result['received']} samples")
#                     return True
            
#             logging.error(f"Submission failed: {response.text}")
#             return False
            
#         except Exception as e:
#             logging.error(f"Error submitting data: {e}")
#             return False
    
#     def run_data_generation(self, samples_per_batch: int = 50, interval: int = 300):
#         """Continuously generate and submit augmented data"""
#         if not self.register_with_server():
#             return False
        
#         logging.info(f"Starting continuous data generation (batch: {samples_per_batch}, interval: {interval}s)")
        
#         while True:
#             try:
#                 samples = self.generate_augmented_samples(samples_per_batch)
#                 if samples:
#                     success = self.submit_augmented_data(samples)
#                     if not success:
#                         logging.warning("Failed to submit batch, will retry after interval")
                
#                 time.sleep(interval)
                
#             except KeyboardInterrupt:
#                 logging.info("Data generation stopped by user")
#                 break
#             except Exception as e:
#                 logging.error(f"Error in data generation loop: {e}")
#                 time.sleep(interval)  # Wait before retrying