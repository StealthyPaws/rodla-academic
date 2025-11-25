# federated/training_server.py

import flask
from flask import Flask, request, jsonify
import threading
import numpy as np
import json
import base64
import io
from PIL import Image
import cv2
import logging
from collections import defaultdict, deque
import time
import torch
import subprocess
import os
from utils.data_utils import DataUtils, FederatedDataConverter

class FederatedTrainingServer:
    def __init__(self, max_clients=10, storage_path='./federated_data', 
                 rodla_config_path='configs/publaynet/rodla_internimage_xl_publaynet.py',
                 model_checkpoint=None):
        self.app = Flask(__name__)
        self.clients = {}
        self.data_queue = deque()
        self.training_data = []  # Store data for training
        self.lock = threading.Lock()
        self.storage_path = storage_path
        self.max_clients = max_clients
        self.processed_samples = 0
        self.rodla_config_path = rodla_config_path
        self.model_checkpoint = model_checkpoint
        self.is_training = False
        self.training_process = None
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        os.makedirs('./federated_training_data', exist_ok=True)
        
        self.setup_routes()
        logging.basicConfig(level=logging.INFO)
        
        # Start training monitor thread
        self.training_thread = threading.Thread(target=self._training_monitor, daemon=True)
        self.training_thread.start()
    
    def setup_routes(self):
        # ... (keep all existing routes: register_client, submit_augmented_data, etc.)
        
        @self.app.route('/start_training', methods=['POST'])
        def start_training():
            """Start RoDLA training with federated data"""
            with self.lock:
                if self.is_training:
                    return jsonify({'status': 'error', 'message': 'Training already in progress'})
                
                if len(self.training_data) < 100:  # Minimum samples to start training
                    return jsonify({'status': 'error', 'message': f'Insufficient data: {len(self.training_data)} samples'})
            
            # Start training in separate thread
            training_thread = threading.Thread(target=self._start_rodla_training)
            training_thread.start()
            
            return jsonify({
                'status': 'success', 
                'message': 'Training started',
                'training_samples': len(self.training_data)
            })
        
        @self.app.route('/training_status', methods=['GET'])
        def training_status():
            """Get current training status"""
            return jsonify({
                'is_training': self.is_training,
                'training_samples': len(self.training_data),
                'total_clients': len(self.clients),
                'total_processed': self.processed_samples
            })
    
    def process_sample(self, sample):
        """Process and validate a sample from client - UPDATED to store for training"""
        try:
            # Decode image
            if 'image_data' in sample:
                image_data = base64.b64decode(sample['image_data'])
                image = Image.open(io.BytesIO(image_data))
                
                # Convert to numpy array (for validation)
                img_array = np.array(image)
                
                # Basic validation
                if img_array.size == 0:
                    return None
                    
            # Validate annotations
            if 'annotations' not in sample:
                return None
            
            # Store sample for training
            with self.lock:
                self.training_data.append(sample)
                
                # Limit training data size to prevent memory issues
                if len(self.training_data) > 10000:
                    self.training_data = self.training_data[-10000:]
            
            # Add metadata
            sample['received_time'] = time.time()
            sample['server_processed'] = True
            
            return sample
            
        except Exception as e:
            logging.warning(f"Failed to process sample: {e}")
            return None
    
    def _start_rodla_training(self):
        """Start RoDLA training with federated data"""
        try:
            self.is_training = True
            logging.info("Starting RoDLA training with federated data...")
            
            # Convert federated data to RoDLA training format
            training_dataset = self._prepare_training_dataset()
            
            # Save training dataset
            dataset_path = self._save_training_dataset(training_dataset)
            
            # Start RoDLA training process
            self._run_rodla_training(dataset_path)
            
        except Exception as e:
            logging.error(f"Training failed: {e}")
        finally:
            self.is_training = False
    
    def _prepare_training_dataset(self):
        """Convert federated samples to RoDLA training format"""
        training_samples = []
        
        for sample in self.training_data:
            try:
                # Convert federated format to RoDLA format
                rodla_sample = FederatedDataConverter.federated_to_rodla(sample)
                training_samples.append(rodla_sample)
            except Exception as e:
                logging.warning(f"Failed to convert sample: {e}")
                continue
        
        logging.info(f"Prepared {len(training_samples)} samples for training")
        return training_samples
    
    def _save_training_dataset(self, training_dataset):
        """Save training dataset to disk in COCO format"""
        dataset_dir = './federated_training_data'
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save images
        images_dir = os.path.join(dataset_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        annotations = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': 1, 'name': 'text'},
                {'id': 2, 'name': 'title'}, 
                {'id': 3, 'name': 'list'},
                {'id': 4, 'name': 'table'},
                {'id': 5, 'name': 'figure'}
            ]
        }
        
        annotation_id = 1
        
        for i, sample in enumerate(training_dataset):
            # Save image
            img_tensor = sample['img']
            img_np = (img_tensor * torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1) + 
                     torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1))
            img_np = img_np.numpy().transpose(1, 2, 0).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            img_filename = f"federated_{i:06d}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            img_pil.save(img_path)
            
            # Add image info
            img_info = {
                'id': i,
                'file_name': img_filename,
                'width': img_np.shape[1],
                'height': img_np.shape[0]
            }
            annotations['images'].append(img_info)
            
            # Add annotations
            bboxes = sample['gt_bboxes']
            labels = sample['gt_labels']
            
            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox.tolist()
                annotation = {
                    'id': annotation_id,
                    'image_id': i,
                    'category_id': label.item(),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                    'area': (x2 - x1) * (y2 - y1),
                    'iscrowd': 0
                }
                annotations['annotations'].append(annotation)
                annotation_id += 1
        
        # Save annotations
        annotations_path = os.path.join(dataset_dir, 'annotations.json')
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logging.info(f"Saved training dataset: {len(annotations['images'])} images, "
                    f"{len(annotations['annotations'])} annotations")
        
        return dataset_dir
    
    def _run_rodla_training(self, dataset_path):
        """Run actual RoDLA training using the provided dataset"""
        try:
            # Create modified config for federated training
            config_content = self._create_federated_config(dataset_path)
            config_path = './configs/federated/rodla_federated_publaynet.py'
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Run RoDLA training command (from their GitHub)
            cmd = [
                'python', 'model/train.py',
                config_path,
                '--work-dir', './work_dirs/federated_rodla',
                '--auto-resume'
            ]
            
            if self.model_checkpoint:
                cmd.extend(['--resume-from', self.model_checkpoint])
            
            logging.info(f"Starting RoDLA training: {' '.join(cmd)}")
            
            # Run training process
            self.training_process = subprocess.Popen(
                cmd,
                cwd='.',  # Assuming we're in RoDLA root directory
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Log training output
            for line in iter(self.training_process.stdout.readline, ''):
                logging.info(f"TRAINING: {line.strip()}")
            
            self.training_process.wait()
            
            if self.training_process.returncode == 0:
                logging.info("RoDLA training completed successfully!")
            else:
                logging.error(f"RoDLA training failed with code {self.training_process.returncode}")
                
        except Exception as e:
            logging.error(f"Error running RoDLA training: {e}")
    
    def _create_federated_config(self, dataset_path):
        """Create modified RoDLA config for federated training"""
        base_config = f'''
_base_ = '../publaynet/rodla_internimage_xl_publaynet.py'

# Federated training settings
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file='{dataset_path}/annotations.json',
        img_prefix='{dataset_path}/images/',
    ),
    val=dict(
        ann_file='{dataset_path}/annotations.json',  # Using same data for val during federated training
        img_prefix='{dataset_path}/images/',
    )
)

# Training schedule for federated learning
runner = dict(max_epochs=12)  # Shorter epochs for frequent updates
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)

# Logging
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)

# Evaluation
evaluation = dict(interval=1, metric=['bbox', 'segm'])
checkpoint_config = dict(interval=1)
'''
        return base_config
    
    def _training_monitor(self):
        """Monitor training process"""
        while True:
            if self.training_process and self.training_process.poll() is not None:
                self.is_training = False
                self.training_process = None
                logging.info("Training process finished")
            
            time.sleep(10)

if __name__ == '__main__':
    server = FederatedTrainingServer(
        rodla_config_path='configs/publaynet/rodla_internimage_xl_publaynet.py',
        model_checkpoint='checkpoints/rodla_internimage_xl_publaynet.pth'  # if available
    )
    server.run()