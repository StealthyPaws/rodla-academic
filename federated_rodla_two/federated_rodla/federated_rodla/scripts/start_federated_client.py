# scripts/start_federated_client.py

import argparse
import sys
import os
import logging
import time
import requests
import json
import base64
import io
import torch
import numpy as np
import shutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmcv import Config
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.apis import train_detector, set_random_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedTrainingClient:
    def __init__(self, client_id, server_url, config_path, data_root, annotation_file, 
                 local_epochs=1, local_lr=0.0001, device='cuda:0', work_dir=None):
        self.client_id = client_id
        self.server_url = server_url
        self.config_path = config_path
        self.data_root = data_root
        self.annotation_file = annotation_file
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.device = device
        self.work_dir = work_dir or f'./work_dirs/client_{client_id}'
        
        self.model = None
        self.cfg = None
        
        # Create work dir
        os.makedirs(self.work_dir, exist_ok=True)

    def register(self):
        """Register with the server"""
        try:
            response = requests.post(
                f"{self.server_url}/register_client",
                json={'client_id': self.client_id}
            )
            if response.status_code == 200:
                logger.info(f"Registered with server: {response.json()}")
                return True
            else:
                logger.error(f"Registration failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Connection error during registration: {e}")
            return False

    def fetch_global_model(self):
        """Fetch global model weights and config from server"""
        try:
            response = requests.get(f"{self.server_url}/get_global_model")
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    return data
                else:
                    logger.warning(f"Server returned error: {data.get('message')}")
                    return None
            elif response.status_code == 503:
                logger.info("Server not ready (waiting for clients/round start).")
                return None
            else:
                logger.error(f"Error fetching model: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Connection error fetching model: {e}")
            return None

    def _train_local_model(self, global_weights_b64, round_num):
        """
        Train the model locally on client data.
        This is the core Federated Learning client loop.
        """
        logger.info(f"Starting local training for round {round_num}...")
        
        # 1. Load Config
        cfg = Config.fromfile(self.config_path)
        
        # 2. Update Config for Local Training
        cfg.data.samples_per_gpu = 2 # Adjust based on memory
        cfg.data.workers_per_gpu = 2
        
        # Point to local data
        cfg.data.train.ann_file = self.annotation_file
        cfg.data.train.img_prefix = self.data_root
        # We don't strictly need val/test for local training loop, but MMDetection might expect them
        cfg.data.val.ann_file = self.annotation_file
        cfg.data.val.img_prefix = self.data_root
        cfg.data.test.ann_file = self.annotation_file
        cfg.data.test.img_prefix = self.data_root

        # Set work directory
        cfg.work_dir = os.path.join(self.work_dir, f'round_{round_num}')
        os.makedirs(cfg.work_dir, exist_ok=True)
        
        # Set epochs
        cfg.runner.max_epochs = self.local_epochs
        
        # Set learning rate (optional: adjust based on round or fixed)
        if cfg.optimizer.get('lr'):
            cfg.optimizer['lr'] = self.local_lr
            
        # Disable some hooks that might interfere or be unnecessary for short local training
        cfg.checkpoint_config = dict(interval=self.local_epochs) # Save only at end
        cfg.log_config.interval = 10
        
        # 3. Build Model
        model = build_detector(cfg.model)
        
        # 4. Load Global Weights
        weights_bytes = base64.b64decode(global_weights_b64)
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        
        # Move to device
        model.to(self.device)
        
        # 5. Build Dataset
        datasets = [build_dataset(cfg.data.train)]
        
        # 6. Run Training
        # Note: train_detector handles the loop, optimizer, etc.
        # We set validate=False to speed up local training
        train_detector(
            model,
            datasets,
            cfg,
            distributed=False,
            validate=False,
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            meta={'exp_name': f'client_{self.client_id}_round_{round_num}'}
        )
        
        logger.info(f"Local training for round {round_num} completed.")
        
        # 7. Return updated state dict
        return model.state_dict()

    def submit_update(self, model_state_dict, round_num):
        """Submit updated model weights to server"""
        try:
            # Serialize weights
            buffer = io.BytesIO()
            torch.save(model_state_dict, buffer)
            weights_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            payload = {
                'client_id': self.client_id,
                'round': round_num,
                'model_weights': weights_b64
            }
            
            response = requests.post(f"{self.server_url}/submit_model_update", json=payload)
            
            if response.status_code == 200:
                logger.info(f"Successfully submitted update for round {round_num}")
                return True
            else:
                logger.error(f"Failed to submit update: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting update: {e}")
            return False

    def run(self):
        """Main client loop"""
        if not self.register():
            return

        last_round_processed = -1
        
        while True:
            try:
                # 1. Check server status / Get global model
                global_model_data = self.fetch_global_model()
                
                if global_model_data:
                    current_round = global_model_data['round']
                    
                    if current_round > last_round_processed:
                        logger.info(f"New round {current_round} detected! (Last processed: {last_round_processed})")
                        
                        # 2. Train locally
                        updated_state_dict = self._train_local_model(
                            global_model_data['model_weights'], 
                            current_round
                        )
                        
                        # 3. Submit update
                        if self.submit_update(updated_state_dict, current_round):
                            last_round_processed = current_round
                            logger.info(f"Round {current_round} complete. Waiting for next round...")
                        else:
                            logger.warning(f"Round {current_round} failed submission. Will retry...")
                            # Don't update last_round_processed so we retry? 
                            # Or maybe wait a bit. For now, let's just wait and retry fetch.
                    else:
                        # Still waiting for next round
                        pass
                
                time.sleep(10) # Poll every 10 seconds
                
            except KeyboardInterrupt:
                logger.info("Client stopping...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in client loop: {e}")
                time.sleep(10)

def main():
    parser = argparse.ArgumentParser(description='Federated Training Client')
    parser.add_argument('--client-id', required=True, help='Unique client identifier')
    parser.add_argument('--server-url', default='http://localhost:8080', help='Server URL')
    parser.add_argument('--config', required=True, help='Path to RoDLA config')
    parser.add_argument('--data-root', required=True, help='Path to local images')
    parser.add_argument('--annotation-file', required=True, help='Path to local annotations')
    parser.add_argument('--local-epochs', type=int, default=1, help='Epochs per round')
    parser.add_argument('--local-lr', type=float, default=0.0001, help='Local learning rate')
    parser.add_argument('--device', default='cuda:0', help='Device (cuda:0 or cpu)')
    
    args = parser.parse_args()
    
    client = FederatedTrainingClient(
        client_id=args.client_id,
        server_url=args.server_url,
        config_path=args.config,
        data_root=args.data_root,
        annotation_file=args.annotation_file,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        device=args.device
    )
    
    client.run()

if __name__ == '__main__':
    main()
