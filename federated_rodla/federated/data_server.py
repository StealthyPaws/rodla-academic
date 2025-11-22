# federated/data_server.py

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
# Uses DataUtils.process_sample() for validation
from utils.data_utils import DataUtils

class FederatedDataServer:
    def __init__(self, max_clients=10, storage_path='./federated_data'):
        self.app = Flask(__name__)
        self.clients = {}
        self.data_queue = deque()
        self.lock = threading.Lock()
        self.storage_path = storage_path
        self.max_clients = max_clients
        self.processed_samples = 0
        
        # Create storage directory
        import os
        os.makedirs(storage_path, exist_ok=True)
        
        self.setup_routes()
        logging.basicConfig(level=logging.INFO)
        
    def setup_routes(self):
        @self.app.route('/register_client', methods=['POST'])
        def register_client():
            data = request.json
            client_id = data['client_id']
            client_info = data['client_info']
            
            with self.lock:
                if len(self.clients) >= self.max_clients:
                    return jsonify({'status': 'error', 'message': 'Server full'})
                
                self.clients[client_id] = {
                    'info': client_info,
                    'last_seen': time.time(),
                    'samples_sent': 0
                }
            
            logging.info(f"Client {client_id} registered")
            return jsonify({'status': 'success', 'client_id': client_id})
        
        @self.app.route('/submit_augmented_data', methods=['POST'])
        def submit_augmented_data():
            try:
                data = request.json
                client_id = data['client_id']
                samples = data['samples']
                
                # Validate client
                with self.lock:
                    if client_id not in self.clients:
                        return jsonify({'status': 'error', 'message': 'Client not registered'})
                
                # Process each sample
                processed_samples = []
                for sample in samples:
                    processed_sample = self.process_sample(sample)
                    if processed_sample:
                        processed_samples.append(processed_sample)
                
                # Add to training queue
                with self.lock:
                    self.data_queue.extend(processed_samples)
                    self.clients[client_id]['samples_sent'] += len(processed_samples)
                    self.processed_samples += len(processed_samples)
                
                logging.info(f"Received {len(processed_samples)} samples from {client_id}")
                return jsonify({
                    'status': 'success', 
                    'received': len(processed_samples),
                    'total_processed': self.processed_samples
                })
                
            except Exception as e:
                logging.error(f"Error processing data: {e}")
                return jsonify({'status': 'error', 'message': str(e)})
        
        @self.app.route('/get_training_batch', methods=['GET'])
        def get_training_batch():
            batch_size = request.args.get('batch_size', 32, type=int)
            
            with self.lock:
                if len(self.data_queue) < batch_size:
                    return jsonify({'status': 'insufficient_data', 'available': len(self.data_queue)})
                
                batch = []
                for _ in range(batch_size):
                    if self.data_queue:
                        batch.append(self.data_queue.popleft())
                
            logging.info(f"Sending batch of {len(batch)} samples for training")
            return jsonify({
                'status': 'success',
                'batch': batch,
                'batch_size': len(batch)
            })
        
        @self.app.route('/server_stats', methods=['GET'])
        def server_stats():
            with self.lock:
                stats = {
                    'total_clients': len(self.clients),
                    'samples_in_queue': len(self.data_queue),
                    'total_processed_samples': self.processed_samples,
                    'clients': {
                        client_id: {
                            'samples_sent': info['samples_sent'],
                            'last_seen': info['last_seen']
                        }
                        for client_id, info in self.clients.items()
                    }
                }
            return jsonify(stats)
    
    def process_sample(self, sample):
        """Process and validate a sample from client"""
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
                
            # Add metadata
            sample['received_time'] = time.time()
            sample['server_processed'] = True
            
            return sample
            
        except Exception as e:
            logging.warning(f"Failed to process sample: {e}")
            return None
    
    def run(self, host='0.0.0.0', port=8080):
        """Start the federated data server"""
        logging.info(f"Starting Federated Data Server on {host}:{port}")
        self.app.run(host=host, port=port, threaded=True)

if __name__ == '__main__':
    server = FederatedDataServer(max_clients=10)
    server.run()