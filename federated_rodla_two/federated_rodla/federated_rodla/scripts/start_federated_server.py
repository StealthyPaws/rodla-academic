# federated/training_server.py (Now acting as FederatedAggregationServer)

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
import copy
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint # For loading initial model state

# from utils.data_utils import DataUtils, FederatedDataConverter # No longer needed for raw data processing

class FederatedAggregationServer:
    def __init__(self, max_clients=10, 
                 rodla_config_path='configs/publaynet/rodla_internimage_xl_publaynet.py',
                 initial_checkpoint=None,
                 num_rounds=10, 
                 clients_per_round=3):
        self.app = Flask(__name__)
        self.clients = {} # Stores client IDs and their registration info
        self.lock = threading.Lock()

        # Federated Learning state
        self.rodla_config_path = rodla_config_path
        self.global_model = None
        self.model_config = None # MMDetection Config object
        self.current_round = 0
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.round_start_time = 0

        # Buffer for client model updates for the current round
        self.client_updates_buffer = {} # {client_id: model_state_dict}
        self.active_clients_in_round = set()

        # Initialize global model (e.g., from a pre-trained checkpoint or randomly)
        self._initialize_global_model(initial_checkpoint)
        
        self.setup_routes()
        logging.basicConfig(level=logging.INFO)
        logger.info("Federated Aggregation Server initialized.")
        logger.info(f"Total FL Rounds: {self.num_rounds}, Clients per Round: {self.clients_per_round}")
        
        # Start FL round management thread
        self.fl_manager_thread = threading.Thread(target=self._fl_round_manager, daemon=True)
        self.fl_manager_thread.start()

    def _initialize_global_model(self, initial_checkpoint):
        """Builds and initializes the global RoDLA model."""
        self.model_config = Config.fromfile(self.rodla_config_path)
        # Modify model to run on CPU if no GPU available for aggregation/saving
        # Or specify device explicitly if server has GPU
        # self.model_config.model.pretrained = None # Avoid downloading new pretrained
        self.global_model = build_detector(self.model_config.model, train_cfg=None, test_cfg=self.model_config.get('test_cfg'))
        
        if initial_checkpoint:
            load_checkpoint(self.global_model, initial_checkpoint, map_location='cpu')
            logger.info(f"Global model initialized from checkpoint: {initial_checkpoint}")
        else:
            logger.info("Global model initialized with random weights (no checkpoint).")
        
        # Move to CPU for saving and sending
        self.global_model.cpu()
        self.global_model.eval() # Set to eval mode

    def setup_routes(self):
        @self.app.route('/register_client', methods=['POST'])
        def register_client():
            client_info = request.json
            client_id = client_info.get('client_id')
            if not client_id:
                return jsonify({'status': 'error', 'message': 'Client ID required'}), 400

            with self.lock:
                if client_id in self.clients:
                    return jsonify({'status': 'error', 'message': f'Client {client_id} already registered'}), 400
                if len(self.clients) >= self.max_clients:
                    return jsonify({'status': 'error', 'message': 'Max clients reached'}), 403
                
                self.clients[client_id] = client_info
                logger.info(f"Client {client_id} registered. Total clients: {len(self.clients)}")
                return jsonify({'status': 'success', 'client_id': client_id, 'message': 'Registered successfully'})

        @self.app.route('/get_global_model', methods=['GET'])
        def get_global_model():
            with self.lock:
                if self.current_round == 0 and len(self.clients) < 1:
                    return jsonify({'status': 'error', 'message': 'No clients registered, cannot start FL.'}), 503
                
                # Get the state dict and serialize it
                state_dict = self.global_model.state_dict()
                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                model_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

                return jsonify({
                    'status': 'success',
                    'round': self.current_round,
                    'model_config': self.model_config.pretty_text, # Send config for client to build model
                    'model_weights': model_data
                })

        @self.app.route('/submit_model_update', methods=['POST'])
        def submit_model_update():
            update_data = request.json
            client_id = update_data.get('client_id')
            round_num = update_data.get('round')
            model_weights_b64 = update_data.get('model_weights')

            if not all([client_id, round_num, model_weights_b64]):
                return jsonify({'status': 'error', 'message': 'Missing data in update'}), 400

            with self.lock:
                if client_id not in self.clients:
                    return jsonify({'status': 'error', 'message': 'Client not registered'}), 403
                if round_num != self.current_round:
                    logger.warning(f"Client {client_id} submitted update for wrong round. Expected {self.current_round}, got {round_num}")
                    return jsonify({'status': 'error', 'message': f'Wrong round, expected {self.current_round}'}), 400
                if client_id not in self.active_clients_in_round:
                    logger.warning(f"Client {client_id} submitted update but not an active client for round {self.current_round}")
                    return jsonify({'status': 'error', 'message': f'Not an active client for round {self.current_round}'}), 403

                logger.info(f"Received model update from {client_id} for round {round_num}")
                
                # Deserialize weights
                model_weights_bytes = base64.b64decode(model_weights_b64)
                buffer = io.BytesIO(model_weights_bytes)
                client_state_dict = torch.load(buffer, map_location='cpu')
                
                self.client_updates_buffer[client_id] = client_state_dict
                
                # Check if all active clients have submitted
                if len(self.client_updates_buffer) >= len(self.active_clients_in_round):
                    logger.info(f"All {len(self.active_clients_in_round)} active clients submitted updates for round {self.current_round}.")
                    # Trigger aggregation (can be done in a separate thread/manager)
                    threading.Thread(target=self._trigger_aggregation, daemon=True).start()

                return jsonify({'status': 'success', 'message': 'Model update received'})
        
        @self.app.route('/server_status', methods=['GET'])
        def server_status():
            with self.lock:
                return jsonify({
                    'status': 'success',
                    'current_round': self.current_round,
                    'num_rounds': self.num_rounds,
                    'total_clients_registered': len(self.clients),
                    'active_clients_in_round': len(self.active_clients_in_round),
                    'updates_received_this_round': len(self.client_updates_buffer),
                    'is_aggregation_pending': (len(self.client_updates_buffer) >= len(self.active_clients_in_round) and len(self.active_clients_in_round) > 0)
                })

    def _fl_round_manager(self):
        """Manages the overall federated learning rounds."""
        time.sleep(5) # Give time for server to start and clients to register
        while self.current_round < self.num_rounds:
            with self.lock:
                if len(self.clients) < self.clients_per_round:
                    logger.info(f"Waiting for at least {self.clients_per_round} clients to register for FL. Current: {len(self.clients)}")
                    time.sleep(30) # Wait for clients
                    continue

                if self.current_round == 0 or (len(self.client_updates_buffer) >= len(self.active_clients_in_round) and len(self.active_clients_in_round) > 0):
                    # If it's the first round, or previous round completed aggregation
                    self._start_new_round()
                else:
                    logger.info(f"Round {self.current_round} still awaiting updates. Received {len(self.client_updates_buffer)}/{len(self.active_clients_in_round)}")
            time.sleep(60) # Check every minute for round progress

        logger.info("Federated learning training completed all rounds.")

    def _start_new_round(self):
        """Prepares for a new federated learning round."""
        with self.lock:
            self.current_round += 1
            if self.current_round > self.num_rounds:
                logger.info("All federated learning rounds completed.")
                return

            logger.info(f"\n--- Starting Federated Round {self.current_round}/{self.num_rounds} ---")
            self.client_updates_buffer.clear()
            
            # Select active clients for this round
            all_client_ids = list(self.clients.keys())
            if len(all_client_ids) < self.clients_per_round:
                self.active_clients_in_round = set(all_client_ids)
                logger.warning(f"Not enough clients ({len(all_client_ids)}) for target clients_per_round ({self.clients_per_round}). Using all available clients.")
            else:
                self.active_clients_in_round = set(np.random.choice(all_client_ids, self.clients_per_round, replace=False))
            
            self.round_start_time = time.time()
            logger.info(f"Active clients for round {self.current_round}: {list(self.active_clients_in_round)}")
            logger.info("Clients can now fetch the global model for this round.")

    def _trigger_aggregation(self):
        """Wrapper to perform aggregation outside the lock for potentially long operations."""
        with self.lock:
            # Check again inside the lock, just in case state changed
            if len(self.client_updates_buffer) >= len(self.active_clients_in_round) and len(self.active_clients_in_round) > 0:
                logger.info(f"Initiating aggregation for round {self.current_round} with {len(self.client_updates_buffer)} updates.")
                self._perform_aggregation()
            else:
                logger.warning("Aggregation triggered but conditions not met. Skipping.")


    def _perform_aggregation(self):
        """Performs Federated Averaging (FedAvg) on received client updates."""
        if not self.client_updates_buffer:
            logger.warning("No client updates to aggregate.")
            return

        # Initialize aggregated weights
        aggregated_state_dict = {}
        for key in self.global_model.state_dict().keys():
            aggregated_state_dict[key] = torch.zeros_like(self.global_model.state_dict()[key])

        total_samples = 0 # In a real scenario, clients would report their data size
        # For simplicity, we assume equal weighting or don't use it explicitly for now,
        # or clients can send n_samples in their update. Here, we'll just average.

        for client_id, client_state_dict in self.client_updates_buffer.items():
            # Assume equal weighting for simplicity (FedAvg)
            # A more robust implementation would use client_data_size for weighted average
            for key in aggregated_state_dict.keys():
                if key in client_state_dict:
                    aggregated_state_dict[key] += client_state_dict[key]
                else:
                    logger.warning(f"Key {key} not found in client {client_id}'s state_dict.")

        num_updates = len(self.client_updates_buffer)
        if num_updates > 0:
            for key in aggregated_state_dict.keys():
                aggregated_state_dict[key] = aggregated_state_dict[key] / num_updates
        
        self.global_model.load_state_dict(aggregated_state_dict)
        logger.info(f"Global model aggregated for round {self.current_round}. New model weights updated.")
        
        # Save aggregated model
        self._save_global_model()

        # Ready for next round, clear buffer
        self.client_updates_buffer.clear()
        self.active_clients_in_round.clear() # Clear active clients to allow re-selection

    def _save_global_model(self):
        """Saves the current global model."""
        save_dir = './federated_checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'global_model_round_{self.current_round}.pth')
        torch.save(self.global_model.state_dict(), model_path)
        logger.info(f"Global model for round {self.current_round} saved to {model_path}")

    def run(self, host='0.0.0.0', port=8080):
        """Runs the Flask application."""
        logger.info(f"Starting Federated Aggregation Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    # This block will be replaced by start_federated_server.py
    # For testing, you can uncomment and provide paths
    # from mmdet.apis import set_random_seed
    # set_random_seed(0, use_deterministic_init=True)
    server = FederatedAggregationServer(
        rodla_config_path='configs/publaynet/rodla_internimage_xl_publaynet.py',
        initial_checkpoint=None, # './checkpoints/rodla_internimage_xl_publaynet.pth',
        num_rounds=5,
        clients_per_round=2
    )
    server.run()