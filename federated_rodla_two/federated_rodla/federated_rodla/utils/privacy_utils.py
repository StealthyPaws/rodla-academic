# utils/privacy_utils.py
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PrivacyEngine:
    """
    Implements Differential Privacy mechanisms for Federated Learning.
    Primarily uses Gaussian Mechanism with Gradient Clipping.
    """
    def __init__(self, noise_multiplier: float = 0.0, max_grad_norm: float = 1.0):
        """
        Args:
            noise_multiplier (float): The ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function to which it is added.
            max_grad_norm (float): The maximum L2 norm of the model updates (clipping threshold).
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        if self.noise_multiplier > 0:
            logger.info(f"Privacy Engine Initialized: Noise Multiplier={noise_multiplier}, Max Clip={max_grad_norm}")
        else:
            logger.info("Privacy Engine Initialized: DP Disabled (Noise=0)")

    def clip_and_noise(self, original_state_dict: dict, trained_state_dict: dict) -> dict:
        """
        Calculates the update (delta), clips it, adds noise, and returns the new privatized state dict.
        
        Args:
            original_state_dict (dict): The global model weights before training.
            trained_state_dict (dict): The local model weights after training.
            
        Returns:
            dict: The privatized state dict (weights) to send to the server.
        """
        if self.noise_multiplier <= 0:
            return trained_state_dict

        privatized_state_dict = {}
        
        # Calculate total norm of the update (delta) for global clipping
        total_norm = 0.0
        deltas = {}
        
        for key in trained_state_dict.keys():
            if key in original_state_dict:
                # Calculate delta: W_new - W_old
                # Note: In FL, we often send the delta or the new weights. 
                # Here we process the delta for clipping.
                delta = trained_state_dict[key].float() - original_state_dict[key].float()
                deltas[key] = delta
                total_norm += delta.norm(2).item() ** 2
        
        total_norm = total_norm ** 0.5
        
        # Calculate clipping factor
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        clip_coef = min(1.0, clip_coef)
        
        logger.info(f"Privacy: Total Update Norm={total_norm:.4f}, Clipping Factor={clip_coef:.4f}")

        for key, delta in deltas.items():
            # 1. Clip
            clipped_delta = delta * clip_coef
            
            # 2. Add Noise
            # Noise std_dev = noise_multiplier * max_grad_norm
            noise_std = self.noise_multiplier * self.max_grad_norm
            noise = torch.normal(mean=0.0, std=noise_std, size=clipped_delta.shape, device=clipped_delta.device)
            
            # 3. Reconstruct privatized weight: W_old + (Clipped_Delta + Noise)
            privatized_weight = original_state_dict[key].float() + clipped_delta + noise
            
            # Cast back to original type (e.g., float32)
            privatized_state_dict[key] = privatized_weight.type(trained_state_dict[key].dtype)
            
        return privatized_state_dict
