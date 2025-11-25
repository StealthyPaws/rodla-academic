# federated/augmentation_engine.py

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import random
from typing import Dict, Tuple

class AugmentationEngine:
    def __init__(self, privacy_level: str = 'medium'):
        self.privacy_level = privacy_level
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """Setup augmentation parameters based on privacy level"""
        if self.privacy_level == 'low':
            self.geometric_strength = 0.1
            self.color_strength = 0.1
            self.noise_strength = 0.05
        elif self.privacy_level == 'medium':
            self.geometric_strength = 0.2
            self.color_strength = 0.2
            self.noise_strength = 0.1
        else:  # high
            self.geometric_strength = 0.3
            self.color_strength = 0.3
            self.noise_strength = 0.15
    
    def get_capabilities(self) -> Dict:
        """Get augmentation capabilities for server registration"""
        return {
            'geometric_augmentations': True,
            'color_augmentations': True,
            'noise_augmentations': True,
            'privacy_level': self.privacy_level
        }
    
    def augment_image(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply augmentations to image"""
        original_size = image.size
        aug_info = {
            'original_size': original_size,
            'applied_transforms': [],
            'parameters': {}
        }
        
        # Apply geometric transformations
        image, geometric_info = self.apply_geometric_augmentations(image)
        aug_info['applied_transforms'].extend(geometric_info['transforms'])
        aug_info['parameters'].update(geometric_info['parameters'])
        
        # Apply color transformations
        image, color_info = self.apply_color_augmentations(image)
        aug_info['applied_transforms'].extend(color_info['transforms'])
        aug_info['parameters'].update(color_info['parameters'])
        
        # Apply noise
        image, noise_info = self.apply_noise_augmentations(image)
        aug_info['applied_transforms'].extend(noise_info['transforms'])
        aug_info['parameters'].update(noise_info['parameters'])
        
        aug_info['final_size'] = image.size
        
        return image, aug_info
    
    def apply_geometric_augmentations(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply geometric transformations"""
        info = {'transforms': [], 'parameters': {}}
        img = image
        
        # Random rotation
        if random.random() < 0.7:
            angle = random.uniform(-15 * self.geometric_strength, 15 * self.geometric_strength)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
            info['transforms'].append('rotation')
            info['parameters']['rotation_angle'] = angle
        
        # Random scaling
        if random.random() < 0.6:
            scale = random.uniform(1.0 - 0.2 * self.geometric_strength, 1.0 + 0.2 * self.geometric_strength)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.BILINEAR)
            info['transforms'].append('scaling')
            info['parameters']['scale_factor'] = scale
        
        # Random perspective (simplified)
        if random.random() < 0.4:
            img = self.apply_perspective_distortion(img)
            info['transforms'].append('perspective')
        
        return img, info
    
    def apply_color_augmentations(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply color transformations"""
        info = {'transforms': [], 'parameters': {}}
        img = image
        
        # Brightness
        if random.random() < 0.7:
            factor = random.uniform(1.0 - 0.3 * self.color_strength, 1.0 + 0.3 * self.color_strength)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
            info['transforms'].append('brightness')
            info['parameters']['brightness_factor'] = factor
        
        # Contrast
        if random.random() < 0.6:
            factor = random.uniform(1.0 - 0.3 * self.color_strength, 1.0 + 0.3 * self.color_strength)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
            info['transforms'].append('contrast')
            info['parameters']['contrast_factor'] = factor
        
        # Color balance
        if random.random() < 0.5:
            factor = random.uniform(1.0 - 0.2 * self.color_strength, 1.0 + 0.2 * self.color_strength)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)
            info['transforms'].append('color_balance')
            info['parameters']['color_factor'] = factor
        
        return img, info
    
    def apply_noise_augmentations(self, image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Apply noise and blur augmentations"""
        info = {'transforms': [], 'parameters': {}}
        img = image
        
        # Gaussian blur
        if random.random() < 0.5:
            radius = random.uniform(0.1, 1.0 * self.noise_strength)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
            info['transforms'].append('gaussian_blur')
            info['parameters']['blur_radius'] = radius
        
        # Convert to numpy for more advanced noise
        if random.random() < 0.4:
            img_np = np.array(img)
            
            # Gaussian noise
            noise = np.random.normal(0, 25 * self.noise_strength, img_np.shape).astype(np.uint8)
            img_np = cv2.add(img_np, noise)
            
            img = Image.fromarray(img_np)
            info['transforms'].append('gaussian_noise')
        
        return img, info
    
    def apply_perspective_distortion(self, image: Image.Image) -> Image.Image:
        """Apply simple perspective distortion"""
        width, height = image.size
        
        # Simple skew effect
        if random.choice([True, False]):
            # Horizontal skew
            skew_factor = random.uniform(-0.1 * self.geometric_strength, 0.1 * self.geometric_strength)
            matrix = (1, skew_factor, -skew_factor * height * 0.5, 
                    0, 1, 0)
        else:
            # Vertical skew
            skew_factor = random.uniform(-0.1 * self.geometric_strength, 0.1 * self.geometric_strength)
            matrix = (1, 0, 0,
                    skew_factor, 1, -skew_factor * width * 0.5)
        
        img = image.transform(
            image.size, 
            Image.AFFINE, 
            matrix, 
            resample=Image.BILINEAR
        )
        
        return img