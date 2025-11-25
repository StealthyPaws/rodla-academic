# scripts/start_data_client.py

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.data_client import FederatedDataClient
import torch
from torch.utils.data import DataLoader, Dataset
from mmdet.datasets import build_dataset, build_dataloader
from mmcv import Config
import json
from PIL import Image
import numpy as np

class PubLayNetDataset(Dataset):
    """Actual PubLayNet dataset loader for federated client"""
    
    def __init__(self, data_root, annotation_file, split='train', max_samples=1000):
        self.data_root = data_root
        self.split = split
        self.max_samples = max_samples
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter images for the specified split
        self.images = [img for img in self.annotations['images'] 
                      if img['file_name'].startswith(split)]
        
        # Limit samples if specified
        if max_samples:
            self.images = self.images[:max_samples]
        
        # Create image id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # PubLayNet categories
        self.categories = {
            1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'
        }
        
        print(f"Loaded {len(self.images)} images from PubLayNet {split} set")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_info = self.images[idx]
            img_path = os.path.join(self.data_root, img_info['file_name'])
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            img_width, img_height = image.size
            
            # Get annotations for this image
            anns = self.img_to_anns.get(img_info['id'], [])
            
            bboxes = []
            labels = []
            
            for ann in anns:
                # Convert COCO bbox format [x, y, width, height] to [x1, y1, x2, y2]
                x, y, w, h = ann['bbox']
                bbox = [x, y, x + w, y + h]
                
                # Filter invalid bboxes
                if (bbox[2] - bbox[0] > 1 and bbox[3] - bbox[1] > 1 and
                    bbox[0] >= 0 and bbox[1] >= 0 and 
                    bbox[2] <= img_width and bbox[3] <= img_height):
                    bboxes.append(bbox)
                    labels.append(ann['category_id'])
            
            if len(bboxes) == 0:
                # Return empty annotations if no valid bboxes
                bboxes = [[0, 0, 1, 1]]  # dummy bbox
                labels = [1]  # text category
            
            # Convert to tensors
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            
            # Convert image to tensor (normalized)
            img_tensor = torch.from_numpy(np.array(image).astype(np.float32)).permute(2, 0, 1)
            img_tensor = (img_tensor - torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)) / \
                         torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
            
            # Create img_meta in RoDLA format
            img_meta = {
                'filename': img_info['file_name'],
                'ori_shape': (img_height, img_width, 3),
                'img_shape': (img_height, img_width, 3),
                'pad_shape': (img_height, img_width, 3),
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
                'gt_bboxes': bboxes_tensor,
                'gt_labels': labels_tensor,
                'img_metas': img_meta
            }
            
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            # Return a dummy sample on error
            return self.create_dummy_sample()

    def create_dummy_sample(self):
        """Create a dummy sample when loading fails"""
        return {
            'img': torch.randn(3, 800, 800),
            'gt_bboxes': torch.tensor([[100, 100, 200, 200]]),
            'gt_labels': torch.tensor([1]),
            'img_metas': {
                'filename': 'dummy.jpg',
                'ori_shape': (800, 800, 3),
                'img_shape': (800, 800, 3),
                'scale_factor': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                'flip': False
            }
        }

def create_publaynet_dataloader(data_root='/path/to/publaynet', 
                               annotation_file='/path/to/annotations.json',
                               split='train', 
                               batch_size=4, 
                               max_samples=1000):
    """Create actual PubLayNet data loader"""
    
    dataset = PubLayNetDataset(
        data_root=data_root,
        annotation_file=annotation_file,
        split=split,
        max_samples=max_samples
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    return dataloader

def collate_fn(batch):
    """Custom collate function for PubLayNet batches"""
    batch_dict = {}
    
    for key in batch[0].keys():
        if key == 'img':
            batch_dict[key] = torch.stack([item[key] for item in batch])
        elif key in ['gt_bboxes', 'gt_labels']:
            batch_dict[key] = [item[key] for item in batch]
        elif key == 'img_metas':
            batch_dict[key] = [item[key] for item in batch]
    
    return batch_dict

def main():
    parser = argparse.ArgumentParser(description='Federated PubLayNet Client')
    parser.add_argument('--client-id', required=True, help='Client ID')
    parser.add_argument('--server-url', default='http://localhost:8080', help='Server URL')
    parser.add_argument('--perturbation-type', 
                       choices=[
                           'background', 'defocus', 'illumination', 'ink_bleeding', 
                           'ink_holdout', 'keystoning', 'rotation', 'speckle', 
                           'texture', 'vibration', 'warping', 'watermark', 'random', 'all'
                       ],
                       default='random', help='PubLayNet-P perturbation type')
    parser.add_argument('--severity-level', type=int, choices=[1, 2, 3], default=2,
                       help='Perturbation severity level (1-3)')
    parser.add_argument('--samples-per-batch', type=int, default=50,
                       help='Number of augmented samples to generate per batch')
    parser.add_argument('--interval', type=int, default=300, 
                       help='Seconds between batches')
    parser.add_argument('--data-root', required=True,
                       help='Path to PubLayNet dataset root directory')
    parser.add_argument('--annotation-file', required=True,
                       help='Path to PubLayNet annotations JSON file')
    parser.add_argument('--split', choices=['train', 'val'], default='train',
                       help='Dataset split to use')
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='Maximum number of samples to use from dataset')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for data loading')
    
    args = parser.parse_args()
    
    # Create actual PubLayNet data loader
    data_loader = create_publaynet_dataloader(
        data_root=args.data_root,
        annotation_file=args.annotation_file,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Create federated client with PubLayNet-P perturbations
    client = FederatedDataClient(
        client_id=args.client_id,
        server_url=args.server_url,
        data_loader=data_loader,
        perturbation_type=args.perturbation_type,
        severity_level=args.severity_level
    )
    
    print(f"Starting federated client {args.client_id}")
    print(f"Perturbation type: {args.perturbation_type}")
    print(f"Severity level: {args.severity_level}")
    print(f"Data source: {args.data_root}")
    
    client.run_data_generation(
        samples_per_batch=args.samples_per_batch,
        interval=args.interval
    )

if __name__ == '__main__':
    main()