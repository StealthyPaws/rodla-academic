# scripts/start_data_client.py

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.data_client import FederatedDataClient
import torch
from torch.utils.data import DataLoader

def create_dummy_dataloader():
    """Create a dummy dataloader for testing - replace with actual M6Doc dataloader"""
    # This is a placeholder - you'll replace this with your actual M6Doc data loader
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=1000):
            self.size = size
            
        def __len__(self):
            return self.size
            
        def __getitem__(self, idx):
            # Return dummy data in RoDLA format
            return {
                'img': torch.randn(3, 800, 1333),
                'gt_bboxes': [torch.tensor([[100, 100, 200, 200]])],
                'gt_labels': [torch.tensor([1])],
                'img_metas': [{'filename': f'dummy_{idx}.jpg', 'ori_shape': (800, 1333, 3)}]
            }
    
    dataset = DummyDataset(1000)
    return DataLoader(dataset, batch_size=4, shuffle=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client-id', required=True, help='Client ID')
    parser.add_argument('--server-url', default='http://localhost:8080', help='Server URL')
    parser.add_argument('--privacy-level', choices=['low', 'medium', 'high'], default='medium')
    parser.add_argument('--samples-per-batch', type=int, default=50)
    parser.add_argument('--interval', type=int, default=300, help='Seconds between batches')
    
    args = parser.parse_args()
    
    # Create data loader (replace with your actual M6Doc data loader)
    data_loader = create_dummy_dataloader()
    
    # Create federated client
    client = FederatedDataClient(
        client_id=args.client_id,
        server_url=args.server_url,
        data_loader=data_loader,
        privacy_level=args.privacy_level
    )
    
    # Start continuous data generation
    client.run_data_generation(
        samples_per_batch=args.samples_per_batch,
        interval=args.interval
    )

if __name__ == '__main__':
    main()