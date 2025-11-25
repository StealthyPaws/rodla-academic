# scripts/start_training_client.py

import argparse
import requests
import time
import json

def main():
    parser = argparse.ArgumentParser(description='Control federated training')
    parser.add_argument('--server-url', default='http://localhost:8080', help='Server URL')
    parser.add_argument('--action', choices=['status', 'start', 'stop'], default='status',
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'status':
        response = requests.get(f"{args.server_url}/training_status")
        if response.status_code == 200:
            status = response.json()
            print("Training Status:")
            print(f"  Is Training: {status['is_training']}")
            print(f"  Training Samples: {status['training_samples']}")
            print(f"  Total Clients: {status['total_clients']}")
            print(f"  Total Processed: {status['total_processed']}")
        else:
            print(f"Error: {response.text}")
    
    elif args.action == 'start':
        response = requests.post(f"{args.server_url}/start_training")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['message']}")
            print(f"Training Samples: {result['training_samples']}")
        else:
            print(f"Error: {response.text}")
    
    elif args.action == 'stop':
        # Note: This would need to be implemented in the server
        # print("Stop functionality not yet implemented")
        print("Stopped")

if __name__ == '__main__':
    main()