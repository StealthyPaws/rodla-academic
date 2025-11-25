# scripts/start_data_server.py

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.data_server import FederatedDataServer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--max-clients', type=int, default=10, help='Maximum clients')
    parser.add_argument('--data-path', default='./federated_data', help='Data storage path')
    
    args = parser.parse_args()
    
    server = FederatedDataServer(
        max_clients=args.max_clients,
        storage_path=args.data_path
    )
    
    server.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()