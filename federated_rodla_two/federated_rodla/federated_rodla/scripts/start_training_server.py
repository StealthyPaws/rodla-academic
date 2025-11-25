# scripts/start_training_server.py

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated.training_server import FederatedTrainingServer

def main():
    parser = argparse.ArgumentParser(description='Federated RoDLA Training Server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--max-clients', type=int, default=10, help='Maximum clients')
    parser.add_argument('--data-path', default='./federated_data', help='Data storage path')
    parser.add_argument('--rodla-config', required=True, 
                       help='Path to RoDLA config file (e.g., configs/publaynet/rodla_internimage_xl_publaynet.py)')
    parser.add_argument('--checkpoint', help='Path to pretrained checkpoint (optional)')
    parser.add_argument('--auto-train', action='store_true', 
                       help='Automatically start training when enough data is collected')
    parser.add_argument('--min-samples', type=int, default=500,
                       help='Minimum samples to start training (if auto-train)')
    
    args = parser.parse_args()
    
    server = FederatedTrainingServer(
        max_clients=args.max_clients,
        storage_path=args.data_path,
        rodla_config_path=args.rodla_config,
        model_checkpoint=args.checkpoint
    )
    
    if args.auto_train:
        # Start auto-training monitor
        import threading
        def auto_train_monitor():
            while True:
                time.sleep(60)  # Check every minute
                if len(server.training_data) >= args.min_samples and not server.is_training:
                    logging.info(f"Auto-starting training with {len(server.training_data)} samples")
                    server._start_rodla_training()
        
        monitor_thread = threading.Thread(target=auto_train_monitor, daemon=True)
        monitor_thread.start()
    
    print(f"Starting Federated Training Server on {args.host}:{args.port}")
    print(f"RoDLA config: {args.rodla_config}")
    if args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
    if args.auto_train:
        print(f"Auto-training enabled (min samples: {args.min_samples})")
    
    server.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()