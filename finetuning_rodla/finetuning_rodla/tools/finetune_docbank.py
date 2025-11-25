#!/usr/bin/env python3
"""
Real RoDLA Fine-tuning on DocBank
Uses actual MMDetection training framework
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_environment():
    """Check if required dependencies are available"""
    try:
        import mmdet
        import mmcv
        print("✓ MMDetection and MMCV are available")
    except ImportError as e:
        print(f" Missing dependencies: {e}")
        print("Please install MMDetection and MMCV first")
        return False
    
    # Check if we're in RoDLA directory
    if not os.path.exists('model') and not os.path.exists('configs'):
        print(" Please run this script from the RoDLA root directory")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/DocBank_coco',
        'work_dirs/rodla_docbank', 
        'checkpoints'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def convert_dataset(docbank_root, output_dir):
    """Convert DocBank to COCO format"""
    print(f"Converting DocBank dataset from {docbank_root} to COCO format...")
    
    cmd = [
        sys.executable, 'tools/convert_docbank_to_coco.py',
        '--docbank-root', docbank_root,
        '--output-dir', output_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Dataset conversion completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Dataset conversion failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def download_pretrained_weights():
    """Download pre-trained weights if not available"""
    checkpoint_path = 'checkpoints/rodla_internimage_xl_publaynet.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"✓ Pre-trained weights found: {checkpoint_path}")
        return True
    
    print(" Pre-trained weights not found.")
    print("Please download RoDLA PubLayNet weights from:")
    print("https://drive.google.com/file/d/1I2CafA-wRKAWCqFgXPgtoVx3OQcRWEjp/view?usp=sharing")
    print(f"And place them at: {checkpoint_path}")
    
    # Alternative: Use ImageNet pre-trained
    imagenet_path = 'checkpoints/internimage_xl_22k_192to384.pth'
    if not os.path.exists(imagenet_path):
        print("\nAlternatively, downloading ImageNet pre-trained weights...")
        os.makedirs('checkpoints', exist_ok=True)
        try:
            import gdown
            url = "https://github.com/OpenGVLab/InternImage/releases/download/cls_model/internimage_xl_22k_192to384.pth"
            gdown.download(url, imagenet_path, quiet=False)
            print("✓ Downloaded ImageNet pre-trained weights")
            
            # Update config to use ImageNet weights
            update_config_for_imagenet()
            return True
        except Exception as e:
            print(f" Failed to download weights: {e}")
            return False
    
    return True

def update_config_for_imagenet():
    """Update config to use ImageNet pre-trained weights"""
    config_path = 'configs/docbank/rodla_internimage_docbank.py'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Update the pretrained path
        content = content.replace(
            "pretrained = 'checkpoints/rodla_internimage_xl_publaynet.pth'",
            "pretrained = 'checkpoints/internimage_xl_22k_192to384.pth'"
        )
        
        with open(config_path, 'w') as f:
            f.write(content)
        
        print("✓ Updated config to use ImageNet pre-trained weights")

def run_training(config_path, work_dir):
    """Run actual MMDetection training"""
    print("Starting RoDLA fine-tuning on DocBank...")
    
    cmd = [
        sys.executable, 'tools/train.py',
        config_path,
        f'--work-dir={work_dir}',
        '--auto-resume',
        '--seed', '42'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the actual training command
        result = subprocess.run(cmd, check=True)
        print("✓ Fine-tuning completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Training failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

def run_evaluation(config_path, checkpoint_path):
    """Run evaluation on test set"""
    print("Running evaluation on DocBank test set...")
    
    cmd = [
        sys.executable, 'tools/test.py',
        config_path,
        checkpoint_path,
        '--eval', 'bbox',
        '--out', f'{os.path.dirname(checkpoint_path)}/results.pkl',
        '--show-dir', f'{os.path.dirname(checkpoint_path)}/visualizations'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Evaluation completed successfully!")
        
        # Print the evaluation results
        if result.stdout:
            print("\nEvaluation Results:")
            print(result.stdout)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f" Evaluation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fine-tune RoDLA on DocBank')
    parser.add_argument('--docbank-root', required=True, 
                       help='Path to DocBank dataset root directory')
    parser.add_argument('--config', default='configs/docbank/rodla_internimage_docbank.py',
                       help='Path to fine-tuning config file')
    parser.add_argument('--work-dir', default='work_dirs/rodla_docbank',
                       help='Work directory for training outputs')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only run evaluation')
    
    args = parser.parse_args()
    
    print("RoDLA DocBank Fine-tuning Pipeline")
    print("=" * 50)
    
    # Step 1: Environment check
    if not check_environment():
        sys.exit(1)
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Convert dataset
    output_dir = 'data/DocBank_coco'
    if not convert_dataset(args.docbank_root, output_dir):
        sys.exit(1)
    
    # Step 4: Download weights
    if not download_pretrained_weights():
        sys.exit(1)
    
    # Step 5: Run training
    if not args.skip_training:
        if not run_training(args.config, args.work_dir):
            sys.exit(1)
    
    # Step 6: Run evaluation
    checkpoint_path = f'{args.work_dir}/latest.pth'
    if os.path.exists(checkpoint_path):
        run_evaluation(args.config, checkpoint_path)
    else:
        print(f" Checkpoint not found: {checkpoint_path}")
        print("Skipping evaluation...")
    
    print("\n" + "=" * 50)
    print("Fine-tuning pipeline completed!")
    print(f"Results in: {args.work_dir}")
    print(f"Checkpoints: {args.work_dir}/epoch_*.pth")
    print(f"Logs: {args.work_dir}/*.log")

if __name__ == '__main__':
    main()