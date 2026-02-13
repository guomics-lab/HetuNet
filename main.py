#!/usr/bin/env python3
"""
Main entry point for CNN-based protein reconstruction training.

This script orchestrates the entire training pipeline:
1. Parse command-line arguments
2. Load and preprocess data
3. Initialize and train the model
4. Save checkpoints and visualizations

Usage:
    python main.py --image_path <path> --mask_path <path> --protein_path <path> --output_dir <path>
"""

from src.config import parse_args
from src.utils import setup_logging, set_seed
from src.data_loader import load_high_res_image, load_mask, load_protein_data
from src.train import train_model_with_L1


def main():
    """Main execution function."""
    # Setup logging
    setup_logging()
    
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load and preprocess data
    high_res_image = load_high_res_image(args.image_path)
    shared_mask = load_mask(args.mask_path)
    all_protein_specific_data = load_protein_data(
        args.protein_path,
        args.protein_name,
        shared_mask
    )
    
    # Train the model
    train_model_with_L1(args, high_res_image, shared_mask, all_protein_specific_data)


if __name__ == '__main__':
    main()
