#!/usr/bin/env python3
"""
Configuration and argument parsing for CNN-based protein reconstruction.
"""
import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Spatio-Temporal Protein Reconstruction Training Script for Cluster"
    )
    
    # Required arguments
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='Path to the high-resolution ome.tif image file.'
    )
    parser.add_argument(
        '--mask_path',
        type=str,
        required=True,
        help='Path to the tissue mask pickle file.'
    )
    parser.add_argument(
        '--protein_path',
        type=str,
        required=True,
        help='Path to the protein data CSV file.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save checkpoints and logs.'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for training.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='Base learning rate.'
    )
    parser.add_argument(
        '--cnn_lr_fold',
        type=float,
        default=1.0,
        help='Fold change for CNN learning rate compared to base.'
    )
    parser.add_argument(
        '--tv_lambda',
        type=float,
        default=1e-5,
        help='Weight for the Total Variation regularization term.'
    )
    parser.add_argument(
        '--correlation_lambda',
        type=float,
        default=1,
        help='Weight for the Pearson Correlation loss term.'
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=25,
        help='Side length of the square patches.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=888,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker processes for data loading.'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Flag to force training from scratch, ignoring existing checkpoints.'
    )
    parser.add_argument(
        '--protein_name',
        type=str,
        default="all",
        help='Protein name that you wanna train.'
    )
    
    args = parser.parse_args()
    
    # Add metadata
    args.user_login = "Wandershy"
    args.start_time = "2025-11-11 05:41:48 UTC"
    
    return args
