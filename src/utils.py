#!/usr/bin/env python3
"""
Utility functions for CNN-based protein reconstruction.
"""
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging


def setup_logging():
    """Configure logging system for real-time output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()  # Output to stdout
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def set_seed(seed_value=42):
    """Set a fixed seed for all relevant random number generators."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_loss_curve(loss_history: list, save_path: str, 
                   title='Training Loss Curve', ylabel='Average Loss'):
    """Plot and save loss curve."""
    if not loss_history:
        return
    
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history) + 1)
    plt.plot(epochs, loss_history, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    
    if len(epochs) <= 20:
        plt.xticks(epochs)
    else:
        plt.xticks(np.arange(1, len(epochs) + 1, step=max(1, len(epochs) // 10)))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def fill_na_with_neighbor_mean(row):
    """Fill NA values with the mean of neighboring values."""
    row = row.copy()
    values = row.values.astype(float)
    
    for i in range(len(values)):
        if values[i] == 0:
            left = values[i - 1] if i - 1 >= 0 else 0
            right = values[i + 1] if i + 1 < len(values) else 0
            neighbors = []
            neighbors.append(left)
            neighbors.append(right)
            if len(neighbors) > 0:
                values[i] = np.mean(neighbors)
    
    return pd.Series(values, index=row.index)
