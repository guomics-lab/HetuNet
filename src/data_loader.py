#!/usr/bin/env python3
"""
Data loading and preprocessing utilities for protein reconstruction.
"""
import pickle
import numpy as np
import pandas as pd
import tifffile
import logging

from src.utils import fill_na_with_neighbor_mean


def load_high_res_image(image_path):
    """Load and preprocess high-resolution image."""
    logging.info(f"Loading high-resolution image from {image_path}...")
    
    # Load image
    fullres_multich_img1 = tifffile.imread(image_path, is_ome=True, level=0, aszarr=False)
    fullres_multich_img = np.transpose(fullres_multich_img1, (1, 2, 0))
    
    # Process each channel
    rgbd_input = []
    for i in range(fullres_multich_img.shape[2]):
        img_matrix = fullres_multich_img[:, :, i]
        img_matrix_nozero = img_matrix[img_matrix > 0]
        matrix_median = img_matrix - np.median(img_matrix_nozero)
        matrix_median[np.where(matrix_median < 0)] = 0
        rgbd_input.append(matrix_median)
    
    if_rgbd_input = np.stack(rgbd_input, axis=2)
    
    # Process outliers
    if_rgbd_input_processed = if_rgbd_input.copy()
    for i in range(if_rgbd_input.shape[2]):
        channel = if_rgbd_input[:, :, i]
        q99 = np.quantile(channel, 0.999)
        median = np.median(channel)
        channel[channel > q99] = median
        if_rgbd_input_processed[:, :, i] = channel
    
    # Normalize
    max_vals = if_rgbd_input_processed.max(axis=(0, 1))
    scale_factors = max_vals / 255.0
    scale_factors[scale_factors == 0] = 1
    if_rgbd_input_32_scaled = if_rgbd_input_processed / scale_factors[np.newaxis, np.newaxis, :]
    
    # Remove specific channels
    high_res_image = np.delete(if_rgbd_input_32_scaled, np.s_[7, 10, 11, 26, 30], axis=2)
    
    logging.info(f"High-resolution image loaded. Shape: {high_res_image.shape}")
    return high_res_image


def load_mask(mask_path):
    """Load tissue mask."""
    logging.info(f"Loading mask from {mask_path}...")
    with open(mask_path, 'rb') as handle:
        shared_mask = pickle.load(handle)
    shared_mask = shared_mask.astype(bool)
    logging.info(f"Mask loaded. Shape: {shared_mask.shape}")
    return shared_mask


def load_protein_data(protein_path, protein_name, shared_mask):
    """Load and process protein data."""
    logging.info(f"Loading protein data from {protein_path}...")
    
    # Load CSV
    rawdata = pd.read_csv(protein_path)
    rawdata.iloc[:, 2:] = rawdata.iloc[:, 2:].fillna(0)
    rawdata.iloc[:, 1] = rawdata.iloc[:, 1].fillna(rawdata.iloc[:, 0])
    rawdata.iloc[:, 2:] = rawdata.iloc[:, 2:].apply(fill_na_with_neighbor_mean, axis=1)
    
    # Determine proteins to process
    if protein_name == "all":
        protein_names = list(rawdata['Genes'])
    else:
        protein_names = protein_name.split(',')
        logging.info(f"Processing specific proteins: {protein_names}")
    
    scale = 255
    all_protein_specific_data = []
    H_grid, W_grid = shared_mask.shape
    
    logging.info(f"Preparing data for {len(protein_names)} proteins...")
    for name in protein_names:
        protein_series = rawdata.loc[rawdata['Genes'] == name].iloc[0]
        
        # Extract target values
        original_target_r = protein_series.iloc[2:2+H_grid].values.astype(float)
        original_target_c = protein_series.iloc[2+H_grid:2+H_grid+W_grid].values.astype(float)
        
        # Remove outliers
        q99_r = np.quantile(original_target_r, 0.99)
        original_target_r[original_target_r > q99_r] = q99_r
        q99_c = np.quantile(original_target_c, 0.99)
        original_target_c[original_target_c > q99_c] = q99_c
        
        # Normalize C to match R
        sum_r = np.sum(original_target_r)
        sum_c = np.sum(original_target_c)
        if sum_c > 1e-9:
            original_target_c *= (sum_r / sum_c)
        
        # Scale
        scale_factor = max(original_target_r.max(), 1) / scale
        scale_factor = max(scale_factor, 1e-9)
        scaled_target_r = original_target_r / scale_factor
        scaled_target_c = original_target_c / scale_factor
        
        all_protein_specific_data.append({
            'name': name,
            'target_R': scaled_target_r,
            'target_C': scaled_target_c,
            'scale_factor': scale_factor
        })
    
    logging.info("Data loading and pre-processing finished.")
    return all_protein_specific_data
