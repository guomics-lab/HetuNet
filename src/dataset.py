#!/usr/bin/env python3
"""
Dataset and DataLoader utilities for spatio-temporal protein reconstruction.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np


class SpatioTemporalDataset(Dataset):
    """Dataset for spatio-temporal protein reconstruction training."""
    
    def __init__(self, all_samples_meta, targets_dict, shared_mask_tensor, 
                 high_res_image, patch_size, h_scale, w_scale):
        super().__init__()
        self.all_samples_meta = all_samples_meta
        self.targets_dict = targets_dict
        self.shared_mask_tensor = shared_mask_tensor
        self.H, self.W = shared_mask_tensor.shape
        self.high_res_image = high_res_image
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.h_scale = h_scale
        self.w_scale = w_scale
    
    def __len__(self) -> int:
        return len(self.all_samples_meta)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not hasattr(self, 'padded_high_res_tensor'):
            self.padded_high_res_tensor = F.pad(
                torch.from_numpy(self.high_res_image.astype(np.float32)).permute(2, 0, 1),
                (self.pad_size,) * 4,
                mode='reflect'
            )
        
        sample_meta = self.all_samples_meta[idx]
        p_name = sample_meta['p_name']
        sample_type = sample_meta['type']
        sample_idx = sample_meta['idx']
        target_value = self.targets_dict[p_name][sample_type][sample_idx]
        
        if sample_type == 'R':
            coords = [(sample_idx, c) for c in range(self.W)]
        else:
            coords = [(r, sample_idx) for r in range(self.H)]
        
        if sample_type == 'R':
            mask_slice = self.shared_mask_tensor[sample_idx, :]
        else:
            mask_slice = self.shared_mask_tensor[:, sample_idx]
        
        patches = []
        for i, (r, c) in enumerate(coords):
            if mask_slice[i]:
                center_h = int(r * self.h_scale + self.h_scale / 2)
                center_w = int(c * self.w_scale + self.w_scale / 2)
                patch = self.padded_high_res_tensor[
                    :,
                    center_h:center_h + self.patch_size,
                    center_w:center_w + self.patch_size
                ]
                patches.append(patch)
        
        return {
            'p_name': p_name,
            'patches': torch.stack(patches) if patches else torch.empty(0),
            'target': target_value
        }


def custom_collate_fn(batch: List[Dict[str, Any]]):
    """Custom collate function for batching samples by protein name."""
    groups = defaultdict(list)
    all_patches = []
    p_name_indices = {}
    start_idx = 0
    
    for s in batch:
        if s['patches'].numel() > 0:
            groups[s['p_name']].append(s)
    
    for p_name, samples in groups.items():
        p_name_indices[p_name] = {
            'targets': [],
            'patch_indices_per_sample': []
        }
        for s in samples:
            num_p = s['patches'].shape[0]
            p_name_indices[p_name]['patch_indices_per_sample'].append(
                (start_idx, start_idx + num_p)
            )
            p_name_indices[p_name]['targets'].append(s['target'])
            all_patches.append(s['patches'])
            start_idx += num_p
    
    if not all_patches:
        return None
    
    return {
        'patches_tensor': torch.cat(all_patches, dim=0),
        'p_name_indices': p_name_indices
    }


class InferenceDataset(Dataset):
    """Dataset for inference on full grid."""
    
    def __init__(self, H, W, high_res_image, patch_size, h_scale, w_scale):
        self.H = H
        self.W = W
        self.high_res_image = high_res_image
        self.patch_size = patch_size
        self.pad_size = patch_size // 2
        self.h_scale = h_scale
        self.w_scale = w_scale
    
    def __len__(self) -> int:
        return self.H * self.W
    
    def __getitem__(self, idx: int):
        if not hasattr(self, 'padded_high_res_tensor'):
            self.padded_high_res_tensor = F.pad(
                torch.from_numpy(self.high_res_image.astype(np.float32)).permute(2, 0, 1),
                (self.pad_size,) * 4,
                mode='reflect'
            )
        
        r = idx // self.W
        c = idx % self.W
        center_h = int(r * self.h_scale + self.h_scale / 2)
        center_w = int(c * self.w_scale + self.w_scale / 2)
        
        return self.padded_high_res_tensor[
            :,
            center_h:center_h + self.patch_size,
            center_w:center_w + self.patch_size
        ]
