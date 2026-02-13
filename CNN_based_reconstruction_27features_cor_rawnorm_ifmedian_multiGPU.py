#!/usr/bin/env python3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from collections import defaultdict
import os
import itertools
import re
from datetime import datetime
import json
import tifffile
import torch.serialization
import argparse
import logging

# --- 日志设置 ---
def setup_logging():
    """配置日志系统，确保实时输出。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler() # 输出到标准输出
        ]
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

# --- 辅助函数 ---
def set_seed(seed_value=42):
    """为所有相关的随机数生成器设置一个固定的种子。"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_loss_curve(loss_history: list, save_path: str, title='Training Loss Curve', ylabel='Average Loss'):
    """绘制并保存损失曲线图。"""
    if not loss_history: return
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
    
# --- [NEW] 相关系数损失函数 ---
class PearsonCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        """计算 x 和 y 之间的皮尔逊相关系数损失。"""
        x = x.float()
        y = y.float()
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + self.eps)
        return 1 - corr
    
# --- 模型定义 ---
class MultiScalePatchCNN(nn.Module):
    """改造后的CNN，可以输出多尺度的特征。"""
    def __init__(self, in_channels=4, patch_size=25, common_dim=256):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.SiLU(), nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(64), nn.SiLU())
        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.SiLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(128), nn.SiLU())
        self.stage3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.SiLU(), nn.Conv2d(256, common_dim, kernel_size=3, padding=1, stride=2), nn.BatchNorm2d(common_dim), nn.SiLU())
        self.proj1 = nn.Conv2d(64, common_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(128, common_dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        s1_out = self.stage1(x)
        s2_out = self.stage2(s1_out)
        s3_out = self.stage3(s2_out)
        f_shallow = torch.flatten(self.pool(self.proj1(s1_out)), 1)
        f_mid = torch.flatten(self.pool(self.proj2(s2_out)), 1)
        f_deep = torch.flatten(self.pool(s3_out), 1)
        return {'shallow': f_shallow, 'mid': f_mid, 'deep': f_deep}

class GatedProteinHead(nn.Module):
    """全新的蛋白质头，包含门控、特征混合、SE注意力和预测器。"""
    def __init__(self, in_features=256, reduction=16):
        super().__init__()
        self.gating_network = nn.Sequential(nn.Linear(in_features, 64), nn.SiLU(), nn.Linear(64, 3), nn.Softmax(dim=-1))
        self.attention = nn.Sequential(nn.Linear(in_features, in_features // reduction, bias=False), nn.SiLU(), nn.Linear(in_features // reduction, in_features, bias=False), nn.Sigmoid())
        self.predictor = nn.Sequential(nn.Linear(in_features, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1), nn.LeakyReLU(negative_slope=0.01))
    def forward(self, multi_scale_features):
        f_shallow, f_mid, f_deep = multi_scale_features['shallow'], multi_scale_features['mid'], multi_scale_features['deep']
        gates = self.gating_network(f_deep)
        g1, g2, g3 = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        f_personalized = g1 * f_shallow + g2 * f_mid + g3 * f_deep
        recalibrated_features = f_personalized * self.attention(f_personalized)
        return self.predictor(recalibrated_features)

# --- 数据集与数据加载器 ---
class SpatioTemporalDataset(Dataset):
    def __init__(self, all_samples_meta, targets_dict, shared_mask_tensor, high_res_image, patch_size, h_scale, w_scale):
        super().__init__()
        self.all_samples_meta, self.targets_dict, self.shared_mask_tensor = all_samples_meta, targets_dict, shared_mask_tensor
        self.H, self.W = shared_mask_tensor.shape
        self.high_res_image, self.patch_size, self.pad_size = high_res_image, patch_size, patch_size // 2
        self.h_scale, self.w_scale = h_scale, w_scale
    def __len__(self) -> int: return len(self.all_samples_meta)
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not hasattr(self, 'padded_high_res_tensor'):
            self.padded_high_res_tensor = F.pad(torch.from_numpy(self.high_res_image.astype(np.float32)).permute(2, 0, 1), (self.pad_size,) * 4, mode='reflect')
        sample_meta = self.all_samples_meta[idx]
        p_name, sample_type, sample_idx = sample_meta['p_name'], sample_meta['type'], sample_meta['idx']
        target_value = self.targets_dict[p_name][sample_type][sample_idx]
        coords = [(sample_idx, c) for c in range(self.W)] if sample_type == 'R' else [(r, sample_idx) for r in range(self.H)]
        mask_slice = self.shared_mask_tensor[sample_idx, :] if sample_type == 'R' else self.shared_mask_tensor[:, sample_idx]
        patches = [self.padded_high_res_tensor[:, int(r*self.h_scale+self.h_scale/2):int(r*self.h_scale+self.h_scale/2)+self.patch_size, int(c*self.w_scale+self.w_scale/2):int(c*self.w_scale+self.w_scale/2)+self.patch_size] for i, (r, c) in enumerate(coords) if mask_slice[i]]
        return {'p_name': p_name, 'patches': torch.stack(patches) if patches else torch.empty(0), 'target': target_value}
def custom_collate_fn(batch: List[Dict[str, Any]]):
    groups=defaultdict(list); all_patches, p_name_indices, start_idx = [], {}, 0
    for s in batch:
        if s['patches'].numel() > 0: groups[s['p_name']].append(s)
    for p_name, samples in groups.items():
        p_name_indices[p_name]={'targets':[], 'patch_indices_per_sample':[]}
        for s in samples:
            num_p=s['patches'].shape[0]; p_name_indices[p_name]['patch_indices_per_sample'].append((start_idx,start_idx+num_p))
            p_name_indices[p_name]['targets'].append(s['target']); all_patches.append(s['patches']); start_idx+=num_p
    if not all_patches: return None
    return {'patches_tensor':torch.cat(all_patches,dim=0), 'p_name_indices':p_name_indices}
class InferenceDataset(Dataset):
    def __init__(self, H, W, high_res_image, patch_size, h_scale, w_scale):
        self.H,self.W=H,W; self.high_res_image,self.patch_size,self.pad_size=high_res_image,patch_size,patch_size//2; self.h_scale,self.w_scale=h_scale,w_scale
    def __len__(self)->int: return self.H*self.W
    def __getitem__(self,idx:int):
        if not hasattr(self,'padded_high_res_tensor'):
            self.padded_high_res_tensor=F.pad(torch.from_numpy(self.high_res_image.astype(np.float32)).permute(2,0,1),(self.pad_size,)*4,mode='reflect')
        r,c=idx//self.W,idx%self.W; center_h,center_w=int(r*self.h_scale+self.h_scale/2),int(c*self.w_scale+self.w_scale/2)
        return self.padded_high_res_tensor[:,center_h:center_h+self.patch_size,center_w:center_w+self.patch_size]

# --- 核心训练函数 ---
def train_model_with_L1(args, high_res_image, shared_mask, all_protein_specific_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"User: {args.user_login}, Timestamp: {args.start_time}")
    logging.info(f"Using device: {device}, Patch Size: {args.patch_size}x{args.patch_size}, TV Lambda: {args.tv_lambda}, Correlation Lambda: {args.correlation_lambda}")
    logging.info(f"Full arguments: {args}")
    
    H, W = shared_mask.shape; h_scale, w_scale = high_res_image.shape[0] / H, high_res_image.shape[1] / W
    shared_mask_tensor = torch.from_numpy(shared_mask).bool()
    
    shared_cnn = MultiScalePatchCNN(in_channels=high_res_image.shape[2], patch_size=args.patch_size).to(device)
    predictor_heads = {p['name']: GatedProteinHead().to(device) for p in all_protein_specific_data}
    
    all_samples_meta, targets_dict = [], {}
    for p in all_protein_specific_data:
        targets_dict[p['name']] = {'R': torch.from_numpy(p['target_R']).float(), 'C': torch.from_numpy(p['target_C']).float()}
        for i in range(H):
            if targets_dict[p['name']]['R'][i] > 0: all_samples_meta.append({'p_name': p['name'], 'type': 'R', 'idx': i})
        for i in range(W):
            if targets_dict[p['name']]['C'][i] > 0: all_samples_meta.append({'p_name': p['name'], 'type': 'C', 'idx': i})
    
    dataset = SpatioTemporalDataset(all_samples_meta, targets_dict, shared_mask_tensor, high_res_image, args.patch_size, h_scale, w_scale)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=args.num_workers > 0)
    param_groups = [{'params': shared_cnn.parameters(), 'lr': args.lr * args.cnn_lr_fold}, {'params': itertools.chain(*[h.parameters() for h in predictor_heads.values()]), 'lr': args.lr}]
    optimizer = torch.optim.Adam(param_groups); mae_loss_fn = nn.L1Loss()
    correlation_loss_fn = PearsonCorrelationLoss()
    os.makedirs(args.output_dir, exist_ok=True); start_epoch = 0; history = {'epoch_avg_loss': [], 'epoch_mae_loss': [], 'epoch_tv_loss': [], 'epoch_corr_loss': []}

    latest_ckpt = max([os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.endswith('.pth')], key=os.path.getctime, default=None)
    if latest_ckpt and not args.no_resume:
        logging.info(f"--- Loading checkpoint: '{latest_ckpt}' ---")
        # with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype]):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        shared_cnn.load_state_dict(ckpt['shared_cnn_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict']); start_epoch = ckpt['epoch'] + 1
        for name, head in predictor_heads.items():
            if name in ckpt['predictor_heads_state_dict']: head.load_state_dict(ckpt['predictor_heads_state_dict'][name])
        history = ckpt.get('history', history); logging.info(f"Resumed successfully from Epoch {start_epoch + 1}.")
    else: 
        logging.info("--- No checkpoint found or --no-resume flag set, starting from scratch. ---")

    for epoch in range(start_epoch, args.epochs):
        shared_cnn.train(); [h.train() for h in predictor_heads.values()]
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_total_loss, epoch_mae_loss, epoch_tv_loss, epoch_corr_loss, num_batches = 0.0, 0.0, 0.0, 0.0, 0
        for batch in pbar:
            if batch is None: continue
            optimizer.zero_grad()
            multi_scale_features = shared_cnn(batch['patches_tensor'].to(device, non_blocking=True))
            # total_mae_loss, total_tv_loss, total_corr_loss, num_groups = 0.0, 0.0, 0.0, 0
            total_mae_loss = torch.tensor(0.0, device=device)
            total_tv_loss = torch.tensor(0.0, device=device)
            total_corr_loss = torch.tensor(0.0, device=device)
            num_groups = 0
            for p_name, indices_info in batch['p_name_indices'].items():
                head, predicted_sums, targets, group_tv_loss = predictor_heads[p_name], [], [], 0.0
                # --- [NEW] 新增逻辑：将 batch 中属于同一个蛋白质的所有行/列聚合起来 ---
                # 我们的目标是计算一个蛋白质的所有行/列预测值与真实值的相关性
                all_preds_for_protein = []
                all_targets_for_protein = []
                for (start, end), target in zip(indices_info['patch_indices_per_sample'], indices_info['targets']):
                    if start == end: continue
                    features_for_sample = {key: val[start:end] for key, val in multi_scale_features.items()}
                    predicted_values = head(features_for_sample).squeeze(-1)
                    if len(predicted_values) > 1: group_tv_loss += torch.sum(torch.abs(predicted_values[1:] - predicted_values[:-1]))
                    
                    # 收集预测和真实的总和
                    current_pred_sum = torch.sum(predicted_values)
                    all_preds_for_protein.append(current_pred_sum)
                    all_targets_for_protein.append(target.to(device, non_blocking=True))
                if not all_preds_for_protein: continue
                # 计算 MAE Loss (与之前相同，但现在用收集到的列表)
                total_mae_loss += mae_loss_fn(torch.stack(all_preds_for_protein), torch.stack(all_targets_for_protein))
                total_tv_loss += group_tv_loss
                # --- [NEW] 计算相关系数损失 ---
                # 只有当一个batch中某个蛋白质的样本数>1时，计算相关性才有意义
                if len(all_preds_for_protein) > 1:
                    preds_tensor = torch.stack(all_preds_for_protein)
                    targets_tensor = torch.stack(all_targets_for_protein)
                    total_corr_loss += correlation_loss_fn(preds_tensor, targets_tensor)
                num_groups += 1
            if num_groups > 0:
                avg_mae_loss = total_mae_loss / num_groups
                avg_tv_loss = total_tv_loss / num_groups
                avg_corr_loss = total_corr_loss / num_groups # 可能是0
                # --- [NEW] 将相关系数损失加入总损失 ---
                final_batch_loss = avg_mae_loss \
                                 + args.tv_lambda * avg_tv_loss \
                                 + args.correlation_lambda * avg_corr_loss
                final_batch_loss.backward(); optimizer.step()
                pbar.set_postfix(mae=avg_mae_loss.item(), tv=(args.tv_lambda * avg_tv_loss).item(), corr=(args.correlation_lambda * avg_corr_loss).item())
                # pbar.set_postfix(mae=float(avg_mae_loss), tv=float(args.tv_lambda * avg_tv_loss), corr=float(args.correlation_lambda * avg_corr_loss))
                epoch_total_loss += final_batch_loss.item()
                epoch_mae_loss += avg_mae_loss.item()
                epoch_tv_loss += (args.tv_lambda * avg_tv_loss).item()
                epoch_corr_loss += (args.correlation_lambda * avg_corr_loss).item()
                # epoch_corr_loss += float(args.correlation_lambda * avg_corr_loss)
                num_batches += 1
        if num_batches > 0:
            history['epoch_avg_loss'].append(epoch_total_loss / num_batches)
            history['epoch_mae_loss'].append(epoch_mae_loss / num_batches)
            history['epoch_tv_loss'].append(epoch_tv_loss / num_batches)
            history['epoch_corr_loss'].append(epoch_corr_loss / num_batches)
            logging.info(f"--- Epoch {epoch+1} Done --- Avg Total Loss: {history['epoch_avg_loss'][-1]:.6f} (MAE: {history['epoch_mae_loss'][-1]:.6f} + TV: {history['epoch_tv_loss'][-1]:.6f}) ---")
        
        logging.info(f"Generating full prediction matrices for Epoch {epoch+1}...")
        shared_cnn.eval(); [h.eval() for h in predictor_heads.values()]
        inference_dataset = InferenceDataset(H, W, high_res_image, args.patch_size, h_scale, w_scale)
        inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size*4, num_workers=args.num_workers, pin_memory=True)
        predicted_matrices = {}
        with torch.no_grad():
            full_predicted_matrices = {p_name: torch.zeros((H, W), device=device) for p_name in predictor_heads.keys()}
            for i, patches in enumerate(tqdm(inference_loader, desc="  - Extracting features and predicting")):
                start_idx = i * (args.batch_size * 4); end_idx = start_idx + len(patches)
                if len(patches) == 0: continue
                multi_scale_features = shared_cnn(patches.to(device, non_blocking=True))
                for p_name, head in predictor_heads.items():
                    preds_batch = head(multi_scale_features).squeeze(-1)
                    for j, p in enumerate(preds_batch):
                        row, col = (start_idx + j) // W, (start_idx + j) % W
                        if row < H: full_predicted_matrices[p_name][row, col] = p
            for p_name, full_matrix in full_predicted_matrices.items():
                final_matrix = full_matrix * shared_mask_tensor.to(device).float()
                scale_factor = next((d['scale_factor'] for d in all_protein_specific_data if d['name'] == p_name), 1.0)
                final_prediction = final_matrix.cpu().numpy() * scale_factor
                final_prediction[final_prediction < 0] = 0
                predicted_matrices[p_name] = final_prediction
                # predicted_matrices[p_name] = final_matrix.cpu().numpy() * scale_factor
        
        ckpt_data = {'epoch': epoch, 'shared_cnn_state_dict': shared_cnn.state_dict(), 'predictor_heads_state_dict': {n: h.state_dict() for n, h in predictor_heads.items()}, 'optimizer_state_dict': optimizer.state_dict(), 'history': history, 'predicted_matrices': predicted_matrices}
        torch.save(ckpt_data, os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pth"))
        if history['epoch_avg_loss']:
            plot_loss_curve(history['epoch_avg_loss'], os.path.join(args.output_dir, "loss_curve_total.png"), title='Total Training Loss')
            plot_loss_curve(history['epoch_mae_loss'], os.path.join(args.output_dir, "loss_curve_mae.png"), title='MAE Reconstruction Loss')
            plot_loss_curve(history['epoch_tv_loss'], os.path.join(args.output_dir, "loss_curve_tv.png"), title='TV Regularization Loss')

    logging.info("--- Training completed. ---")

def fill_na_with_neighbor_mean(row):
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

# --- 主函数入口 ---
def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Spatio-Temporal Protein Reconstruction Training Script for Cluster")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the high-resolution ome.tif image file.')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to the tissue mask pickle file.')
    parser.add_argument('--protein_path', type=str, required=True, help='Path to the protein data CSV file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints and logs.')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Base learning rate.')
    parser.add_argument('--cnn_lr_fold', type=float, default=1.0, help='Fold change for CNN learning rate compared to base.')
    parser.add_argument('--tv_lambda', type=float, default=1e-5, help='Weight for the Total Variation regularization term.')
    parser.add_argument('--patch_size', type=int, default=25, help='Side length of the square patches.')
    parser.add_argument('--seed', type=int, default=888, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading.')
    parser.add_argument('--no_resume', action='store_true', help='Flag to force training from scratch, ignoring existing checkpoints.')
    parser.add_argument('--correlation_lambda', type=float, default=1, help='Weight for the Pearson Correlation loss term.')
    parser.add_argument('--protein_name', type=str, default="all", help='Protein name that you wanna train.')
    args = parser.parse_args()
    
    args.user_login = "Wandershy"
    args.start_time = "2025-11-11 05:41:48 UTC"
    
    set_seed(args.seed)

    # --- 完整的数据加载和预处理逻辑 ---
    logging.info("Loading and pre-processing data from paths specified in arguments...")
    # 1. 加载高分辨率图像
    fullres_multich_img1 = tifffile.imread(args.image_path, is_ome=True, level=0, aszarr=False)
    fullres_multich_img = np.transpose(fullres_multich_img1, (1, 2, 0))
    rgbd_input = []
    for i in range(fullres_multich_img.shape[2]):
        img_matrix = fullres_multich_img[:, :, i]
        img_matrix_nozero = img_matrix[img_matrix > 0]
        matrix_median = img_matrix - np.median(img_matrix_nozero)
        # matrix_median = img_matrix - np.median(img_matrix)
        matrix_median[np.where(matrix_median<0)] = 0
        rgbd_input.append(matrix_median)
    if_rgbd_input = np.stack(rgbd_input, axis=2)
    if_rgbd_input_processed = if_rgbd_input.copy()
    for i in range(if_rgbd_input.shape[2]):
        channel = if_rgbd_input[:, :, i]
        q99 = np.quantile(channel, 0.999)
        median = np.median(channel)
        channel[channel > q99] = median
        if_rgbd_input_processed[:, :, i] = channel
    max_vals = if_rgbd_input_processed.max(axis=(0, 1))
    scale_factors = max_vals / 255.0
    scale_factors[scale_factors == 0] = 1
    # 注意：这里假设您的多通道图像最后会被选择为4通道，如果不是，需要调整
    if_rgbd_input_32_scaled = if_rgbd_input_processed / scale_factors[np.newaxis, np.newaxis, :]
    # high_res_image = if_rgbd_input_32_scaled[:,:,[0,2,14,20]]
    high_res_image = np.delete(if_rgbd_input_32_scaled, np.s_[7,10,11,26,30], axis=2)
    # 为了通用性，我们直接使用处理后的图像
    # high_res_image = if_rgbd_input_processed / scale_factors[np.newaxis, np.newaxis, :]

    # 2. 加载Mask
    with open(args.mask_path, 'rb') as handle:
        shared_mask = pickle.load(handle)
    shared_mask = shared_mask.astype(bool)
    
    # 3. 加载并处理蛋白质数据
    rawdata = pd.read_csv(args.protein_path)
    rawdata.iloc[:,2:] = rawdata.iloc[:,2:].fillna(0)
    rawdata.iloc[:, 1] = rawdata.iloc[:, 1].fillna(rawdata.iloc[:, 0])
    rawdata.iloc[:, 2:] = rawdata.iloc[:, 2:].apply(fill_na_with_neighbor_mean, axis=1)
    
    if args.protein_name == "all":
        protein_names = list(rawdata['Genes'])
    else:
        protein_names = args.protein_name
        protein_names = protein_names.split(',')
        print(protein_names)
    # protein_names = ['VIM', 'CDH1']
    scale = 255
    all_protein_specific_data = []
    H_grid, W_grid = shared_mask.shape
    
    logging.info(f"Preparing data for {len(protein_names)} proteins...")
    for name in protein_names:
        protein_series = rawdata.loc[rawdata['Genes'] == name].iloc[0]
        # 注意：这里的切片索引是硬编码的，如果您的CSV格式变化，需要修改
        original_target_r = protein_series.iloc[2:2+H_grid].values.astype(float)
        original_target_c = protein_series.iloc[2+H_grid:2+H_grid+W_grid].values.astype(float)
        
        q99_r = np.quantile(original_target_r, 0.99)
        median_r = np.median(original_target_r)
        original_target_r[original_target_r > q99_r] = q99_r
        q99_c = np.quantile(original_target_c, 0.99)
        median_c = np.median(original_target_c)
        original_target_c[original_target_c > q99_c] = q99_c
        
        sum_r = np.sum(original_target_r)
        sum_c = np.sum(original_target_c)
        if sum_c > 1e-9: original_target_c *= (sum_r / sum_c)
        scale_factor = max(original_target_r.max(), 1) / scale
        scale_factor = max(scale_factor, 1e-9)
        scaled_target_r = original_target_r / scale_factor
        scaled_target_c = original_target_c / scale_factor
        all_protein_specific_data.append({
            'name': name, 'target_R': scaled_target_r,
            'target_C': scaled_target_c, 'scale_factor': scale_factor
        })
    logging.info("Data loading and pre-processing finished.")

    # 调用主训练函数
    train_model_with_L1(args, high_res_image, shared_mask, all_protein_specific_data)

if __name__ == '__main__':
    main()