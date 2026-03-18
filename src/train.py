#!/usr/bin/env python3
"""
Training logic for spatio-temporal protein reconstruction.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import itertools

from src.model import MultiScalePatchCNN, GatedProteinHead, PearsonCorrelationLoss
from src.dataset import SpatioTemporalDataset, InferenceDataset, custom_collate_fn
from src.utils import plot_loss_curve


def train_model_with_L1(args, high_res_image, shared_mask, all_protein_specific_data):
    """Main training function for protein reconstruction model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"User: {args.user_login}, Timestamp: {args.start_time}")
    logging.info(f"Using device: {device}, Patch Size: {args.patch_size}x{args.patch_size}, "
                f"TV Lambda: {args.tv_lambda}, Correlation Lambda: {args.correlation_lambda}")
    logging.info(f"Full arguments: {args}")
    
    H, W = shared_mask.shape
    h_scale = high_res_image.shape[0] / H
    w_scale = high_res_image.shape[1] / W
    shared_mask_tensor = torch.from_numpy(shared_mask).bool()
    
    # Initialize models
    shared_cnn = MultiScalePatchCNN(
        in_channels=high_res_image.shape[2],
        patch_size=args.patch_size
    ).to(device)
    predictor_heads = {
        p['name']: GatedProteinHead().to(device)
        for p in all_protein_specific_data
    }
    
    # Prepare training data
    all_samples_meta = []
    targets_dict = {}
    for p in all_protein_specific_data:
        targets_dict[p['name']] = {
            'R': torch.from_numpy(p['target_R']).float(),
            'C': torch.from_numpy(p['target_C']).float()
        }
        for i in range(H):
            if targets_dict[p['name']]['R'][i] > 0:
                all_samples_meta.append({'p_name': p['name'], 'type': 'R', 'idx': i})
        for i in range(W):
            if targets_dict[p['name']]['C'][i] > 0:
                all_samples_meta.append({'p_name': p['name'], 'type': 'C', 'idx': i})
    
    # Create dataset and dataloader
    dataset = SpatioTemporalDataset(
        all_samples_meta, targets_dict, shared_mask_tensor,
        high_res_image, args.patch_size, h_scale, w_scale
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    
    # Setup optimizer
    param_groups = [
        {'params': shared_cnn.parameters(), 'lr': args.lr * args.cnn_lr_fold},
        {'params': itertools.chain(*[h.parameters() for h in predictor_heads.values()]), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(param_groups)
    
    # Loss functions
    mae_loss_fn = nn.L1Loss()
    correlation_loss_fn = PearsonCorrelationLoss()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    start_epoch = 0
    history = {
        'epoch_avg_loss': [],
        'epoch_mae_loss': [],
        'epoch_tv_loss': [],
        'epoch_corr_loss': []
    }
    
    # Load checkpoint if exists
    latest_ckpt = max(
        [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir) if f.endswith('.pth')],
        key=os.path.getctime,
        default=None
    )
    if latest_ckpt and not args.no_resume:
        logging.info(f"--- Loading checkpoint: '{latest_ckpt}' ---")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        shared_cnn.load_state_dict(ckpt['shared_cnn_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        for name, head in predictor_heads.items():
            if name in ckpt['predictor_heads_state_dict']:
                head.load_state_dict(ckpt['predictor_heads_state_dict'][name])
        history = ckpt.get('history', history)
        logging.info(f"Resumed successfully from Epoch {start_epoch + 1}.")
    else:
        logging.info("--- No checkpoint found or --no-resume flag set, starting from scratch. ---")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        shared_cnn.train()
        [h.train() for h in predictor_heads.values()]
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_total_loss = 0.0
        epoch_mae_loss = 0.0
        epoch_tv_loss = 0.0
        epoch_corr_loss = 0.0
        num_batches = 0
        
        for batch in pbar:
            if batch is None:
                continue
            
            optimizer.zero_grad()
            multi_scale_features = shared_cnn(batch['patches_tensor'].to(device, non_blocking=True))
            
            total_mae_loss = torch.tensor(0.0, device=device)
            total_tv_loss = torch.tensor(0.0, device=device)
            total_corr_loss = torch.tensor(0.0, device=device)
            num_groups = 0
            
            for p_name, indices_info in batch['p_name_indices'].items():
                head = predictor_heads[p_name]
                group_tv_loss = 0.0
                all_preds_for_protein = []
                all_targets_for_protein = []
                
                for (start, end), target in zip(
                    indices_info['patch_indices_per_sample'],
                    indices_info['targets']
                ):
                    if start == end:
                        continue
                    
                    features_for_sample = {
                        key: val[start:end]
                        for key, val in multi_scale_features.items()
                    }
                    predicted_values = head(features_for_sample).squeeze(-1)
                    
                    if len(predicted_values) > 1:
                        group_tv_loss += torch.sum(
                            torch.abs(predicted_values[1:] - predicted_values[:-1])
                        )
                    
                    current_pred_sum = torch.sum(predicted_values)
                    all_preds_for_protein.append(current_pred_sum)
                    all_targets_for_protein.append(target.to(device, non_blocking=True))
                
                if not all_preds_for_protein:
                    continue
                
                # Compute losses
                total_mae_loss += mae_loss_fn(
                    torch.stack(all_preds_for_protein),
                    torch.stack(all_targets_for_protein)
                )
                total_tv_loss += group_tv_loss
                
                # Compute correlation loss if more than one sample
                if len(all_preds_for_protein) > 1:
                    preds_tensor = torch.stack(all_preds_for_protein)
                    targets_tensor = torch.stack(all_targets_for_protein)
                    total_corr_loss += correlation_loss_fn(preds_tensor, targets_tensor)
                
                num_groups += 1
            
            if num_groups > 0:
                avg_mae_loss = total_mae_loss / num_groups
                avg_tv_loss = total_tv_loss / num_groups
                avg_corr_loss = total_corr_loss / num_groups
                
                final_batch_loss = (avg_mae_loss + 
                                   args.tv_lambda * avg_tv_loss + 
                                   args.correlation_lambda * avg_corr_loss)
                
                final_batch_loss.backward()
                optimizer.step()
                
                pbar.set_postfix(
                    mae=avg_mae_loss.item(),
                    tv=(args.tv_lambda * avg_tv_loss).item(),
                    corr=(args.correlation_lambda * avg_corr_loss).item(),
                    refresh=False
                )
                
                epoch_total_loss += final_batch_loss.item()
                epoch_mae_loss += avg_mae_loss.item()
                epoch_tv_loss += (args.tv_lambda * avg_tv_loss).item()
                epoch_corr_loss += (args.correlation_lambda * avg_corr_loss).item()
                num_batches += 1
        
        # Record history
        if num_batches > 0:
            history['epoch_avg_loss'].append(epoch_total_loss / num_batches)
            history['epoch_mae_loss'].append(epoch_mae_loss / num_batches)
            history['epoch_tv_loss'].append(epoch_tv_loss / num_batches)
            history['epoch_corr_loss'].append(epoch_corr_loss / num_batches)
            logging.info(
                f"--- Epoch {epoch+1} Done --- "
                f"Avg Total Loss: {history['epoch_avg_loss'][-1]:.6f} "
                f"(MAE: {history['epoch_mae_loss'][-1]:.6f} + "
                f"TV: {history['epoch_tv_loss'][-1]:.6f}) ---"
            )
        
        # Generate predictions
        logging.info(f"Generating full prediction matrices for Epoch {epoch+1}...")
        shared_cnn.eval()
        [h.eval() for h in predictor_heads.values()]
        
        inference_dataset = InferenceDataset(H, W, high_res_image, args.patch_size, h_scale, w_scale)
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size * 4,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        predicted_matrices = {}
        with torch.no_grad():
            full_predicted_matrices = {
                p_name: torch.zeros((H, W), device=device)
                for p_name in predictor_heads.keys()
            }
            
            for i, patches in enumerate(tqdm(inference_loader, desc="  - Extracting features and predicting")):
                start_idx = i * (args.batch_size * 4)
                end_idx = start_idx + len(patches)
                if len(patches) == 0:
                    continue
                
                multi_scale_features = shared_cnn(patches.to(device, non_blocking=True))
                for p_name, head in predictor_heads.items():
                    preds_batch = head(multi_scale_features).squeeze(-1)
                    for j, p in enumerate(preds_batch):
                        row = (start_idx + j) // W
                        col = (start_idx + j) % W
                        if row < H:
                            full_predicted_matrices[p_name][row, col] = p
            
            for p_name, full_matrix in full_predicted_matrices.items():
                final_matrix = full_matrix * shared_mask_tensor.to(device).float()
                scale_factor = next(
                    (d['scale_factor'] for d in all_protein_specific_data if d['name'] == p_name),
                    1.0
                )
                final_prediction = final_matrix.cpu().numpy() * scale_factor
                final_prediction[final_prediction < 0] = 0
                predicted_matrices[p_name] = final_prediction
        
        # Save checkpoint
        ckpt_data = {
            'epoch': epoch,
            'shared_cnn_state_dict': shared_cnn.state_dict(),
            'predictor_heads_state_dict': {n: h.state_dict() for n, h in predictor_heads.items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'predicted_matrices': predicted_matrices
        }
        torch.save(ckpt_data, os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pth"))
        
        # Plot loss curves
        if history['epoch_avg_loss']:
            plot_loss_curve(
                history['epoch_avg_loss'],
                os.path.join(args.output_dir, "loss_curve_total.png"),
                title='Total Training Loss'
            )
            plot_loss_curve(
                history['epoch_mae_loss'],
                os.path.join(args.output_dir, "loss_curve_mae.png"),
                title='MAE Reconstruction Loss'
            )
            plot_loss_curve(
                history['epoch_tv_loss'],
                os.path.join(args.output_dir, "loss_curve_tv.png"),
                title='TV Regularization Loss'
            )
    
    logging.info("--- Training completed. ---")
