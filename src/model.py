#!/usr/bin/env python3
"""
Model definitions for CNN-based protein reconstruction.
Contains the MultiScalePatchCNN, GatedProteinHead, and PearsonCorrelationLoss.
"""
import torch
import torch.nn as nn


class PearsonCorrelationLoss(nn.Module):
    """Pearson correlation coefficient loss function."""
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        """Compute Pearson correlation loss between x and y."""
        x = x.float()
        y = y.float()
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + self.eps)
        return 1 - corr


class MultiScalePatchCNN(nn.Module):
    """Multi-scale patch CNN that outputs features at different scales."""
    
    def __init__(self, in_channels=4, patch_size=25, common_dim=256):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, common_dim, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(common_dim),
            nn.SiLU()
        )
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
    """Gated protein head with feature mixing, SE attention and predictor."""
    
    def __init__(self, in_features=256, reduction=16):
        super().__init__()
        self.gating_network = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.LeakyReLU(negative_slope=0.01)
        )
    
    def forward(self, multi_scale_features):
        f_shallow = multi_scale_features['shallow']
        f_mid = multi_scale_features['mid']
        f_deep = multi_scale_features['deep']
        gates = self.gating_network(f_deep)
        g1, g2, g3 = gates[:, 0:1], gates[:, 1:2], gates[:, 2:3]
        f_personalized = g1 * f_shallow + g2 * f_mid + g3 * f_deep
        recalibrated_features = f_personalized * self.attention(f_personalized)
        return self.predictor(recalibrated_features)
