import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import fftn, ifftn, fftshift, ifftshift

class FrequencyAwareModule(nn.Module):
    """Processes frequency domain data using convolutional layers."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(8, out_channels)

    def forward(self, x):
        x = F.silu(self.gn(self.conv1(x)))
        x = self.conv2(x)
        return x


class SpatialChannelAttention(nn.Module):
    """Applies spatial and channel attention to refine features."""
    def __init__(self, channels):
        super().__init__()
        self.spatial_attention = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // 8), kernel_size=1),  # Ensure at least 1 channel
            nn.SiLU(),
            nn.Conv2d(max(1, channels // 8), channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial attention
        spatial_weights = torch.sigmoid(self.spatial_attention(x))  # [B, 1, H, W]
        x = x * spatial_weights

        # Channel attention
        channel_weights = self.channel_attention(x)  # [B, C, 1, 1]
        x = x * channel_weights

        return x

class MultiScaleFusion(nn.Module):
    """Fuses features from different scales."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        fused = x1 + x3 + x5
        return fused
class Kspace_Net_MT(nn.Module):

    def __init__(self, act_dim, feature_dim, mt_shape, dropout=0.0):
        super().__init__()
        self.act_dim = act_dim
        self.mt_shape = mt_shape
        self.feature_dim = feature_dim
        self.dropout = dropout

        # Frequency-aware k-space processing
        self.kspace_conv = nn.Sequential(
            FrequencyAwareModule(2, 16),
            nn.MaxPool2d(2),
            FrequencyAwareModule(16, 32),
            nn.MaxPool2d(2)
        )

        # Image processing with spatial-channel attention
        self.image_conv = nn.Sequential(
            SpatialChannelAttention(1),
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.MaxPool2d(2),
            SpatialChannelAttention(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )

        # Multi-scale feature fusion
        self.fusion = MultiScaleFusion(64, 32)

        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(32768, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),
        )

        # Policy head
        self.policy_layer = nn.Sequential(
            nn.Linear(self.feature_dim + self.mt_shape[0], 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, self.act_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        kspace = input_dict['kspace'] + 1e-6
        mt = input_dict['mt']
        mt_vec = F.one_hot(mt, num_classes=self.mt_shape[0]).float()

        # Process k-space
        kspace_combined = torch.cat([kspace.real, kspace.imag], dim=1)
        kspace_features = self.kspace_conv(kspace_combined)

        # Process image
        image = ifftshift(ifftn(fftshift(kspace, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1)).abs()
        image_features = self.image_conv(image)

        # Fuse features
        combined_features = torch.cat([kspace_features, image_features], dim=1)
        fused_features = self.fusion(combined_features)

        features_flat = fused_features.reshape(fused_features.size(0), -1)
        # Trunk and policy
        h = self.trunk(features_flat)
        if len(mt_vec.shape) == 1:
            h_combined = torch.cat((h, mt_vec.repeat(h.shape[0], 1)), dim=-1)
        else:
            h_combined = torch.cat((h, mt_vec), dim=-1)

        return self.policy_layer(h_combined)

class Kspace_Net_Critic_MT(nn.Module):

    def __init__(self, feature_dim, mt_shape, dropout=0.0):
        super().__init__()
        self.mt_shape = mt_shape
        self.feature_dim = feature_dim
        self.dropout = dropout

        # Frequency-aware k-space processing
        self.kspace_conv = nn.Sequential(
            FrequencyAwareModule(2, 16),
            nn.MaxPool2d(2),
            FrequencyAwareModule(16, 32),
            nn.MaxPool2d(2)
        )

        # Image processing with spatial-channel attention
        self.image_conv = nn.Sequential(
            SpatialChannelAttention(1),
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(8, 16),
            nn.SiLU(),
            nn.MaxPool2d(2),
            SpatialChannelAttention(16),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.MaxPool2d(2)
        )

        # Multi-scale feature fusion
        self.fusion = MultiScaleFusion(64, 32)

        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(32768, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),
        )

        # Critic layer
        self.critic_layer = nn.Sequential(
            nn.Linear(self.feature_dim + self.mt_shape[0], 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        kspace = input_dict['kspace'] + 1e-6
        mt = input_dict['mt']
        mt_vec = F.one_hot(mt, num_classes=self.mt_shape[0]).float()

        # Process k-space
        kspace_combined = torch.cat([kspace.real, kspace.imag], dim=1)
        kspace_features = self.kspace_conv(kspace_combined)

        # Process image
        image = ifftshift(ifftn(fftshift(kspace, dim=(-2, -1)), dim=(-2, -1)), dim=(-2, -1)).abs()
        image_features = self.image_conv(image)

        # Fuse features
        combined_features = torch.cat([kspace_features, image_features], dim=1)
        fused_features = self.fusion(combined_features)

        features_flat = fused_features.reshape(fused_features.size(0), -1)
        # Trunk and policy
        h = self.trunk(features_flat)

        if len(mt_vec.shape) == 1:
            h_combined = torch.cat((h, mt_vec.repeat(h.shape[0], 1)), dim=-1)
        else:
            h_combined = torch.cat((h, mt_vec), dim=-1)

        # Compute value estimate
        value = self.critic_layer(h_combined).squeeze()

        return value