import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) with local feature fusion.
    """
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        self.growth_rate = growth_rate
        self.in_channels = in_channels

        for i in range(num_layers):
            self.layers.append(
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            )

        self.local_fusion = nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, kernel_size=1)

    def forward(self, x):
        inputs = x
        feature_maps = [x]
        for layer in self.layers:
            out = F.relu(layer(torch.cat(feature_maps, dim=1)))
            feature_maps.append(out)

        fused_features = self.local_fusion(torch.cat(feature_maps, dim=1))
        return fused_features + inputs


class DualAttentionBlock(nn.Module):
    """
    Dual Attention Block with Channel, Spatial, and Frequency Attention.
    """
    def __init__(self, in_channels):
        super(DualAttentionBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.frequency_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        ca = self.channel_attention(x) * x

        # Spatial Attention
        sa = self.spatial_attention(x) * x

        # Frequency Attention
        freq = torch.fft.fft2(x, dim=(-2, -1))
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        freq = torch.abs(freq)
        freq_attention = self.frequency_attention(freq)
        freq_attention = torch.fft.ifftshift(freq_attention, dim=(-2, -1))
        freq_attention = torch.fft.ifft2(freq_attention, dim=(-2, -1)).real
        fa = freq_attention * x

        return ca + sa + fa


class ResNetBlock(nn.Module):
    """
    Enhanced ResNet Block with Residual Dense Block and Dual Attention.
    """
    def __init__(self, in_channels=2, num_layers=4, num_filters=64, growth_rate=32):
        super(ResNetBlock, self).__init__()
        self.rdb = ResidualDenseBlock(num_filters, growth_rate=growth_rate, num_layers=num_layers)
        self.attention = DualAttentionBlock(num_filters)
        self.adjust_channels = nn.Conv2d(in_channels, num_filters, kernel_size=1, stride=1, padding=0) # Adjusting the number of filters for compatibility
        self.conv = nn.Conv2d(num_filters, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = x['input']

        # Adjust the input channels to match the number of filters
        adjusted_input = self.adjust_channels(residual)

        # Apply Residual Dense Block
        out = self.rdb(adjusted_input)

        # Apply Dual Attention
        out = self.attention(out)

        # Add back the residual connection
        out += adjusted_input

        out = self.conv(out)

        # Convert back to the original number of channels
        x['input'] = out
        return x

class DataConsistency(nn.Module):
    """
    Data Consistency Block with Learnable Weight.
    """
    def __init__(self):
        super(DataConsistency, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable parameter for consistency weight

    def forward(self, x):
        image = x['input'].permute(0, 2, 3, 1)  # Prepare for FFT
        kspace_pred = fastmri.fft2c(image)

        # Apply sampling mask
        kspace_combined = (1 - x['sampling_mask']) * kspace_pred + x['masked_kspace']

        # Transform back to image space
        image_reconstructed = fastmri.ifft2c(kspace_combined)
        image_reconstructed = image_reconstructed.permute(0, 3, 1, 2).float()

        x['input'] = image_reconstructed
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-Scale Feature Extractor with different kernel sizes.
    """
    def __init__(self, in_channels, num_filters=64):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, num_filters, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, num_filters, kernel_size=7, padding=3)

        self.fusion = nn.Conv2d(num_filters * 3, in_channels, kernel_size=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))

        out = torch.cat([out1, out2, out3], dim=1)
        out = self.fusion(out)
        return out


class CascadeNetwork(nn.Module):
    """
    Upgraded Cascade Network with Enhanced Blocks and Learnable Data Consistency.
    """
    def __init__(self, num_cascades=5, num_layers=4, num_filters=64, growth_rate=32):
        super(CascadeNetwork, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_cascades):
            self.blocks.append(ResNetBlock(2, num_layers, num_filters, growth_rate))
            self.blocks.append(DataConsistency())

    def forward(self, masked_kspace, sampling_mask):
        masked_kspace_cmp = torch.stack((masked_kspace.real, masked_kspace.imag), dim=-1)
        undersampled_img = fastmri.ifft2c(masked_kspace_cmp).permute(0, 3, 1, 2)
        x = {
            "input": undersampled_img,  # Normalized image-space data
            "sampling_mask": sampling_mask,  # Fully sampled k-space data
            "masked_kspace": masked_kspace_cmp,  # Cartesian sampling mask
        }
        for block in self.blocks:
            x = block(x)
        return fastmri.complex_abs(x["input"].permute(0, 2, 3, 1)).unsqueeze(1)

def build_reconstruction_model():
    kengal_model = CascadeNetwork().cuda()
    return kengal_model