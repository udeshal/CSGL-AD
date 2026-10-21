import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Net3(nn.Module):
    def __init__(self, in_channels, num_classes=1, kernels=64):
        super(Net3, self).__init__()
        self.kernels = kernels

        # Fine Scale Branch
        self.fine_block1 = ConvBlock(in_channels, kernels, kernel_size=3)
        self.fine_block2 = ConvBlock(kernels, kernels * 2, kernel_size=3)
        self.fine_pool = nn.AdaptiveAvgPool1d(1)

        # Mid Scale Branch
        self.mid_block1 = ConvBlock(in_channels, kernels, kernel_size=6)
        self.mid_block2 = ConvBlock(kernels, kernels * 2, kernel_size=6)
        self.mid_pool = nn.AdaptiveAvgPool1d(1)

        # Coarse Scale Branch
        self.coarse_block1 = ConvBlock(in_channels, kernels, kernel_size=12)
        self.coarse_block2 = ConvBlock(kernels, kernels * 2, kernel_size=12)
        self.coarse_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        print('x-----: ', x.shape)
        x_permuted = x.permute(0, 2, 1) # [Batch, Channel, Time]

        # Fine Scale Path
        x_fine = self.fine_block1(x_permuted)
        x_fine = self.fine_block2(x_fine)
        cov1 = self.fine_pool(x_fine).squeeze(-1)

        # Mid Scale Path
        x_mid = self.mid_block1(x_permuted)
        x_mid = self.mid_block2(x_mid)
        cov2 = self.mid_pool(x_mid).squeeze(-1)

        # Coarse Scale Path
        x_coarse = self.coarse_block1(x_permuted)
        x_coarse = self.coarse_block2(x_coarse)
        cov3 = self.coarse_pool(x_coarse).squeeze(-1)
        print ('COV1 ---:', cov1.shape)
        print ('COV2 ---:', cov2.shape)
        print ('COV3 ---:', cov3.shape)

        x = torch.cat([cov1, cov2, cov3], dim=0)  # feed forward

        return x