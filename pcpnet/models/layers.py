#!/usr/bin/env python3
# Developed by Zhen Luo, Junyi Ma, and Zijie Zhou
# This file is covered by the LICENSE file in the root of the project PCPNet:
# https://github.com/Blurryface0814/PCPNet
# Brief: Implementation of Layers of PCPNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math


class CustomConv3d(nn.Module):
    """Custom 3D Convolution that enables circular padding along the width dimension only"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        bias=False,
        circular_padding=False,
    ):
        """Init custom 3D Conv with circular padding"""
        super().__init__()
        self.circular_padding = circular_padding
        self.padding = padding

        if self.circular_padding:
            # Only apply zero padding in time and height
            zero_padding = (self.padding[0], self.padding[1], 0)
        else:
            zero_padding = padding

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=zero_padding,
            padding_mode="zeros",
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        """Forward custom 3D convolution

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        if self.circular_padding:
            x = F.pad(
                x, (self.padding[2], self.padding[2], 0, 0, 0, 0), mode="circular"
            )
        x = self.conv(x)
        return x


class Normalization(nn.Module):
    """Custom Normalization layer to enable different normalization strategies"""

    def __init__(self, cfg, n_channels):
        """Init custom normalization layer"""
        super(Normalization, self).__init__()
        self.cfg = cfg
        self.norm_type = self.cfg["MODEL"]["NORM"]
        n_channels_per_group = self.cfg["MODEL"]["N_CHANNELS_PER_GROUP"]

        if self.norm_type == "batch":
            self.norm = nn.BatchNorm3d(n_channels)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(n_channels // n_channels_per_group, n_channels)
        elif self.norm_type == "instance":
            self.norm = nn.InstanceNorm3d(n_channels)
        elif self.norm_type == "none":
            self.norm = nn.Identity()

    def forward(self, x):
        """Forward normalization pass

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Output tensor
        """
        x = self.norm(x)
        return x


class DownBlock(nn.Module):
    """Downsamples the input tensor"""

    def __init__(
        self, cfg, in_channels, out_channels, down_stride_H=2, down_stride_W=4, skip=False
    ):
        """Init module"""
        super(DownBlock, self).__init__()
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.down_stride_H = down_stride_H
        self.down_stride_W = down_stride_W
        self.conv0 = CustomConv3d(
            in_channels,
            in_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm0 = Normalization(cfg, in_channels)
        self.channel_attention_0 = eca_block(in_channels, b=1, gamma=2)
        self.relu = nn.LeakyReLU()
        self.conv1 = CustomConv3d(
            in_channels,
            out_channels,
            kernel_size=(1, self.down_stride_H, self.down_stride_W),
            stride=(1, down_stride_H, down_stride_W),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm1 = Normalization(cfg, out_channels)
        self.channel_attention_1 = eca_block(out_channels, b=1, gamma=2)

    def forward(self, x):
        """Forward pass for downsampling

        Args:
            x (torch.tensor): Input tensor

        Returns:
            torch.tensor: Downsampled output tensor
        """
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.channel_attention_0(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.channel_attention_1(x)
        return x


class UpBlock(nn.Module):
    """Upsamples the input tensor using transposed convolutions"""

    def __init__(
        self, cfg, in_channels, out_channels, up_stride_H=2, up_stride_W=4, skip=False
    ):
        """Init module"""
        super(UpBlock, self).__init__()
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.up_stride_H = up_stride_H
        self.up_stride_W = up_stride_W
        if self.skip:
            self.conv_skip = CustomConv3d(
                2 * in_channels,
                in_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False,
                circular_padding=self.circular_padding,
            )
            self.norm_skip = Normalization(cfg, in_channels)
            self.channel_attention_skip = eca_block(in_channels, b=1, gamma=2)
        self.conv0 = nn.ConvTranspose3d(
            in_channels,
            in_channels,
            kernel_size=(1, self.up_stride_H, self.up_stride_W),
            stride=(1, self.up_stride_H, self.up_stride_W),
            bias=False,
        )
        self.norm0 = Normalization(cfg, in_channels)
        self.channel_attention_0 = eca_block(in_channels, b=1, gamma=2)
        self.relu = nn.LeakyReLU()
        self.conv1 = CustomConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False,
            circular_padding=self.circular_padding,
        )
        self.norm1 = Normalization(cfg, out_channels)
        self.channel_attention_1 = eca_block(out_channels, b=1, gamma=2)

    def forward(self, x, skip=None):
        """Forward pass for upsampling

        Args:
            x (torch.tensor): Input tensor
            skip (bool, optional): Use skip connection. Defaults to None.

        Returns:
            torch.tensor: Upsampled output tensor
        """
        if self.skip:
            x = torch.cat((x, skip), dim=1)
            x = self.conv_skip(x)
            x = self.norm_skip(x)
            x = self.relu(x)
            x = self.channel_attention_skip(x)
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.channel_attention_0(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.channel_attention_1(x)
        return x


class Transformer_W(nn.Module):
    """Transformer on W"""

    def __init__(self, cfg, channels, layers=2, skip=False):
        """Init module"""
        super(Transformer_W, self).__init__()
        self.cfg = cfg
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.layers = layers

        # [B, C, T, H=16, W=128]
        self.conv1 = DownBlock(cfg, channels, 2*channels, down_stride_H=4, down_stride_W=1, skip=skip)
        # 4, 128
        self.conv2 = DownBlock(cfg, 2*channels, 4*channels, down_stride_H=4, down_stride_W=1, skip=skip)
        # 1, 128

        self.convLast1 = nn.Conv1d(4*channels, 4*channels, kernel_size=1, stride=1, bias=False)
        self.bnLast1 = nn.BatchNorm1d(4*channels)
        self.relu = nn.ReLU(inplace=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=4*channels, nhead=4, dim_feedforward=1024,
                                                   activation='relu', dropout=0.1)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        self.convLast2 = nn.Conv1d(8*channels, 4*channels, kernel_size=1, stride=1, bias=False)
        self.bnLast2 = nn.BatchNorm1d(4*channels)

        # 1, 128
        self.conv1_up = UpBlock(cfg, 4*channels, 2*channels, up_stride_H=4, up_stride_W=1, skip=skip)
        # 4, 128
        self.conv2_up = UpBlock(cfg, 2*channels, channels, up_stride_H=4, up_stride_W=1, skip=skip)
        # 16, 128

    def forward(self, x):
        """Forward pass for transfomer_W

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: output tensor
        """
        # [B, C, T, H=16, W=128]
        skip_list = []
        if self.skip:
            x = self.conv1(x)
            skip_list.append(x.clone())
            x = self.conv2(x)
            skip_list.append(x.clone())
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        # [B, C, T, H=1, W=128]
        x = x.squeeze(3)                                                   # [B, C, T, W]
        B, _, T, W_out = x.shape
        x = torch.reshape(x, (B, x.shape[1], T * W_out))                   # [B, C, T*W]

        x_1 = self.relu(self.bnLast1(self.convLast1(x)))
        x = self.transformer_encoder(x.permute(2, 0, 1)).permute(1, 2, 0)  # [B, C, T*W]
        x = torch.cat((x_1, x), dim=1)
        x = self.relu(self.bnLast2(self.convLast2(x)))
        x = torch.reshape(x, (B, x.shape[1], T, W_out))                    # [B, C, T, W]
        x = x.unsqueeze(3)                                                 # [B, C, T, H, W]

        # [B, C, T, H=1, W=128]
        if self.skip:
            x = self.conv1_up(x, skip_list.pop())
            x = self.conv2_up(x, skip_list.pop())
        else:
            x = self.conv1_up(x)
            x = self.conv2_up(x)
        # [B, C, T, H=16, W=128]

        return x


class Transformer_H(nn.Module):
    """Transformer on H"""

    def __init__(self, cfg, channels, layers=2, skip=False):
        """Init module"""
        super(Transformer_H, self).__init__()
        self.cfg = cfg
        self.skip = skip
        self.circular_padding = cfg["MODEL"]["CIRCULAR_PADDING"]
        self.layers = layers

        # [B, C, T, H=16, W=128]
        self.conv1 = DownBlock(cfg, channels, 2*channels, down_stride_H=1, down_stride_W=4, skip=skip)
        # 16, 32
        self.conv2 = DownBlock(cfg, 2*channels, 2*channels, down_stride_H=1, down_stride_W=4, skip=skip)
        # 16, 8
        self.conv3 = DownBlock(cfg, 2*channels, 4*channels, down_stride_H=1, down_stride_W=8, skip=skip)
        # 16, 1

        self.convLast1 = nn.Conv1d(4*channels, 4*channels, kernel_size=1, stride=1, bias=False)
        self.bnLast1 = nn.BatchNorm1d(4*channels)
        self.relu = nn.ReLU(inplace=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=4*channels, nhead=4, dim_feedforward=1024,
                                                   activation='relu', dropout=0.1)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        self.convLast2 = nn.Conv1d(8*channels, 4*channels, kernel_size=1, stride=1, bias=False)
        self.bnLast2 = nn.BatchNorm1d(4*channels)

        # 16, 1
        self.conv1_up = UpBlock(cfg, 4*channels, 2*channels, up_stride_H=1, up_stride_W=8, skip=skip)
        # 16, 8
        self.conv2_up = UpBlock(cfg, 2*channels, 2*channels, up_stride_H=1, up_stride_W=4, skip=skip)
        # 16, 32
        self.conv3_up = UpBlock(cfg, 2*channels, channels, up_stride_H=1, up_stride_W=4, skip=skip)
        # 16, 128

    def forward(self, x):
        """Forward pass for transfomer_H

        Args:
            x (torch.tensor): Input tensor
        Returns:
            torch.tensor: output tensor
        """
        # [B, C, T, H=16, W=128]
        skip_list = []
        if self.skip:
            x = self.conv1(x)
            skip_list.append(x.clone())
            x = self.conv2(x)
            skip_list.append(x.clone())
            x = self.conv3(x)
            skip_list.append(x.clone())
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)

        # [B, C, T, H=16, W=1]
        x = x.squeeze(4)                                                   # [B, C, T, H]
        B, _, T, H_out = x.shape
        x = torch.reshape(x, (B, x.shape[1], T * H_out))                   # [B, C, T*H]

        x_1 = self.relu(self.bnLast1(self.convLast1(x)))
        x = self.transformer_encoder(x.permute(2, 0, 1)).permute(1, 2, 0)  # [B, C, T*H]
        x = torch.cat((x_1, x), dim=1)
        x = self.relu(self.bnLast2(self.convLast2(x)))
        x = torch.reshape(x, (B, x.shape[1], T, H_out))                    # [B, C, T, H]
        x = x.unsqueeze(4)                                                 # [B, C, T, H, W]

        # [B, C, T, H=16, W=1]
        if self.skip:
            x = self.conv1_up(x, skip_list.pop())
            x = self.conv2_up(x, skip_list.pop())
            x = self.conv3_up(x, skip_list.pop())
        else:
            x = self.conv1_up(x)
            x = self.conv2_up(x)
            x = self.conv3_up(x)
        # [B, C, T, H=16, W=128]

        return x


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=(kernel_size, 1),
                              padding=((kernel_size - 1) // 2, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.size()

        y = self.avg_pool(x).view([b, 1, c, t])
        y = self.conv(y)
        y = self.sigmoid(y).view([b, c, t, 1, 1])
        return x * y
