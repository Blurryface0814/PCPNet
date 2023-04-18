#!/usr/bin/env python3
# Developed by Zhen Luo, Junyi Ma, and Zijie Zhou
# This file is covered by the LICENSE file in the root of the project PCPNet:
# https://github.com/Blurryface0814/PCPNet
# Brief: Main Network Architecture of PCPNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math

from pcpnet.models.base import BasePredictionModel
from pcpnet.models.layers import CustomConv3d, DownBlock, UpBlock, Transformer_W, Transformer_H, eca_block


class PCPNet(BasePredictionModel):
    def __init__(self, cfg):
        """Init all layers needed for range image-based point cloud prediction"""
        super().__init__(cfg)
        self.channels = self.cfg["MODEL"]["CHANNELS"]
        self.skip_if_channel_size = self.cfg["MODEL"]["SKIP_IF_CHANNEL_SIZE"]
        self.circular_padding = self.cfg["MODEL"]["CIRCULAR_PADDING"]
        self.transformer_h_layers = self.cfg["MODEL"]["TRANSFORMER_H_LAYERS"]
        self.transformer_w_layers = self.cfg["MODEL"]["TRANSFORMER_W_LAYERS"]

        self.input_layer = CustomConv3d(
            self.n_inputs,
            self.channels[0],
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            bias=True,
            circular_padding=self.circular_padding,
        )

        self.DownLayers = nn.ModuleList()
        for i in range(len(self.channels) - 1):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.DownLayers.append(
                    DownBlock(
                        self.cfg,
                        self.channels[i],
                        self.channels[i + 1],
                        down_stride_H=2,
                        down_stride_W=4,
                        skip=True,
                    )
                )
            else:
                self.DownLayers.append(
                    DownBlock(
                        self.cfg,
                        self.channels[i],
                        self.channels[i + 1],
                        down_stride_H=2,
                        down_stride_W=4,
                        skip=False,
                    )
                )

        self.Transformer_H = Transformer_H(self.cfg, self.channels[-1],
                                           layers=self.transformer_h_layers, skip=True)
        self.Transformer_W = Transformer_W(self.cfg, self.channels[-1],
                                           layers=self.transformer_w_layers, skip=True)

        self.channel_attention = eca_block(2 * self.channels[-1], b=1, gamma=2)

        self.mid_layer = CustomConv3d(
            2 * self.channels[-1],
            self.channels[-1],
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            bias=True,
            circular_padding=self.circular_padding,
        )

        self.UpLayers = nn.ModuleList()
        for i in reversed(range(len(self.channels) - 1)):
            if self.channels[i + 1] in self.skip_if_channel_size:
                self.UpLayers.append(
                    UpBlock(
                        self.cfg,
                        self.channels[i + 1],
                        self.channels[i],
                        up_stride_H=2,
                        up_stride_W=4,
                        skip=True,
                    )
                )
            else:
                self.UpLayers.append(
                    UpBlock(
                        self.cfg,
                        self.channels[i + 1],
                        self.channels[i],
                        up_stride_H=2,
                        up_stride_W=4,
                        skip=False,
                    )
                )

        self.n_outputs = 2
        self.output_layer = CustomConv3d(
            self.channels[0],
            self.n_outputs,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            bias=True,
            circular_padding=self.circular_padding,
        )

    def forward(self, x):
        """Forward range image-based point cloud prediction

        Args:
            x (torch.tensor): Input tensor of concatenated, unnormalize range images

        Returns:
            dict: Containing the predicted range tensor and mask logits
        """
        # Only select inputs specified in base model
        x = x[:, self.inputs, :, :, :]
        batch_size, n_inputs, n_past_steps, H, W = x.size()
        assert n_inputs == self.n_inputs

        # Get mask of valid points
        past_mask = x != -1.0

        # Standardization and set invalid points to zero
        mean = self.mean[None, self.inputs, None, None, None]
        std = self.std[None, self.inputs, None, None, None]
        x = torch.true_divide(x - mean, std)
        x = x * past_mask

        skip_list = []
        x = x.view(batch_size, n_inputs, n_past_steps, H, W)   # [B, C, T, H, W]

        x = self.input_layer(x)
        for layer in self.DownLayers:
            x = layer(x)
            if layer.skip:
                skip_list.append(x.clone())

        # [B, C, T, H=16, W=128]
        x_h = self.Transformer_H(x)
        x_w = self.Transformer_W(x)
        x = torch.cat((x_h, x_w), dim=1)
        x = self.channel_attention(x)
        x = self.mid_layer(x)

        for layer in self.UpLayers:
            if layer.skip:
                x = layer(x, skip_list.pop())
            else:
                x = layer(x)

        x = self.output_layer(x)

        output = {}
        output["rv"] = self.min_range + nn.Sigmoid()(x[:, 0, :, :, :]) * (
            self.max_range - self.min_range
        )
        output["mask_logits"] = x[:, 1, :, :, :]

        return output
