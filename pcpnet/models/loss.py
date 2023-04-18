# Developed by Zhen Luo, Junyi Ma, and Zijie Zhou
# This file is covered by the LICENSE file in the root of the project PCPNet:
# https://github.com/Blurryface0814/PCPNet
# Brief: Implementation of the Loss Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import math
import random

from pyTorchChamferDistance.chamfer_distance import ChamferDistance
from pcpnet.utils.projection import projection
from pcpnet.utils.logger import map
import importlib
import numpy as np


class Loss(nn.Module):
    """Combined loss for point cloud prediction"""

    def __init__(self, cfg):
        """Init"""
        super().__init__()
        self.cfg = cfg
        self.n_future_steps = self.cfg["MODEL"]["N_FUTURE_STEPS"]
        self.loss_weight_cd = self.cfg["TRAIN"]["LOSS_WEIGHT_CHAMFER_DISTANCE"]
        self.loss_weight_rv = self.cfg["TRAIN"]["LOSS_WEIGHT_RANGE_VIEW"]
        self.loss_weight_mask = self.cfg["TRAIN"]["LOSS_WEIGHT_MASK"]
        self.loss_weight_semantic = self.cfg["TRAIN"]["LOSS_WEIGHT_SEMANTIC"]
        self.cd_every_n_val_epoch = self.cfg["TRAIN"]["CHAMFER_DISTANCE_EVERY_N_VAL_EPOCH"]

        self.loss_range = loss_range(self.cfg)
        self.chamfer_distance = chamfer_distance(self.cfg)
        self.loss_mask = loss_mask(self.cfg)
        self.loss_semantic = loss_semantic(self.cfg)

    def forward(self, output, target, mode, current_epoch=None):
        """Forward pass with multiple loss components

        Args:
        output (dict): Predicted mask logits and ranges
        target (torch.tensor): Target range image
        mode (str): Mode (train,val,test)

        Returns:
        dict: Dict with loss components
        """
        cd_flag = ((current_epoch + 1) % self.cd_every_n_val_epoch == 0)

        target_range_image = target[:, 0, :, :, :]
        target_label = target[:, 5, :, :, :]

        # Range view
        loss_range_view = self.loss_range(output, target_range_image)

        # Mask
        loss_mask = self.loss_mask(output, target_range_image)

        # Semantic
        # if self.loss_weight_semantic > 0.0 or mode == "val" or mode == "test":
        if self.loss_weight_semantic > 0.0:
            loss_semantic, output_argmax, target_argmax, label, rand_t, \
            similarity_output, similarity_target = self.loss_semantic(
                output, target_range_image, target_label, mode
            )
        else:
            loss_semantic = torch.zeros(1).type_as(target_range_image)
            B, T, H, W = target_range_image.shape
            output_argmax = torch.zeros((B, 1, H, W)).type_as(target_range_image)
            target_argmax = output_argmax
            label = output_argmax
            rand_t = "none"
            similarity_output = torch.zeros(1).type_as(target_range_image)
            similarity_target = torch.zeros(1).type_as(target_range_image)

            # Chamfer Distance
        if self.loss_weight_cd > 0.0 or (mode == "val" and cd_flag) or mode == "test":
            chamfer_distance, chamfer_distances_tensor = self.chamfer_distance(
                output, target, self.cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"]
            )
            loss_chamfer_distance = sum([cd for cd in chamfer_distance.values()]) / len(
                chamfer_distance
            )
            detached_chamfer_distance = {
                step: cd.detach() for step, cd in chamfer_distance.items()
            }
        else:
            chamfer_distance = dict(
                (step, torch.zeros(1).type_as(target_range_image))
                for step in range(self.n_future_steps)
            )
            chamfer_distances_tensor = torch.zeros(self.n_future_steps, 1)
            loss_chamfer_distance = torch.zeros_like(loss_range_view)
            detached_chamfer_distance = chamfer_distance

        loss = (
            self.loss_weight_cd * loss_chamfer_distance
            + self.loss_weight_rv * loss_range_view
            + self.loss_weight_mask * loss_mask
            + self.loss_weight_semantic * loss_semantic
        )

        loss_dict = {
            "loss": loss,
            "chamfer_distance": detached_chamfer_distance,
            "chamfer_distances_tensor": chamfer_distances_tensor.detach(),
            "mean_chamfer_distance": loss_chamfer_distance.detach(),
            "final_chamfer_distance": chamfer_distance[
                self.n_future_steps - 1
            ].detach(),
            "loss_range_view": loss_range_view.detach(),
            "loss_mask": loss_mask.detach(),
            "loss_semantic": loss_semantic.detach()
        }

        if mode == "val" or mode == "test":
            return loss_dict, \
                   self.loss_semantic.learning_map_inv, \
                   self.loss_semantic.color_map, \
                   output_argmax, \
                   target_argmax, \
                   label, \
                   str(rand_t), \
                   similarity_output, \
                   similarity_target

        return loss_dict


class loss_mask(nn.Module):
    """Binary cross entropy loss for prediction of valid mask"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.projection = projection(self.cfg)

    def forward(self, output, target_range_view):
        target_mask = self.projection.get_target_mask_from_range_view(target_range_view)
        loss = self.loss(output["mask_logits"], target_mask)
        return loss


class loss_range(nn.Module):
    """L1 loss for range image prediction"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = nn.L1Loss(reduction="mean")

    def forward(self, output, target_range_image):
        # Do not count L1 loss for invalid GT points
        gt_masked_output = output["rv"].clone()
        gt_masked_output[target_range_image == -1.0] = -1.0

        loss = self.loss(gt_masked_output, target_range_image)
        return loss


class loss_semantic(nn.Module):
    """Semantic loss for prediction of rv"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.semantic_net = self.cfg["MODEL"]["SEMANTIC_NET"]
        self.semantic_data_config = self.cfg["MODEL"]["SEMANTIC_DATA_CONFIG"]
        self.semantic_pretrained = self.cfg["MODEL"]["SEMANTIC_PRETRAINED_MODEL"]
        self.projection = projection(self.cfg)

        pretrained = "semantic_net/" + self.semantic_net + "/model/" + self.semantic_pretrained
        config_base_path = "semantic_net/" + self.semantic_net + "/tasks/semantic/config/"
        arch_config = pretrained + "/arch_cfg.yaml"
        data_config = config_base_path + "labels/" + self.semantic_data_config

        self.DATA = yaml.safe_load(open(data_config, 'r'))
        self.n_classes = len(self.DATA["learning_map_inv"])
        self.learning_map_inv = self.DATA["learning_map_inv"]
        self.learning_map = self.DATA["learning_map"]
        self.color_map = self.DATA["color_map"]

        module = "semantic_net." + self.semantic_net + ".tasks.semantic.modules.segmentator"
        segmentator_module = importlib.import_module(module)

        self.loss = nn.L1Loss(reduction="mean")

        if cfg["TRAIN"]["LOSS_WEIGHT_SEMANTIC"] > 0.0:
            self.ARCH = yaml.safe_load(open(arch_config, 'r'))
            self.sensor = self.ARCH["dataset"]["sensor"]
            self.sensor_img_means = torch.tensor(self.sensor["img_means"], dtype=torch.float)
            self.sensor_img_stds = torch.tensor(self.sensor["img_stds"], dtype=torch.float)
            with torch.no_grad():
                self.semantic_model = segmentator_module.Segmentator(self.ARCH, self.n_classes, pretrained)
                self.semantic_model.eval()

        # Semantic Similarity
        self.criterion = nn.NLLLoss()
        self.semantic_similarity = self.cfg["TEST"]["SEMANTIC_SIMILARITY"]

    def forward(self, output, target_range_image, target_label, mode=None):
        B, T, H, W = output["rv"].shape
        sensor_img_means = self.sensor_img_means.type_as(target_range_image)
        sensor_img_stds = self.sensor_img_stds.type_as(target_range_image)

        label = target_label
        output = output["rv"].unsqueeze(2)  # [B, T, 1, H, W]
        target = target_range_image.unsqueeze(2)

        output_mask = output != -1.0
        target_mask = target != -1.0

        output_masked = (output - sensor_img_means[None, None, 0:1, None, None]
                         ) / sensor_img_stds[None, None, 0:1, None, None]

        target_masked = (target - sensor_img_means[None, None, 0:1, None, None]
                         ) / sensor_img_stds[None, None, 0:1, None, None]

        output_masked = output_masked * output_mask
        target_masked = target_masked * target_mask

        rand_t = random.randint(0, T-1)
        output_t = self.semantic_model(output_masked[:, rand_t, :, :, :])   # [B, 20, H, W]
        target_t = self.semantic_model(target_masked[:, rand_t, :, :, :])
        label = label[:, rand_t, :, :]

        # get labels from semantic model output
        output_argmax = output_t.argmax(dim=1)         # [B, H, W]
        target_argmax = target_t.argmax(dim=1)
        label = map(label.cpu().numpy().astype(np.int32), self.learning_map)
        label = torch.tensor(label).type_as(target_label)

        # Do not count L1 loss for invalid GT points
        target_mask = target[:, rand_t, :, :, :].repeat(1, 20, 1, 1)
        output_last = output_t.clone()
        target_last = target_t.clone()
        output_last[target_mask == -1.0] = -1.0
        target_last[target_mask == -1.0] = -1.0

        # set argmax image for visualization
        output_argmax[target_last[:, 0, :, :] == -1.0] = 0.0
        target_argmax[target_last[:, 0, :, :] == -1.0] = 0.0

        # Semantic Similarity
        if mode == "test" and self.semantic_similarity:
            similarity_output = self.n_classes / self.criterion(torch.log(output_t.clamp(min=1e-8)), label.long())
            similarity_target = self.n_classes / self.criterion(torch.log(target_t.clamp(min=1e-8)), label.long())
        else:
            similarity_output = torch.zeros(1).type_as(output_t)
            similarity_target = torch.zeros(1).type_as(output_t)

        loss = self.loss(output_last, target_last)

        return loss, output_argmax.cpu().numpy(), target_argmax.cpu().numpy(), \
               label.cpu().numpy(), rand_t, similarity_output, similarity_target


class chamfer_distance(nn.Module):
    """Chamfer distance loss. Additionally, the implementation allows the evaluation
    on downsampled point cloud (this is only for comparison to other methods but not recommended,
    because it is a bad approximation of the real Chamfer distance.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss = ChamferDistance()
        self.projection = projection(self.cfg)

    def forward(self, output, target, n_samples):
        batch_size, n_future_steps, H, W = output["rv"].shape
        masked_output = self.projection.get_masked_range_view(output)
        chamfer_distances = {}
        chamfer_distances_tensor = torch.zeros(n_future_steps, batch_size)
        for s in range(n_future_steps):
            chamfer_distances[s] = 0
            for b in range(batch_size):
                output_points = self.projection.get_valid_points_from_range_view(
                    masked_output[b, s, :, :]
                ).view(1, -1, 3)
                target_points = target[b, 1:4, s, :, :].permute(1, 2, 0)
                target_points = target_points[target[b, 0, s, :, :] > 0.0].view(
                    1, -1, 3
                )

                if n_samples != -1:
                    n_output_points = output_points.shape[1]
                    n_target_points = target_points.shape[1]

                    sampled_output_indices = random.sample(
                        range(n_output_points), n_samples
                    )
                    sampled_target_indices = random.sample(
                        range(n_target_points), n_samples
                    )

                    output_points = output_points[:, sampled_output_indices, :]
                    target_points = target_points[:, sampled_target_indices, :]

                dist1, dist2 = self.loss(output_points, target_points)
                dist_combined = torch.mean(dist1) + torch.mean(dist2)
                chamfer_distances[s] += dist_combined
                chamfer_distances_tensor[s, b] = dist_combined
            chamfer_distances[s] = chamfer_distances[s] / batch_size
        return chamfer_distances, chamfer_distances_tensor
