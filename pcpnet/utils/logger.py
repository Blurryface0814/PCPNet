#!/usr/bin/env python3
# Developed by Zhen Luo, Junyi Ma, and Zijie Zhou
# This file is covered by the LICENSE file in the root of the project PCPNet:
# https://github.com/Blurryface0814/PCPNet
# Brief: Logging and saving point clouds and range images
import os
import torch
import matplotlib
import open3d as o3d
import numpy as np
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


point_size_config = {"material": {"cls": "PointsMaterial", "size": 0.1}}


def log_point_clouds(
    logger, projection, current_epoch, batch, output, sample_index, sequence, frame
):
    """Log point clouds to tensorboard

    Args:
        logger (TensorBoardLogger): Logger instance
        projection (projection): Projection instance
        current_epoch (int): Current epoch
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
        sample_index ([int): Selected sample in batch
        sequence (int): Selected dataset sequence
        frame (int): Selected frame number
    """
    for step in [0, 4]:
        gt_points, pred_points, gt_colors, pred_colors = get_pred_and_gt_point_cloud(
            projection, batch, output, step, sample_index
        )
        concat_points = torch.cat(
            (gt_points.view(1, -1, 3), pred_points.view(1, -1, 3)), 1
        )
        concat_colors = torch.cat((gt_colors, pred_colors), 1)
        logger.add_mesh(
            "prediction_and_gt_sequence_"
            + str(sequence)
            + "_frame_"
            + str(frame)
            + "_step_"
            + str(step),
            vertices=concat_points,
            colors=concat_colors,
            global_step=current_epoch,
            config_dict=point_size_config,
        )
        logger.add_mesh(
            "prediction_sequence_"
            + str(sequence)
            + "_frame_"
            + str(frame)
            + "_step_"
            + str(step),
            vertices=pred_points,
            colors=pred_colors,
            global_step=current_epoch,
            config_dict=point_size_config,
        )


def save_point_clouds(cfg, projection, batch, output):
    """Save ground truth and predicted point clouds as .ply

    Args:
        cfg (dict): Parameters
        projection (projection): Projection instance
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
    """
    batch_size, n_future_steps, H, W = output["rv"].shape
    seq, frame = batch["meta"]
    n_past_steps = cfg["MODEL"]["N_PAST_STEPS"]

    for sample_index in range(batch_size):
        current_global_frame = frame[sample_index].item()

        path_to_point_clouds = os.path.join(
            cfg["LOG_DIR"],
            cfg["TEST"]["DIR_NAME"],
            "point_clouds",
            str(seq[sample_index].item()).zfill(2),
        )
        gt_path = make_path(os.path.join(path_to_point_clouds, "gt"))
        pred_path = make_path(
            os.path.join(
                path_to_point_clouds, "pred", str(current_global_frame).zfill(6)
            )
        )

        for step in range(n_future_steps):
            predicted_global_frame = current_global_frame + step + 1
            (
                gt_points,
                pred_points,
                gt_colors,
                pred_colors,
            ) = get_pred_and_gt_point_cloud(
                projection, batch, output, step, sample_index
            )
            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = o3d.utility.Vector3dVector(
                gt_points.view(-1, 3).detach().cpu().numpy()
            )
            o3d.io.write_point_cloud(
                gt_path + "/" + str(predicted_global_frame).zfill(6) + ".ply", gt_pcd
            )

            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(
                pred_points.view(-1, 3).detach().cpu().numpy()
            )
            o3d.io.write_point_cloud(
                pred_path + "/" + str(predicted_global_frame).zfill(6) + ".ply",
                pred_pcd,
            )


def get_pred_and_gt_point_cloud(projection, batch, output, step, sample_index):
    """Extract GT and predictions from batch and output dicts

    Args:
        projection (projection): Projection instance
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
        step (int): Prediction step
        sample_index ([int): Selected sample in batch

    Returns:
        list: GT and predicted points and colors
    """
    future_range = batch["fut_data"][sample_index, 0, step, :, :]
    masked_prediction = projection.get_masked_range_view(output)
    gt_points = batch["fut_data"][sample_index, 1:4, step, :, :].permute(1, 2, 0)
    gt_points = gt_points[future_range > 0.0].view(1, -1, 3)
    gt_colors = torch.zeros(gt_points.view(1, -1, 3).shape)
    gt_colors[:, :, 0] = 255

    pred_points = projection.get_valid_points_from_range_view(
        masked_prediction[sample_index, step, :, :]
    ).view(1, -1, 3)
    pred_colors = torch.zeros(pred_points.view(1, -1, 3).shape)
    pred_colors[:, :, 2] = 255

    return gt_points, pred_points, gt_colors, pred_colors


def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


def to_color(label, learning_map_inv, color_map):
    # put label in original values
    label = map(label, learning_map_inv)
    # put label in color
    return map(label, color_map)


def save_range_mask_and_semantic(cfg,
                                 projection,
                                 batch,
                                 output,
                                 sample_index,
                                 sequence,
                                 frame,
                                 learning_map_inv,
                                 color_map,
                                 output_argmax,
                                 target_argmax,
                                 label,
                                 rand_t,
                                 test=False):
    """Saves GT and predicted range images and masks to a file

    Args:

        cfg (dict): Parameters
        projection (projection): Projection instance
        batch (dict): Batch containing past and future range images
        output (dict): Contains predicted mask logits and ranges
        sample_index ([int): Selected sample in batch
        sequence (int): Selected dataset sequence
        frame (int): Selected frame number
        learning_map_inv (dict): Learning map inv dict
        color_map (dict): Color map dict
        output_argmax (np.array): Semantic pred argmax
        target_argmax (np.array): Semantic target argmax
        label (np.array): Semantic label
        rand_t (str): Semantic frame number
        test (bool): Test mode or not
    """
    _, _, n_past_steps, H, W = batch["past_data"].shape
    _, _, n_future_steps, _, _ = batch["fut_data"].shape

    min_range = -1.0  # due to invalid points
    max_range = cfg["DATA_CONFIG"]["MAX_RANGE"]

    past_range = batch["past_data"][sample_index, 0, :, :, :].view(n_past_steps, H, W)
    future_range = batch["fut_data"][sample_index, 0, :, :, :].view(
        n_future_steps, H, W
    )

    pred_rv = output["rv"][sample_index, :, :, :].view(n_future_steps, H, W)

    # ########
    # pred_rv[future_range == -1.0] = -1.0

    # Get masks
    past_mask = projection.get_target_mask_from_range_view(past_range)
    future_mask = projection.get_target_mask_from_range_view(future_range)
    pred_mask = projection.get_mask_from_output(output)[sample_index, :, :, :].view(
        n_future_steps, H, W
    )
    pred_mask_binary = torch.zeros(pred_mask.shape).type_as(pred_mask)
    pred_mask_binary[pred_mask >= cfg["MODEL"]["MASK_THRESHOLD"]] = 1.0

    concat_pred_mask = torch.cat(
        (torch.zeros(past_mask.shape).type_as(past_mask), pred_mask), 0
    )
    concat_pred_mask_binary = torch.cat(
        (torch.zeros(past_mask.shape).type_as(past_mask), pred_mask_binary), 0
    )
    concat_gt_mask = torch.cat((past_mask, future_mask), 0)

    # Get normalized range views
    concat_gt_rv = torch.cat((past_range, future_range), 0)
    concat_gt_rv_normalized = (concat_gt_rv - min_range) / (max_range - min_range)

    concat_pred_rv = torch.cat(
        (torch.zeros(past_range.shape).type_as(past_range), pred_rv), 0
    )
    concat_pred_rv_normalized = (concat_pred_rv - min_range) / (max_range - min_range)

    # Combine mask and rv predition
    masked_prediction = projection.get_masked_range_view(output)[
        sample_index, :, :, :
    ].view(n_future_steps, H, W)
    concat_combined_pred_rv = torch.cat(
        (torch.zeros(past_range.shape).type_as(past_range), masked_prediction), 0
    )
    concat_combined_pred_rv_normalized = (concat_combined_pred_rv - min_range) / (
        max_range - min_range
    )

    # Get semantics
    gt_color = to_color(
        target_argmax[sample_index: sample_index + 1, :, :].astype(np.int32), learning_map_inv, color_map
    )
    pred_color = to_color(
        output_argmax[sample_index: sample_index + 1, :, :].astype(np.int32), learning_map_inv, color_map
    )
    label_color = to_color(
        label[sample_index: sample_index + 1, :, :].astype(np.int32), learning_map_inv, color_map
    )

    gt_color = torch.tensor(gt_color).type_as(past_mask).repeat(n_future_steps, 1, 1, 1).cuda()
    pred_color = torch.tensor(pred_color).type_as(past_mask).repeat(n_future_steps, 1, 1, 1).cuda()
    label_color = torch.tensor(label_color).type_as(past_mask).repeat(n_future_steps, 1, 1, 1).cuda()

    past_zero = torch.zeros((n_past_steps, H, W, 3)).type_as(past_mask)

    concat_gt_semantics = torch.cat((past_zero, gt_color), 0) / 255
    concat_pred_semantics = torch.cat((past_zero, pred_color), 0) / 255
    concat_label_semantics = torch.cat((past_zero, label_color), 0) / 255

    index = [2, 1, 0]
    concat_gt_semantics = concat_gt_semantics[:, :, :, index]
    concat_pred_semantics = concat_pred_semantics[:, :, :, index]
    concat_label_semantics = concat_label_semantics[:, :, :, index]

    for s in range(n_past_steps + n_future_steps):
        step = "{0:02d}".format(s)

        # Save rv and mask predictions
        if not test:
            path = make_path(
                os.path.join(
                    cfg["LOG_DIR"], "range_view_predictions", str(sequence), str(frame)
                )
            )
        else:
            path = make_path(
                os.path.join(
                    cfg["LOG_DIR"], cfg["TEST"]["DIR_NAME"], "range_view_predictions", str(sequence), str(frame)
                )
            )

        ratio = 12 * H / W
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        fig, axs = plt.subplots(9, 1, sharex=True, figsize=(30, 30 * ratio))
        axs[0].imshow(concat_gt_rv_normalized[s, :, :].cpu().detach().numpy())
        axs[0].text(
            0.01,
            0.8,
            "GT RV",
            transform=axs[0].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[1].imshow(
            concat_combined_pred_rv_normalized[s, :, :].cpu().detach().numpy()
        )
        axs[1].text(
            0.01,
            0.8,
            "Pred Comb",
            transform=axs[1].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[2].imshow(concat_pred_rv_normalized[s, :, :].cpu().detach().numpy())
        axs[2].text(
            0.01,
            0.8,
            "Pred RV",
            transform=axs[2].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[3].imshow(concat_gt_mask[s, :, :].cpu().detach().numpy())
        axs[3].text(
            0.01,
            0.8,
            "GT Mask",
            transform=axs[3].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[4].imshow(concat_pred_mask[s, :, :].cpu().detach().numpy())
        axs[4].text(
            0.01,
            0.8,
            "Pred Mask",
            transform=axs[4].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[5].imshow(concat_pred_mask_binary[s, :, :].cpu().detach().numpy())
        axs[5].text(
            0.01,
            0.8,
            "Pred Mask Binary",
            transform=axs[5].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[6].imshow(concat_label_semantics[s, :, :].cpu().detach().numpy())
        axs[6].text(
            0.01,
            0.8,
            "Label Semantics Future Frame " + rand_t,
            transform=axs[6].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[7].imshow(concat_gt_semantics[s, :, :].cpu().detach().numpy())
        axs[7].text(
            0.01,
            0.8,
            "GT Semantics Future Frame " + rand_t,
            transform=axs[7].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[8].imshow(concat_pred_semantics[s, :, :].cpu().detach().numpy())
        axs[8].text(
            0.01,
            0.8,
            "Pred Semantics Future Frame " + rand_t,
            transform=axs[8].transAxes,
            fontsize=14,
            verticalalignment="top",
            bbox=props,
        )

        axs[0].set_title(
            "Step "
            + step
            + " of sequence "
            + str(sequence)
            + " from frame "
            + str(frame)
        )
        plt.savefig(path + "/" + step + ".png", bbox_inches="tight", transparent=False)
        plt.close(fig)
