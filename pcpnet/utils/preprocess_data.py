#!/usr/bin/env python3
# Developed by Zhen Luo, Junyi Ma, and Zijie Zhou
# This file is covered by the LICENSE file in the root of the project PCPNet:
# https://github.com/Blurryface0814/PCPNet
# Brief: Preprocessing point cloud to range images
import os
import numpy as np
import torch

from pcpnet.utils.utils import load_files, range_projection


def prepare_data(cfg, dataset_path, rawdata_path):
    """Loads point clouds and labels and pre-processes them into range images

    Args:
        cfg (dict): Config
    """
    sequences = (
        cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"]
        + cfg["DATA_CONFIG"]["SPLIT"]["VAL"]
        + cfg["DATA_CONFIG"]["SPLIT"]["TEST"]
    )

    proj_H = cfg["DATA_CONFIG"]["HEIGHT"]
    proj_W = cfg["DATA_CONFIG"]["WIDTH"]

    for seq in sequences:
        seqstr = "{0:02d}".format(int(seq))
        scan_folder = os.path.join(rawdata_path, seqstr, "velodyne")
        label_folder = os.path.join(rawdata_path, seqstr, "labels")
        dst_folder = os.path.join(dataset_path, seqstr, "processed")
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        # Load LiDAR scan files and label files
        scan_paths = load_files(scan_folder)
        label_paths = load_files(label_folder)

        if len(scan_paths) != len(label_paths):
            print("Points files: ", len(scan_paths))
            print("Label files: ", len(label_paths))
            raise ValueError("Scan and Label don't contain same number of files")

        # Iterate over all scan files and label files
        for idx in range(len(scan_paths)):
            print(
                "Processing file {:d}/{:d} of sequence {:d}".format(
                    idx, len(scan_paths), seq
                )
            )

            # Load and project a point cloud
            current_vertex = np.fromfile(scan_paths[idx], dtype=np.float32)
            current_vertex = current_vertex.reshape((-1, 4))
            label = np.fromfile(label_paths[idx], dtype=np.int32)
            label = label.reshape((-1))
            # only fill in attribute if the right size
            if current_vertex.shape[0] != label.shape[0]:
                print("Points shape: ", current_vertex.shape[0])
                print("Label shape: ", label.shape[0])
                raise ValueError("Scan and Label don't contain same number of points")

            proj_range, proj_vertex, proj_intensity, proj_idx = range_projection(
                current_vertex,
                fov_up=cfg["DATA_CONFIG"]["FOV_UP"],
                fov_down=cfg["DATA_CONFIG"]["FOV_DOWN"],
                proj_H=cfg["DATA_CONFIG"]["HEIGHT"],
                proj_W=cfg["DATA_CONFIG"]["WIDTH"],
                max_range=cfg["DATA_CONFIG"]["MAX_RANGE"],
            )

            # Save range
            dst_path_range = os.path.join(dst_folder, "range")
            if not os.path.exists(dst_path_range):
                os.makedirs(dst_path_range)
            file_path = os.path.join(dst_path_range, str(idx).zfill(6))
            np.save(file_path, proj_range)

            # Save xyz
            dst_path_xyz = os.path.join(dst_folder, "xyz")
            if not os.path.exists(dst_path_xyz):
                os.makedirs(dst_path_xyz)
            file_path = os.path.join(dst_path_xyz, str(idx).zfill(6))
            np.save(file_path, proj_vertex)

            # Save intensity
            dst_path_intensity = os.path.join(dst_folder, "intensity")
            if not os.path.exists(dst_path_intensity):
                os.makedirs(dst_path_intensity)
            file_path = os.path.join(dst_path_intensity, str(idx).zfill(6))
            np.save(file_path, proj_intensity)

            # Save label
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16  # instance id in upper half
            # sanity check
            assert ((sem_label + (inst_label << 16) == label).all())

            proj_sem_label = np.zeros((proj_H, proj_W), dtype=np.int32)  # [H,W]  label
            mask = proj_idx >= 0
            proj_sem_label[mask] = sem_label[proj_idx[mask]]

            dst_path_label = os.path.join(dst_folder, "labels")
            if not os.path.exists(dst_path_label):
                os.makedirs(dst_path_label)
            file_path = os.path.join(dst_path_label, str(idx).zfill(6))
            np.save(file_path, proj_sem_label)


def compute_mean_and_std(cfg, train_loader):
    """Compute training data statistics

    Args:
        cfg (dict): Config
        train_loader (DataLoader): Pytorch DataLoader to access training data
    """
    n_channels = train_loader.dataset.n_channels
    mean = [0] * n_channels
    std = [0] * n_channels
    max = [0] * n_channels
    min = [0] * n_channels
    for i, data in enumerate(train_loader):
        past = data["past_data"]
        batch_size, n_channels, frames, H, W = past.shape

        for j in range(n_channels):
            channel = past[:, j, :, :, :].view(batch_size, 1, frames, H, W)
            mean[j] += torch.mean(channel[channel != -1.0]) / len(train_loader)
            std[j] += torch.std(channel[channel != -1.0]) / len(train_loader)
            max[j] += torch.max(channel[channel != -1.0]) / len(train_loader)
            min[j] += torch.min(channel[channel != -1.0]) / len(train_loader)

    print("Mean and standard deviation of training data:")
    for j in range(n_channels):
        print(
            "Input {:d}: Mean {:.3f}, std {:.3f}, min {:.3f}, max {:.3f}".format(
                j, mean[j], std[j], min[j], max[j]
            )
        )
