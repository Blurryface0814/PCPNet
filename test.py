#!/usr/bin/env python3
# Developed by Zhen Luo, Junyi Ma, and Zijie Zhou
# This file is covered by the LICENSE file in the root of the project PCPNet:
# https://github.com/Blurryface0814/PCPNet
# Brief: Test script for range-image-based point cloud prediction
import os
import time
import argparse
import random
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

import pcpnet.datasets.datasets as datasets
import pcpnet.models.PCPNet as PCPNet
from pcpnet.utils.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./test.py")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Model to be tested"
    )
    parser.add_argument(
        "--limit_test_batches",
        "-l",
        type=float,
        default=1.0,
        help="Percentage of test data to be tested",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        required=True,
        help="The path of processed KITTI odometry and SemanticKITTI dataset",
    )
    parser.add_argument(
        "--save",
        "-s",
        action="store_true",
        help="Save point clouds"
    )
    parser.add_argument(
        "--only_save_pc",
        "-o",
        action="store_true",
        help="Only save point clouds, not compute loss"
    )
    parser.add_argument(
        "--cd_downsample",
        type=int,
        default=-1,
        help="Number of downsampled points for evaluating Chamfer Distance",
    )
    parser.add_argument("--path", "-p", type=str, default=None, help="Path to data")
    parser.add_argument(
        "-seq",
        "--sequence",
        type=int,
        nargs="+",
        default=None,
        help="Sequence to be tested",
    )

    args, unparsed = parser.parse_known_args()
    dataset_path = args.dataset
    if dataset_path:
        pass
    else:
        raise Exception("Please enter the path of dataset")

    # load config file
    config_filename = os.path.dirname(os.path.dirname(os.path.dirname(args.model))) + "/hparams.yaml"
    cfg = yaml.safe_load(open(config_filename))
    print("Starting testing model ", cfg["LOG_NAME"])
    """Manually set these"""
    cfg["DATA_CONFIG"]["COMPUTE_MEAN_AND_STD"] = False
    cfg["DATA_CONFIG"]["GENERATE_FILES"] = False

    if args.only_save_pc:
        cfg["TEST"]["ONLY_SAVE_POINT_CLOUDS"] = args.only_save_pc
        print("Only save point clouds")
    else:
        cfg["TEST"]["SAVE_POINT_CLOUDS"] = args.save

    cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"] = args.cd_downsample
    print("Evaluating CD on ", cfg["TEST"]["N_DOWNSAMPLED_POINTS_CD"], " points.")

    if args.sequence:
        cfg["DATA_CONFIG"]["SPLIT"]["TEST"] = args.sequence
        cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"] = args.sequence
        cfg["DATA_CONFIG"]["SPLIT"]["VAL"] = args.sequence

    ###### Set random seed for torch, numpy and python
    set_seed(cfg["DATA_CONFIG"]["RANDOM_SEED"])
    print("Random seed is ", cfg["DATA_CONFIG"]["RANDOM_SEED"])

    data = datasets.KittiOdometryModule(cfg, dataset_path)
    data.setup()

    checkpoint_path = args.model
    cfg["TEST"]["USED_CHECKPOINT"] = checkpoint_path
    test_dir_name = "test_" + time.strftime("%Y%m%d_%H%M%S")
    cfg["TEST"]["DIR_NAME"] = test_dir_name

    model = PCPNet.PCPNet.load_from_checkpoint(checkpoint_path, cfg=cfg)

    # Only log if test is done on full data
    if args.limit_test_batches == 1.0 and not args.only_save_pc:
        logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg["LOG_DIR"],
            default_hp_metric=False,
            name=test_dir_name,
            version=""
        )
    else:
        logger = False

    trainer = Trainer(
        limit_test_batches=args.limit_test_batches,
        gpus=cfg["TRAIN"]["N_GPUS"],
        logger=logger,
    )

    results = trainer.test(model, data.test_dataloader())

    if logger:
        filename = os.path.join(
            cfg["LOG_DIR"], cfg["TEST"]["DIR_NAME"], "results" + ".yaml"
        )
        log_to_save = {**{"results": results}, **vars(args), **cfg}
        with open(filename, "w") as yaml_file:
            yaml.dump(log_to_save, yaml_file, default_flow_style=False)
