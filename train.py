#!/usr/bin/env python3
# Developed by Zhen Luo, Junyi Ma, and Zijie Zhou
# This file is covered by the LICENSE file in the root of the project PCPNet:
# https://github.com/Blurryface0814/PCPNet
# Brief: Train script for range-image-based point cloud prediction
import os
import time
import argparse
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pcpnet.datasets.datasets import KittiOdometryModule
from pcpnet.models.PCPNet import PCPNet
from pcpnet.utils.utils import set_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser("./train.py")
    parser.add_argument(
        "--comment", "-c", type=str, default="", help="Add a comment to the LOG ID."
    )
    parser.add_argument(
        "-res",
        "--resume",
        type=str,
        default=None,
        help="Resume training from specified model.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="Init model with weights from specified model",
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
        "-raw",
        "--rawdata",
        type=str,
        default=None,
        help="The path of raw KITTI odometry and SemanticKITTI dataset",
    )
    parser.add_argument(
        "-r",
        "--range",
        type=float,
        default=None,
        help="Change weight of range image loss.",
    )
    parser.add_argument(
        "-m", "--mask", type=float, default=None, help="Change weight of mask loss."
    )
    parser.add_argument(
        "-cd",
        "--chamfer",
        type=float,
        default=None,
        help="Change weight of Chamfer distance loss.",
    )
    parser.add_argument(
        "-s",
        "--semantic",
        type=float,
        default=None,
        help="Change weight of semantic loss.",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=None, help="Number of training epochs."
    )
    parser.add_argument(
        "-seq",
        "--sequence",
        type=int,
        nargs="+",
        default=None,
        help="Sequences for training.",
    )
    parser.add_argument(
        "-u",
        "--update-cfg",
        type=bool,
        default=False,
        help="Update config file.",
    )
    args, unparsed = parser.parse_known_args()

    dataset_path = args.dataset
    if dataset_path:
        pass
    else:
        raise Exception("Please enter the path of dataset")

    model_path = args.resume if args.resume else args.weights
    if model_path and not args.update_cfg:
        ###### Load config and update parameters
        checkpoint_path = model_path
        config_filename = os.path.dirname(model_path)
        if os.path.basename(config_filename) == "val":
            config_filename = os.path.dirname(config_filename)
        config_filename = os.path.dirname(config_filename) + "/hparams.yaml"

        cfg = yaml.safe_load(open(config_filename))

        if args.weights and not args.comment:
            args.comment = "_pretrained"

        cfg["LOG_DIR"] = cfg["LOG_DIR"] + args.comment
        cfg["LOG_NAME"] = cfg["LOG_NAME"] + args.comment
        print("New log name is ", cfg["LOG_DIR"])

        """Manually set these"""
        cfg["DATA_CONFIG"]["COMPUTE_MEAN_AND_STD"] = False
        cfg["DATA_CONFIG"]["GENERATE_FILES"] = False

        if args.epochs:
            cfg["TRAIN"]["MAX_EPOCH"] = args.epochs
            print("Set max_epochs to ", args.epochs)
        if args.range:
            cfg["TRAIN"]["LOSS_WEIGHT_RANGE_VIEW"] = args.range
            print("Overwriting LOSS_WEIGHT_RANGE_VIEW =", args.range)
        if args.mask:
            cfg["TRAIN"]["LOSS_WEIGHT_MASK"] = args.mask
            print("Overwriting LOSS_WEIGHT_MASK =", args.mask)
        if args.chamfer:
            cfg["TRAIN"]["LOSS_WEIGHT_CHAMFER_DISTANCE"] = args.chamfer
            print("Overwriting LOSS_WEIGHT_CHAMFER_DISTANCE =", args.chamfer)
        if args.semantic:
            cfg["TRAIN"]["LOSS_WEIGHT_SEMANTIC"] = args.semantic
            print("Overwriting LOSS_WEIGHT_SEMANTIC =", args.semantic)
        if args.sequence:
            cfg["DATA_CONFIG"]["SPLIT"]["TRAIN"] = args.sequence
            print("Training on sequences ", args.sequence)
    else:
        ###### Create new log
        resume_from_checkpoint = None
        config_filename = "config/parameters.yaml"
        cfg = yaml.safe_load(open(config_filename))
        if args.update_cfg:
            checkpoint_path = model_path
            print("Updated config file manually")
        if args.comment:
            cfg["EXPERIMENT"]["ID"] = args.comment
        cfg["LOG_NAME"] = cfg["EXPERIMENT"]["ID"] + "_" + time.strftime("%Y%m%d_%H%M%S")
        cfg["LOG_DIR"] = os.path.join("./runs", cfg["LOG_NAME"])
        if not os.path.exists(cfg["LOG_DIR"]):
            os.makedirs(cfg["LOG_DIR"])
        print("Starting experiment with log name", cfg["LOG_NAME"])

    model_file_path = "./pcpnet/models"
    os.system('cp -r %s %s' % (model_file_path, cfg["LOG_DIR"]))

    ###### Set random seed for torch, numpy and python
    set_seed(cfg["DATA_CONFIG"]["RANDOM_SEED"])

    ###### Logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg["LOG_DIR"], default_hp_metric=False, name="", version=""
    )

    ###### Dataset
    data = KittiOdometryModule(cfg, dataset_path, args.rawdata)

    ###### Model
    model = PCPNet(cfg)

    ###### Load checkpoint
    if args.resume:
        resume_from_checkpoint = checkpoint_path
        print("Resuming from checkpoint ", checkpoint_path)
    elif args.weights:
        model = model.load_from_checkpoint(checkpoint_path, cfg=cfg)
        resume_from_checkpoint = None
        print("Loading weigths from ", checkpoint_path)

    ###### Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(cfg["LOG_DIR"], "checkpoints"),
        filename="{val/loss:.3f}-{epoch:02d}",
        mode="min",
        save_top_k=5,
        save_last=True
    )

    ###### Trainer
    trainer = Trainer(
        gpus=cfg["TRAIN"]["N_GPUS"],
        logger=tb_logger,
        accumulate_grad_batches=cfg["TRAIN"]["BATCH_ACC"],
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"],
        log_every_n_steps=cfg["TRAIN"][
            "LOG_EVERY_N_STEPS"
        ],  # times accumulate_grad_batches
        callbacks=[lr_monitor, checkpoint],
    )

    ###### Training
    trainer.fit(model, data, ckpt_path=resume_from_checkpoint)
