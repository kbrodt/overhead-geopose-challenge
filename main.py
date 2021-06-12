import argparse
import os
import numpy as np
from utilities.ml_utils import train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--augmentation", action="store_true", help="whether or not to use augmentation"
    )
    parser.add_argument("--train", action="store_true", help="train model")
    parser.add_argument("--test", action="store_true", help="generate test predictions")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=1
    )
    parser.add_argument(
        "--num-epochs", type=int, help="number of epochs to train", default=205
    )
    parser.add_argument(
        "--save-period",
        type=int,
        help="epoch frequency to save model checkpoints",
        default=1,
    )
    parser.add_argument("--save-best", action="store_true", help="save best weights")
    parser.add_argument(
        "--val-period",
        type=int,
        help="epoch frequency for running validation (zero if none)",
        default=1,
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=4)
    parser.add_argument(
        "--downsample",
        type=int,
        help="factor for downsampling image at test time",
        default=1,
    )
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--predictions-dir", type=str, default="./predictions")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Default is most recent in checkpoint dir",
        default=None,
    )
    parser.add_argument(
        "--dataset-dir", type=str, help="dataset directory", default="./dataset"
    )
    parser.add_argument(
        "--train-sub-dir",
        type=str,
        help="train folder within datset-dir",
        default="train",
    )
    parser.add_argument(
        "--train-path-df",
        type=str,
        help="path to train df",
        default="geopose_train.csv",
    )
    parser.add_argument(
        "--test-sub-dir", type=str, help="test folder within datset-dir", default="test"
    )
    parser.add_argument(
        "--valid-sub-dir",
        type=str,
        help="validation folder within datset-dir",
        default="valid",
    )
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--T-max", type=int, default=5)
    parser.add_argument(
        "--sample-size",
        type=int,
        help="number of images to randomly sample for training",
        default=None,
    )
    parser.add_argument("--agl-weight", type=float, help="agl loss weight", default=1)
    parser.add_argument("--mag-weight", type=float, help="mag loss weight", default=2)
    parser.add_argument(
        "--angle-weight", type=float, help="angle loss weight", default=10
    )
    parser.add_argument(
        "--scale-weight", type=float, help="scale loss weight", default=10
    )
    parser.add_argument(
        "--rgb-suffix", type=str, help="suffix for rgb files", default="j2k"
    )
    parser.add_argument(
        "--nan-placeholder",
        type=int,
        help="placeholder value for nans. use 0 for no placeholder",
        default=65535,
    )
    parser.add_argument(
        "--unit",
        type=str,
        help="unit of AGLS (m, cm) -- converted inputs are in cm, downsampled data is in m",
        default="cm",
    )
    parser.add_argument(
        "--convert-predictions-to-cm-and-compress",
        type=bool,
        help="Whether to process predictions by converting to cm and compressing",
        default=True,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        help="random seed",
        default=314159,
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        help="number of folds",
        default=10,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="fold",
        default=0,
    )
    parser.add_argument(
        "--tta",
        type=int,
        help="test time augmentation",
        default=1,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="fold",
        default=0,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu",
        default=0,
    )
    parser.add_argument("--distributed", action="store_true", help="distributed training")
    parser.add_argument("--deterministic", action="store_true", help="deterministic training")
    parser.add_argument("--resume", type=str, default="", help="path to pretrained model to resume training")
    parser.add_argument("--lmdb", type=str, default=None, help="path to lmdb")
    parser.add_argument('--channels-last', action='store_true', help='Use channels_last memory layout')
    parser.add_argument('--prefetch', action='store_true', help='Use prefetching')
    parser.add_argument('--from-zero', action='store_true', help='Do not load optimizaer and scheduler state dicts')

    args = parser.parse_args()
    print(args)

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    if args.train:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        train(args)

    if args.test:
        os.makedirs(args.predictions_dir, exist_ok=True)
        test(args)
