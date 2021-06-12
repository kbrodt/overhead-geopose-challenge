import os
import numpy as np
import sys
import itertools
from collections import OrderedDict

from glob import glob
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import lmdb

from tqdm import tqdm

import albumentations as A
import apex
import json
import cv2
import pandas as pd
import pickle

from segmentation_models_pytorch.utils.meter import AverageValueMeter
from sklearn.model_selection import StratifiedKFold

from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from utilities.misc_utils import (
    convert_and_compress_prediction_dir,
    load_image,
    load_vflow,
    get_rms,
    get_r2_info,
    get_angle_error,
    get_r2,
    save_image,
)
from utilities.augmentation_vflow import augment_vflow
from utilities.unet_vflow import UnetVFLOW


RNG = np.random.RandomState(4321)

p = 0.5
crop_fn = A.RandomCrop(1024, 1024)
albu_train = A.Compose([
    A.RandomCrop(512, 512),
    A.CoarseDropout(max_holes=32, max_height=32, max_width=32, p=p),

    A.OneOf([
        A.Blur(blur_limit=3, p=1),
        A.MotionBlur(p=1),
        A.MedianBlur(blur_limit=3, p=1),
        A.GlassBlur(p=1),
    ], p=p),

    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
        A.RandomToneCurve(p=1),
    ], p=p),

    A.OneOf([
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
        A.FancyPCA(p=0.2),
    ], p=p),
])


class Dataset(BaseDataset):
    def __init__(
        self,
        sub_dir,
        args,
        rng=RNG,
        is_val=False,
    ):

        self.is_test = False
        self.is_val = is_val
        self.rng = rng
        if isinstance(sub_dir, str):
            assert sub_dir == args.test_sub_dir
            assert self.is_val
            self.is_test = sub_dir == args.test_sub_dir
        
            # create all paths with respect to RGB path ordering to maintain alignment of samples
            dataset_dir = Path(args.dataset_dir) / sub_dir
            rgb_paths = list(dataset_dir.glob(f"*_RGB.{args.rgb_suffix}"))
            if rgb_paths == []: rgb_paths = list(dataset_dir.glob(f"*_RGB*.{args.rgb_suffix}")) # original file names
            agl_paths = list(
                pth.with_name(pth.name.replace("_RGB", "_AGL")).with_suffix(".tif")
                for pth in rgb_paths
            )
            vflow_paths = list(
                pth.with_name(pth.name.replace("_RGB", "_VFLOW")).with_suffix(".json")
                for pth in rgb_paths
            )
        else:
            rgb_paths = sub_dir.rgb.apply(lambda x: (Path(args.dataset_dir) / x).with_suffix(f'.{args.rgb_suffix}')).tolist()
            agl_paths = sub_dir.agl.apply(lambda x: Path(args.dataset_dir) / x).tolist()
            vflow_paths = sub_dir.json.apply(lambda x: Path(args.dataset_dir) / x).tolist()

        if self.is_test:
            self.paths_list = rgb_paths
        else:
            self.paths_list = [
                (rgb_paths[i], vflow_paths[i], agl_paths[i])
                for i in range(len(rgb_paths))
            ]

            self.paths_list = [
                self.paths_list[ind]
                for ind in self.rng.permutation(len(self.paths_list))
            ]
            if args.sample_size is not None:
                self.paths_list = self.paths_list[: args.sample_size]
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(
            args.backbone, "imagenet"
        )

        if args.lmdb is not None:
            self.env = lmdb.open(
                str(args.lmdb),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )

        self.args = args

    def __getitem__(self, i):

        if self.is_test:
            rgb_path = self.paths_list[i]
            image = load_image(rgb_path, self.args, use_cv=True)
        else:
            rgb_path, vflow_path, agl_path = self.paths_list[i]
            if hasattr(self, "env"):
                with self.env.begin(write=False, buffers=True) as txn:
                    image = pickle.loads(txn.get(rgb_path.stem.encode()))
                    agl = pickle.loads(txn.get(rgb_path.stem.replace("RGB", "AGL").encode()))
            else:
                image = load_image(rgb_path, self.args)
                agl = load_image(agl_path, self.args)

            # max_agl = np.nanmax(agl)
            if (not self.is_test) and (not self.is_val):
                data = crop_fn(image=image, mask=agl)
                image = data["image"]
                agl = data["mask"]

            mag, xdir, ydir, vflow_data = load_vflow(vflow_path, agl, self.args)
            scale = vflow_data["scale"]
            if (not self.is_val) and self.args.augmentation:
                image, mag, xdir, ydir, agl, scale = augment_vflow(
                    image,
                    mag,
                    xdir,
                    ydir,
                    vflow_data["angle"],
                    vflow_data["scale"],
                    agl=agl,
                    rotate90_prob=0.5,
                    rotate_prob=0.3,
                    flip_prob=0.5,
                    scale_prob=0.5,
                    agl_prob=0.5,
                    # max_agl=max_agl,
                )
            xdir = np.float32(xdir)
            ydir = np.float32(ydir)
            mag = mag.astype("float32")
            agl = agl.astype("float32")
            scale = np.float32(scale)

            xydir = np.array([xdir, ydir])

        if self.is_test and self.args.downsample > 1:
            image = cv2.resize(
                image,
                (
                    int(image.shape[0] / self.args.downsample),
                    int(image.shape[1] / self.args.downsample),
                ),
                interpolation=cv2.INTER_NEAREST,
            )

        image = image.astype("uint8")
        if (not self.is_test) and (not self.is_val):
            data = albu_train(image=image, masks=[mag, agl])
            image = data["image"]
            mag, agl = data["masks"]

        image = self.preprocessing_fn(image).astype("float32")
        image = np.transpose(image, (2, 0, 1))

        if self.is_test:
            return image, str(rgb_path)
        else:
            return image, xydir, agl, mag, scale

    def __len__(self):
        return len(self.paths_list)

    @staticmethod
    def fast_collate(batch):
        image, xydir, agl, mag, scale = zip(*batch)

        image, xydir, agl, mag, scale = map(
            lambda x: torch.tensor(x),
            (image, xydir, agl, mag, scale)
        )

        return image, xydir, agl, mag, scale

#         return torch.stack(x), torch.stack(y)

#         targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
#         assert len(targets) == batch_size
#         tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)
#         for i in range(batch_size):
#             tensor[i] += torch.from_numpy(batch[i][0])
#         return tensor, targets


class PrefetchLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_image, next_xydir, next_agl, next_mag, next_scale in self.loader:
            with torch.cuda.stream(stream):
                next_image, next_xydir, next_agl, next_mag, next_scale = map(
                    lambda x: x.cuda(non_blocking=True),
                    (next_image, next_xydir, next_agl, next_mag, next_scale)
                )

            if not first:
                yield image, xydir, agl, mag, scale
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            image, xydir, agl, mag, scale = next_image, next_xydir, next_agl, next_mag, next_scale

        yield image, xydir, agl, mag, scale

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class Epoch:
    def __init__(
        self,
        model,
        args,
        dense_loss=None,
        angle_loss=None,
        scale_loss=None,
        stage_name=None,
        device="cpu",
        verbose=True,
        local_rank=0,
        channels_last=False,
    ):
        self.args = args
        self.model = model
        self.dense_loss = dense_loss
        self.angle_loss = angle_loss
        self.scale_loss = scale_loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.local_rank = local_rank
        self.channels_last = channels_last

        self.loss_names = ["combined", "agl", "mag", "angle", "scale"]

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if self.stage_name != "valid":
            self.dense_loss.to(self.device)
            self.angle_loss.to(self.device)
            self.scale_loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, desc=""):

        self.on_epoch_start()

        logs = {}

        loss_meters = {}
        for loss_name in self.loss_names:
            loss_meters[loss_name] = AverageValueMeter()

        if self.stage_name == "valid":
            agl_count, agl_error_sum, agl_gt_sq_sum, agl_sum = 0, 0, 0, 0
            mag_count, mag_error_sum, mag_gt_sq_sum, mag_sum = 0, 0, 0, 0
            angle_errors = []
            agl_rms = []
            mag_rms = []
            scale_errors = []

        if self.local_rank == 0:
            iterator = tqdm(
                total=len(dataloader),
                desc=f"{self.stage_name} {desc}",
                file=sys.stdout,
                disable=not (self.verbose),
                mininterval=2,
                leave=False,
            )

        for itr_data in dataloader:
            image, xydir, agl, mag, scale = itr_data
            scale = torch.unsqueeze(scale, 1)

            image = image.to(self.device, non_blocking=True)

            if self.stage_name != "valid":
                xydir, agl, mag, scale = (
                    xydir.to(self.device, non_blocking=True),
                    agl.to(self.device, non_blocking=True),
                    mag.to(self.device, non_blocking=True),
                    scale.to(self.device, non_blocking=True),
                )
                y = [xydir, agl, mag, scale]

                (
                    loss,
                    xydir_pred,
                    agl_pred,
                    mag_pred,
                    scale_pred,
                ) = self.batch_update(image, y)

                loss_logs = {}

                for name in self.loss_names:
                    curr_loss = loss[name].cpu().detach().numpy()
                    if name == "scale":
                        curr_loss = np.mean(curr_loss)
                    loss_meters[name].add(curr_loss)
                    loss_logs[name] = loss_meters[name].mean

                logs.update(loss_logs)
            else:
                xydir_pred, agl_pred, mag_pred, scale_pred = self.batch_update(
                    image
                )

                xydir = xydir.cpu().numpy()
                agl = agl.cpu().numpy()
                mag = mag.cpu().numpy()
                scale = scale.cpu().numpy()

                xydir_pred = xydir_pred.cpu().numpy()
                agl_pred = agl_pred.cpu().numpy()
                mag_pred = mag_pred.cpu().numpy()
                scale_pred = scale_pred.cpu().numpy()

                for batch_ind in range(agl.shape[0]):

                    count, error_sum, rms, data_sum, gt_sq_sum = get_r2_info(
                        agl[batch_ind], agl_pred[batch_ind]
                    )
                    agl_count += count
                    agl_error_sum += error_sum
                    agl_rms.append(rms)
                    agl_sum += data_sum
                    agl_gt_sq_sum += gt_sq_sum

                    vflow = np.zeros((agl[batch_ind].squeeze().shape[0], agl[batch_ind].squeeze().shape[1], 2))
                    vflow[..., 0] = mag[batch_ind].squeeze() * xydir[batch_ind, 0]
                    vflow[..., 1] = mag[batch_ind].squeeze() * xydir[batch_ind, 1]

                    vflow_pred = np.zeros_like(vflow)
                    vflow_pred[..., 0] = mag_pred[batch_ind].squeeze() * xydir_pred[batch_ind, 0]
                    vflow_pred[..., 1] = mag_pred[batch_ind].squeeze() * xydir_pred[batch_ind, 1]

                    count, error_sum, rms, data_sum, gt_sq_sum = get_r2_info(
                        vflow, vflow_pred,
                    )
                    mag_count += count
                    mag_error_sum += error_sum
                    mag_rms.append(rms)
                    mag_sum += data_sum
                    mag_gt_sq_sum += gt_sq_sum

                    dir_pred = xydir_pred[batch_ind, :]
                    dir_gt = xydir[batch_ind, :]

                    angle_error = get_angle_error(dir_pred, dir_gt)

                    angle_errors.append(angle_error)
                    scale_errors.append(
                        np.abs(scale[batch_ind] - scale_pred[batch_ind])
                    )

            torch.cuda.synchronize()

            if self.local_rank == 0 and self.verbose:
                s = self._format_logs(logs)
                iterator.set_postfix_str(s)
                iterator.update()

        if self.stage_name == "valid":
            logs.update(dict(
                agl_error_sum=agl_error_sum,
                agl_gt_sq_sum=agl_gt_sq_sum,
                agl_sum=agl_sum,
                agl_count=agl_count,

                mag_error_sum=mag_error_sum,
                mag_gt_sq_sum=mag_gt_sq_sum,
                mag_sum=mag_sum,
                mag_count=mag_count,

                angle_errors=angle_errors,
                scale_errors=scale_errors,

                agl_rms=agl_rms,
                mag_rms=mag_rms,
            ))

        if self.local_rank == 0:
            iterator.close()

        return logs


class TrainEpoch(Epoch):
    def __init__(
        self,
        model,
        args,
        dense_loss,
        angle_loss,
        scale_loss,
        optimizer,
        scaler=None,
        device="cpu",
        verbose=True,
        local_rank=0,
        channels_last=False,
    ):
        super().__init__(
            model=model,
            args=args,
            dense_loss=dense_loss,
            angle_loss=angle_loss,
            scale_loss=scale_loss,
            stage_name="train",
            device=device,
            verbose=verbose,
            local_rank=local_rank,
            channels_last=channels_last,
        )
        self.optimizer = optimizer
        self.scaler = scaler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        # if self.channels_last:
            # x = x.contiguous(memory_format=torch.channels_last)

        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                xydir_pred, agl_pred, mag_pred, scale_pred = self.model.forward(x)
                scale_pred = torch.unsqueeze(scale_pred, 1)

                xydir, agl, mag, scale = y
                loss_agl = self.dense_loss(agl_pred, agl)
                loss_mag = self.dense_loss(mag_pred, mag)
                loss_angle = self.angle_loss(xydir_pred, xydir)

                loss_scale = self.scale_loss(scale_pred, scale)

                loss_combined = (
                    self.args.agl_weight * loss_agl
                    + self.args.mag_weight * loss_mag
                    + self.args.angle_weight * loss_angle
                    + self.args.scale_weight * loss_scale
                )
        else:
            xydir_pred, agl_pred, mag_pred, scale_pred = self.model.forward(x)

            scale_pred = torch.unsqueeze(scale_pred, 1)

            xydir, agl, mag, scale = y
            loss_agl = self.dense_loss(agl_pred, agl)
            loss_mag = self.dense_loss(mag_pred, mag)
            loss_angle = self.angle_loss(xydir_pred, xydir)

            loss_scale = self.scale_loss(scale_pred, scale)

            loss_combined = (
                self.args.agl_weight * loss_agl
                + self.args.mag_weight * loss_mag
                + self.args.angle_weight * loss_angle
                + self.args.scale_weight * loss_scale
            )

        loss = {
            "combined": loss_combined,
            "agl": loss_agl,
            "mag": loss_mag,
            "angle": loss_angle,
            "scale": loss_scale,
        }

        if self.scaler is None:
            loss_combined.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
        else:
            self.scaler.scale(loss_combined).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return loss, xydir_pred, agl_pred, mag_pred, scale_pred


class ValidEpoch(Epoch):
    def __init__(self, model, args, device="cpu", verbose=True, local_rank=0, channels_last=False):
        super().__init__(
            model=model,
            args=args,
            stage_name="valid",
            device=device,
            verbose=verbose,
            local_rank=local_rank,
            channels_last=channels_last,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x):
        # if self.channels_last:
            # x = x.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            xydir_pred, agl_pred, mag_pred, scale_pred = self.model.forward(x)

            scale_pred = torch.unsqueeze(scale_pred, 1)

        return xydir_pred, agl_pred, mag_pred, scale_pred


class NoNaNMSE(smp.utils.base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, output, target):
        diff = torch.squeeze(output) - target
        not_nan = ~torch.isnan(diff)
        loss = torch.mean(diff.masked_select(not_nan) ** 2)
        return loss


class AngleLoss(smp.utils.base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def forward(self, output, target):
        dot = (output * target).sum(-1)
        loss = torch.mean((dot - 1) ** 2)

        return loss


class MSELoss(smp.utils.base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = torch.nn.MSELoss()

    def forward(self, output, target):
        return self.mse(output, target)
    
    
class L1SmoothLoss(smp.utils.base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.l1 = torch.nn.SmoothL1Loss()

    def forward(self, output, target):
        return self.l1(output, target)
    
    
def train_dev_split(geopose, args):
    geopose['area'] = geopose.rgb.str.split('_').str[0]
    geopose['fold'] = None

    n_col = len(geopose.columns) - 1
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    for fold, (_, dev_index) in enumerate(skf.split(geopose, geopose.area)):
        geopose.iloc[dev_index, n_col] = fold
        
    train, dev = geopose[geopose.fold != args.fold].copy(), geopose[geopose.fold == args.fold].copy()

    return train, dev


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_dist(args):
    # to autotune convolutions and other algorithms
    # to pick the best for current configuration
    torch.backends.cudnn.benchmark = True

    if args.deterministic:
        set_seed(args.random_state)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
    ]


def train(args):
    if args.distributed:
        init_dist(args)

    torch.backends.cudnn.benchmark = True

    summary_writer = None
    if args.local_rank == 0:
        summary_writer = SummaryWriter(Path(args.checkpoint_dir) / "logs")  # /exp_name

    model = build_model(args)
    model = model.cuda()
    if args.resume:
        path_to_resume = Path(args.resume).expanduser()
        if path_to_resume.is_file():
            print(f"=> loading resume checkpoint '{path_to_resume}'")
            checkpoint = torch.load(
                path_to_resume,
                map_location=lambda storage, loc: storage.cuda(args.gpu),  # change here!
            )
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            print(
                f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"=> no checkpoint found at '{path_to_resume}'")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    weight_decay = args.weight_decay
    if weight_decay > 0:  # and filter_bias_and_bn:
        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()

        parameters = add_weight_decay(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    parameters = model.parameters()
    optimizer = apex.optimizers.FusedAdam(  # torch.optim.Adam(
        parameters,
        adam_w_mode=True,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.resume:
        optimizer.load_state_dict(checkpoint["opt_state_dict"])

    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        # model = apex.parallel.convert_syncbn_model(model)
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    
    df = pd.read_csv(args.train_path_df)
    train_df, dev_df = train_dev_split(df, args)

    CITIES = ["ARG", "ATL", "JAX", "OMA"]
    train_dataset = Dataset(train_df, args=args, is_val=False)
    dev_df = dev_df[dev_df.area == CITIES[args.local_rank]].reset_index(drop=True)
    val_dataset = Dataset(dev_df, args=args, is_val=True)

    train_sampler = None
    val_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    args.num_workers = min(max(args.num_workers, args.batch_size), 16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=None,  # train_dataset.fast_collate,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(args.batch_size // 4, 1),
        shuffle=False,
        sampler=val_sampler,
        collate_fn=None,  # val_dataset.fast_collate,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
    )
    if args.prefetch:
        train_loader = PrefetchLoader(train_loader)
        val_loader = PrefetchLoader(val_loader)
    
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            # optimizer, T_max=args.T_max, eta_min=max(args.learning_rate * 1e-2, 1e-6)
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5 * args.T_max, T_mult=1, eta_min=max(args.learning_rate * 1e-2, 1e-6)  # T_mult=sqrt(2)
    )

    scaler = None
    scaler = torch.cuda.amp.GradScaler()

    dense_loss = NoNaNMSE()
    angle_loss = MSELoss()
    scale_loss = MSELoss()

    train_epoch = TrainEpoch(
        model,
        args=args,
        dense_loss=dense_loss,
        angle_loss=angle_loss,
        scale_loss=scale_loss,
        optimizer=optimizer,
        scaler=scaler,
        device="cuda",
        local_rank=args.local_rank,
        channels_last=args.channels_last,
    )

    val_epoch = ValidEpoch(
        model,
        args=args,
        device="cuda",
        local_rank=args.local_rank,
        channels_last=args.channels_last,
    )
    
    best_score = 0
    def saver(path):
        torch.save(
            {
                "epoch": i,
                "best_score": best_score,
                "state_dict": model.state_dict(),
                "opt_state_dict": optimizer.state_dict(),
                "sched_state_dict": scheduler.state_dict()
                if scheduler is not None
                else None,
                "scaler": scaler.state_dict()
                if scaler is not None
                else None,
                "args": args,
            },
            path,
        )

    start_epoch = 0
    if not args.from_zero and args.resume:
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        if checkpoint["sched_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["sched_state_dict"])

        if checkpoint["scaler"] is not None:
            scaler.load_state_dict(checkpoint["scaler"])

    for i in range(start_epoch, args.num_epochs):
        if args.distributed:
            train_sampler.set_epoch(i)

        desc = f"{i}/{args.num_epochs}"
        train_logs = train_epoch.run(train_loader, desc=desc)
        # train_logs_out = [None for _ in range(torch.distributed.get_world_size())]
        # torch.distributed.all_gather_object(train_logs_out, train_logs)
        # train_lgs = {}
        # for name in list(train_logs):
            # train_lgs[name] = sum(x[name] for x in train_logs_out) / len(train_logs_out)
        # train_logs = train_lgs
        for name in list(train_logs):
            train_logs_out = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(train_logs_out, train_logs[name])
            train_logs[name] = sum(train_logs_out) / len(train_logs_out)

        if args.val_period > 0 and ((i + 1) % args.val_period) == 0:
            valid_logs = val_epoch.run(val_loader, desc=desc)
            # valid_logs_out = [None for _ in range(torch.distributed.get_world_size())]
            # torch.distributed.all_gather_object(valid_logs_out, valid_logs)
            # valid_lgs = {}
            # for name in list(valid_logs):
                # valid_lgs[name] = list(itertools.chain(*[x[name] for x in valid_logs_out]))
            # valid_logs = valid_lgs
            for name in list(valid_logs):
                valid_logs_out = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(valid_logs_out, valid_logs[name])
                valid_logs[name] = valid_logs_out

        if args.local_rank == 0 and ((i + 1) % args.save_period) == 0:
            saver(
                os.path.join(args.checkpoint_dir, "./model_last.pth"),
            )
        
        if scheduler is not None:
            scheduler.step()

        if args.local_rank == 0 and args.val_period > 0 and args.save_best:
            r2_agl_per_city = {}
            r2_mag_per_city = {}
            score_per_city = {}
            for city_index, city in enumerate(CITIES):
                agl_error_sum = valid_logs["agl_error_sum"][city_index]
                agl_gt_sq_sum = valid_logs["agl_gt_sq_sum"][city_index]
                agl_sum = valid_logs["agl_sum"][city_index]
                agl_count = valid_logs["agl_count"][city_index]
                r2_agl = get_r2(agl_error_sum, agl_gt_sq_sum, agl_sum, agl_count)

                r2_agl_per_city[city] = r2_agl

                mag_error_sum = valid_logs["mag_error_sum"][city_index]
                mag_gt_sq_sum = valid_logs["mag_gt_sq_sum"][city_index]
                mag_sum = valid_logs["mag_sum"][city_index]
                mag_count = valid_logs["mag_count"][city_index]
                r2_mag = get_r2(mag_error_sum, mag_gt_sq_sum, mag_sum, mag_count)

                r2_mag_per_city[city] = r2_mag

                score_per_city[city] = (r2_agl + r2_mag) / 2

            # agl_error_sum = sum(valid_logs["agl_error_sum"])
            # agl_gt_sq_sum = sum(valid_logs["agl_gt_sq_sum"])
            # agl_sum = sum(valid_logs["agl_sum"])
            # agl_count = sum(valid_logs["agl_count"])
            # r2_agl = get_r2(agl_error_sum, agl_gt_sq_sum, agl_sum, agl_count)

            # mag_error_sum = sum(valid_logs["mag_error_sum"])
            # mag_gt_sq_sum = sum(valid_logs["mag_gt_sq_sum"])
            # mag_sum = sum(valid_logs["mag_sum"])
            # mag_count = sum(valid_logs["mag_count"])
            # r2_mag = get_r2(mag_error_sum, mag_gt_sq_sum, mag_sum, mag_count)

            # angle_rms = get_rms(valid_logs["angle_errors"])
            # scale_rms = get_rms(valid_logs["scale_errors"])
            # agl_rms = get_rms(valid_logs["agl_rms"])
            # mag_rms = get_rms(valid_logs["mag_rms"])

            score = sum(score_per_city.values()) / len(score_per_city)

            if i > 0:
                for idx, param_group in enumerate(optimizer.param_groups):
                    lr = param_group['lr']
                    summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=i)

                summary_writer.add_scalars('train/mse', train_logs, global_step=i)

                summary_writer.add_scalars("val/r2_agl", r2_agl_per_city, global_step=i)
                summary_writer.add_scalars("val/r2_mag", r2_mag_per_city, global_step=i)
                summary_writer.add_scalars("val/scores", score_per_city, global_step=i)
                summary_writer.add_scalar("val/score", score, global_step=i)

            if score > best_score:
                best_score = score
                saver(
                    os.path.join(args.checkpoint_dir, "./model_best.pth"),
                )

    if args.local_rank == 0:
        summary_writer.close()


def test(args):

    torch.backends.cudnn.benchmark = True

    if args.model_path is None:
        model_paths = glob(os.path.join(args.checkpoint_dir, "*.pth"))
        nums = [int(path.split("_")[-1].replace(".pth", "")) for path in model_paths]
        idx = np.argsort(nums)[::-1]
        model_path = model_paths[idx[0]]
    else:
        model_path = args.model_path

    model = build_model(args)
    checkpoint = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint["state_dict"].items():
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to("cuda")
    model.eval()

    # if 'efficientnet' in args.backbone:
        # model.encoder.set_swish(memory_efficient=False)

    with torch.no_grad():
        # model = torch.jit.trace(model, torch.rand(2, 3, 512, 512).cuda())

        test_dataset = Dataset(sub_dir=args.test_sub_dir, args=args, is_val=True)
        args.num_workers = min(max(args.num_workers, args.batch_size), 16)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        )
        predictions_dir = Path(args.predictions_dir)
        for images, rgb_paths in tqdm(test_loader, mininterval=2):

            images = images.to("cuda", non_blocking=True)
            pred = model(images)
            pred = list(pred)

            if args.tta > 1:  # vertical flip
                pred_tta = model(torch.flip(images, dims=[-1]))
                xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                xydir_pred_tta[:, 0] *= -1
                agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1])

                pred[0] += xydir_pred_tta
                pred[1] += agl_pred_tta
                pred[3] += scale_pred_tta

            if args.tta > 2:  # horizontal flip
                pred_tta = model(torch.flip(images, dims=[-2]))
                xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                xydir_pred_tta[:, 1] *= -1
                agl_pred_tta = torch.flip(agl_pred_tta, dims=[-2])

                pred[0] += xydir_pred_tta
                pred[1] += agl_pred_tta
                pred[3] += scale_pred_tta

            if args.tta > 3:  # vertical+horizontal flip
                pred_tta = model(torch.flip(images, dims=[-1, -2]))
                xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                xydir_pred_tta *= -1
                agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1, -2])

                pred[0] += xydir_pred_tta
                pred[1] += agl_pred_tta
                pred[3] += scale_pred_tta

            if args.tta > 7:  # rotate90
                images_rot90 = torch.rot90(images, k=1, dims=[-2, -1])

                pred_tta = model(images_rot90)
                xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                xydir_pred_tta = torch.stack([-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1)
                agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                pred[0] += xydir_pred_tta
                pred[1] += agl_pred_tta
                pred[3] += scale_pred_tta

                # vertical flip
                pred_tta = model(torch.flip(images_rot90, dims=[-1]))
                xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                xydir_pred_tta[:, 0] *= -1
                agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1])
                xydir_pred_tta = torch.stack([-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1)
                agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                pred[0] += xydir_pred_tta
                pred[1] += agl_pred_tta
                pred[3] += scale_pred_tta

                # horizontal flip
                pred_tta = model(torch.flip(images_rot90, dims=[-2]))
                xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                xydir_pred_tta[:, 1] *= -1
                agl_pred_tta = torch.flip(agl_pred_tta, dims=[-2])
                xydir_pred_tta = torch.stack([-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1)
                agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                pred[0] += xydir_pred_tta
                pred[1] += agl_pred_tta
                pred[3] += scale_pred_tta

                # vertical+horizontal flip
                pred_tta = model(torch.flip(images_rot90, dims=[-1, -2]))
                xydir_pred_tta, agl_pred_tta, _, scale_pred_tta = pred_tta
                xydir_pred_tta *= -1
                agl_pred_tta = torch.flip(agl_pred_tta, dims=[-1, -2])
                xydir_pred_tta = torch.stack([-xydir_pred_tta[:, 1], xydir_pred_tta[:, 0]], dim=1)
                agl_pred_tta = torch.rot90(agl_pred_tta, k=-1, dims=[-2, -1])

                pred[0] += xydir_pred_tta
                pred[1] += agl_pred_tta
                pred[3] += scale_pred_tta

            pred[0] /= args.tta
            pred[1] /= args.tta
            pred[3] /= args.tta

            numpy_preds = []
            for i in range(len(pred)):
                numpy_preds.append(pred[i].cpu().numpy())

            xydir_pred, agl_pred, _, scale_pred = numpy_preds

            if scale_pred.ndim == 0:
                scale_pred = np.expand_dims(scale_pred, axis=0)

            for batch_ind in range(agl_pred.shape[0]):
                # vflow pred
                angle = np.arctan2(xydir_pred[batch_ind][0], xydir_pred[batch_ind][1])
                vflow_data = {
                    "scale": np.float64(
                        scale_pred[batch_ind] * args.downsample
                    ),  # upsample
                    "angle": np.float64(angle),
                }

                # agl pred
                curr_agl_pred = agl_pred[batch_ind, 0, :, :]
                curr_agl_pred[curr_agl_pred < 0] = 0
                agl_resized = curr_agl_pred
                if args.downsample > 1:
                    agl_resized = cv2.resize(
                        curr_agl_pred,
                        (
                            curr_agl_pred.shape[0] * args.downsample,  # upsample
                            curr_agl_pred.shape[1] * args.downsample,  # upsample
                        ),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # save
                rgb_path = predictions_dir / Path(rgb_paths[batch_ind]).name
                agl_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_AGL")
                ).with_suffix(".tif")
                vflow_path = rgb_path.with_name(
                    rgb_path.name.replace("_RGB", "_VFLOW")
                ).with_suffix(".json")

                json.dump(vflow_data, vflow_path.open("w"))
                save_image(agl_path, agl_resized)  # save_image assumes units of meters

    # creates new dir predictions_dir_con
    if args.convert_predictions_to_cm_and_compress:
        convert_and_compress_prediction_dir(predictions_dir=predictions_dir)


def build_model(args):
    model = UnetVFLOW(args.backbone, encoder_weights="imagenet")
    return model
