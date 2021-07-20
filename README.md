# Overhead Geopose Challenge

[Overhead Geopose Challenge](https://www.drivendata.org/competitions/78/overhead-geopose-challenge/)

[4th
place](https://www.drivendata.org/competitions/78/overhead-geopose-challenge/leaderboard/)
out of 444 participants with $0.8731$ R2 coefficient of determination (top1 $0.924$).

## Prerequisites

- At least 4 GPUs with 32GB VRAM (e.~g. Tesla V100)
- [NVIDIA apex](https://github.com/NVIDIA/apex)

You can use 1 GPU but the training process will take almost 4 times longer.

## Usage

### Data preprocessing

Download data from the competition link into `data` folder and run

```bash
sh ./preprocess.sh
```

Optionally you may save images to `lmdb` to speedup training.

```bash
sh ./save_to_lmdb.sh
```

Note you will need ~200GB disk space and RAM (rewrite the code for iterative
preprocessing not loading all data into the memory).

### Training

```bash
sh ./dist_train.sh
```

It will take about 1 week on 4 GPUS.

### Inference

You can download pretrained models
[here](https://disk.yandex.com/d/YKBkCPWV1jaYrg) (extract via `unzip models.zip
-d chkps_dist`).

```bash

sh ./dist_test.sh
```

## Approach

Train Unet-like model with various encoders (`efficientnet-b{6,7}` and
`senet154`) to predict height fields with two additional heads in the
bottleneck correspoding to scale and angle. At inference time average the
predictions from all models taking full size `2048x2048` image. Note, that
a single model also ranks at 4th place with $0.86$ `R2` coefficient of
determination.

Pretrain models with heavy augmentations (like flips, rotations, color
jittering, scaling, height augmentations etc.) for 525 epochs and finetune
another 1025 epochs without any augmentations.

*Remark*: Models without any augmentations train faster and better according to
validation splits.

## Highlights

- Unet-like models with `efficientnet` and `senet` encoders
- Train on random crops with `512x512` size to speedup training
- Batchsize $8$ (without syncronized batch norm to speedup training)
- `AdamW` optimizer with `1e-4` learning rate
- `CosineAnnealingLR` scheduler with period $25$ epochs
- Pretrain with heavy augmentations and finetune without any ones
- Average the predictions from all models
