#!/bin/bash


GPU=${GPU:-0,1,2,3,4,5,6,7}
PORT=${PORT:-29500}
N_GPUS=8
BATCH_SIZE=6
DATASET_DIR=./data/test
TEST_DF=./data/geopose_test.csv
RGB_SUFFIX=j2k
UNIT=cm
TTA=1
PREDS_DIR=./chkps_dist/test_sn_f0_e384_f0_e885_f2_e957_eff5_f2_e685_eff6_f1_e483_eff7_f2_e525_tta${TTA}


CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --nproc_per_node=${N_GPUS} --master_port=${PORT} \
    ./overhead-geopose-challenge/main.py \
    --test \
    --distributed \
    --use-city \
    --tta=${TTA} \
    --model-pt \
        ./chkps_dist/senet154/0_aug_ft_unet/model_best_384.pt \
        ./chkps_dist/senet154/0_noaug_unet/model_best_885.pt \
        ./chkps_dist/senet154/2_city_finetune/model_best_957.pt \
        ./chkps_dist/efficientnet-b5/2_aug_ft_unet/model_best_685.pt \
        ./chkps_dist/efficientnet-b6/1_aug_ft_unet/model_best_483.pt \
        ./chkps_dist/efficientnet-b7/2_aug_ft_unet/model_best_525.pt \
    --use-cities t t t t t t \
    --predictions-dir=${PREDS_DIR} \
    --dataset-dir=${DATASET_DIR} \
    --test-path-df=${TEST_DF} \
    --batch-size=${BATCH_SIZE} \
    --rgb-suffix=${RGB_SUFFIX} \
    --unit=${UNIT} \
    --convert-predictions-to-cm-and-compress=True
