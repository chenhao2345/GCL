#!/bin/sh

#OAR -p gpu='YES' and gpumem>=20000
#OAR -n market_init_JVTC_unsupervised
#OAR -l//nodes=1/gpunum=1, walltime=48
#OAR --notify mail:hao.chen@inria.fr

module load cuda/10.0
module load cudnn/7.4-cuda-10.0
source activate pytorch1

python examples/main.py \
    --name market_init_JVTC_unsupervised \
    --dataset-target market1501 \
    --stage 2 \
    --epochs 40 \
    --init ./examples/logs/JVTC/market/resnet50_market075_epoch00045.pth \
    --mesh-dir /data/stars/user/yaowang/data/reid/market/