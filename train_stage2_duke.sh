CUDA_VISIBLE_DEVICES=0 python examples/main.py \
    --name duke_init_JVTC_unsupervised \
    --dataset-target dukemtmc-reid \
    --stage 2 \
    --epochs 40 \
    --init ./examples/logs/JVTC/duke/resnet50_duke075_epoch00040.pth \
    --mesh-dir ./examples/mesh/DukeMTMC/
