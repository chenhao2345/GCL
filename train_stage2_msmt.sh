CUDA_VISIBLE_DEVICES=0 python examples/main.py \
    --name msmt_init_JVTC_unsupervised \
    --dataset-target msmt17 \
    --stage 2 \
    --epochs 40 \
    --init ./examples/logs/JVTC/msmt/resnet50_msmt085eps4_epoch00099.pth \
    --mesh-dir ./examples/mesh/msmt17/
