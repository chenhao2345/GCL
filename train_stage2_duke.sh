python examples/main.py \
    --name duke_init_JVTC_unsupervised \
    --dataset-target dukemtmc-reid \
    --idnet-fix \
    --stage 2 \
    --epochs 40 \
    --init ./examples/logs/JVTC/duke/resnet50_duke075_epoch00040.pth \
    --mesh-dir /data/stars/user/yaowang/data/reid/DukeMTMC/