
python examples/main.py \
    --name market_init_JVTC_unsupervised \
    --dataset-target market1501 \
    --idnet-fix \
    --stage 2 \
    --epochs 40 \
    --init ./examples/logs/JVTC/market/resnet50_market075_epoch00045.pth \
    --mesh-dir /data/stars/user/yaowang/data/reid/market/