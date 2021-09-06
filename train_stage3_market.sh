CUDA_VISIBLE_DEVICES=0 python examples/main.py \
--name market_init_JVTC_unsupervised \
--resume \
--stage 3 \
--epochs 20 \
--dataset-target market1501 \
--mesh-dir ./examples/mesh/market/ \
--rho 0.002 \
--k1 30
