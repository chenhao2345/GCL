CUDA_VISIBLE_DEVICES=0 python examples/main.py \
--name msmt_init_JVTC_unsupervised \
--resume \
--stage 3 \
--epochs 20 \
--dataset-target msmt17 \
--mesh-dir ./examples/mesh/msmt17/ \
--rho 0.0012 \
--k1 30
