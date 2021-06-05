CUDA_VISIBLE_DEVICES=0 python examples/main.py \
--name duke_init_JVTC_unsupervised \
--resume \
--stage 3 \
--epochs 20 \
--dataset-target dukemtmc-reid \
--mesh-dir ./examples/mesh/DukeMTMC/ \
--rho 0.002 \
--k1 30
