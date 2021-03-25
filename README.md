# Joint Generative and Contrastive Learning for Unsupervised Person Re-identification

Implement of paper:[Joint Generative and Contrastive Learning for Unsupervised Person Re-identification](https://arxiv.org/pdf/2012.09071.pdf).

# Qualitative results
#### Market1501
![demo](figs/supp1.png)
#### Msmt17
![demo](figs/supp3.png)

Code coming soon...

## Installation

### Install HMR for Mesh Estimation
Please refer to [HMR](https://github.com/akanazawa/hmr).

Requirements
* Python 2.7
* TensorFlow 1.3 
```shell
conda create --name py2 python=2.7
source activate py2
pip install tensorflow-gpu==1.3.0
git clone https://github.com/akanazawa/hmr.git
cd hmr
pip install -r requirements.txt
```

### Install GCL
Requirements
* Python 3.6
* Pytorch
```shell
conda create --name py3 python=3.6
source activate py3
git clone https://github.com/chenhao2345/GCL
cd GCL
python setup.py develop
```

## Prepare Datasets

```shell
cd examples && mkdir data
```
Download the raw datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [MSMT17](https://arxiv.org/abs/1711.08565),
and then unzip them under the directory like
```
ABMT/examples/data
├── dukemtmc-reid
│   └── DukeMTMC-reID
├── market1501
└── msmt17
    └── MSMT17_V1(or MSMT17_V2)
```

## Train GCL
Only support 1 GPU training for the moment.
### Stage 1: Warm up identity encoder
Train a ResNet50 with an unsupervised method, for example, [JVTC](https://github.com/ljn114514/JVTC).
### Stage 2: Warm up structure encoder and discriminator
```shell
sh train_stage2_market.sh
```

### Stage 3: Joint training
```shell
sh train_stage3_market.sh
```


## Citation
```bibtex
@article{chen2020joint,
  title={Joint Generative and Contrastive Learning for Unsupervised Person Re-identification},
  author={Chen, Hao and Wang, Yaohui and Lagadec, Benoit and Dantcheva, Antitza and Bremond, Francois},
  journal={arXiv preprint arXiv:2012.09071},
  year={2020}
}
```