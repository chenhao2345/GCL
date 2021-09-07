from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import random
import numpy as np

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision
from gcl import datasets
from gcl import models
from gcl.trainer import DGNet_Trainer
from gcl.utils.data import transforms as T
from gcl.utils.data.preprocessor import Preprocessor, AllMeshPreprocessor
from gcl.utils.serialization import load_checkpoint
from gcl.utils.gan_utils import get_config
from gcl.evaluators import Evaluator
from torchvision import utils
from tqdm import tqdm


start_epoch = best_mAP = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def get_display_loader(dataset, height, width, batch_size, workers, testset=None, mesh_dir=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    mesh_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    if (testset is None):
        testset = dataset.gallery
        mesh_dir = mesh_dir + 'test/'
    else:
        mesh_dir = mesh_dir + 'train/'

    test_loader = DataLoader(
        AllMeshPreprocessor(testset, root=dataset.images_dir, transform=test_transformer, mesh_dir=mesh_dir,
                            mesh_transform=mesh_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)

    if args.init == '':
        print('No idnet init.')
    else:
        checkpoint = load_checkpoint(args.init)
        model_1.load_state_dict(checkpoint, strict=False)

    model_1.cuda()

    return model_1


def denormalize_recon(x):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    x_recon = (x * std) + mean

    return x_recon

def denormalize_mesh(x):
    mean = torch.FloatTensor([0.5]).cuda()
    std = torch.FloatTensor([0.5]).cuda()
    x_recon = (x * std) + mean

    return x_recon

def generate_recon(trainer, train_display_loader, path):
    for i, (img, mesh_org, all_mesh_nv, fname, pid, camid, index) in enumerate(tqdm(train_display_loader)):
        img_name = fname[0].split('/')[-1]  # image name

        img = img.cuda()
        mesh_org = mesh_org.cuda()

        img_recon = trainer.sample_recon(img, mesh_org)
        img_recon = denormalize_recon(img_recon)

        utils.save_image(
            img_recon,
            osp.join(path, img_name),
            normalize=True,
            range=(0, 1.0),
        )


def generate_nv(trainer, train_display_loader, path, degree):
    for i, (img, mesh_org, all_mesh_nv, fname, pid, camid, index) in enumerate(tqdm(train_display_loader)):
        idx = int(degree / 45 - 1)
        img_name = fname[0].split('/')[-1]  # image name

        img = img.cuda()
        mesh_nv = all_mesh_nv[idx].cuda()

        img_nv = trainer.sample_nv(img, mesh_nv)
        img_nv = denormalize_recon(img_nv)

        utils.save_image(
            img_nv,
            osp.join(path, img_name),
            normalize=True,
            range=(0, 1.0),
        )


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True

    config = get_config(args.config)

    # loading model
    print('==> loading model')
    if args.dataset_target == 'market1501':
        checkpoint_path = 'outputs/market_init_JVTC_unsupervised/checkpoints'
        model_1 = create_model(args)
        trainer = DGNet_Trainer(config, model_1, args.idnet_fix).cuda()
        iterations = trainer.resume(checkpoint_path, hyperparameters=config)
        output_path = 'outputs/market_init_JVTC_unsupervised/images'
        os.makedirs(output_path, exist_ok=True)
    elif args.dataset_target == 'dukemtmc-reid':
        checkpoint_path = 'outputs/duke_init_JVTC_unsupervised/checkpoints'
        model_1 = create_model(args)
        trainer = DGNet_Trainer(config, model_1, args.idnet_fix).cuda()
        iterations = trainer.resume(checkpoint_path, hyperparameters=config)
        output_path = 'outputs/duke_init_JVTC_unsupervised/images'
        os.makedirs(output_path, exist_ok=True)
    elif args.dataset_target == 'msmt17':
        checkpoint_path = 'outputs/msmt_init_JVTC_unsupervised/checkpoints'
        model_1 = create_model(args)
        trainer = DGNet_Trainer(config, model_1, args.idnet_fix).cuda()
        iterations = trainer.resume(checkpoint_path, hyperparameters=config)
        output_path = 'outputs/msmt_init_JVTC_unsupervised/images'
        os.makedirs(output_path, exist_ok=True)
    else:
        raise NotImplementedError

    # prepare dataset
    print('==> preparing dataset')
    # Create data loaders
    dataset_target = get_data(args.dataset_target, args.data_dir)

    # # evaluation
    # test_loader_target = get_test_loader(dataset_target, args.height, args.width, 128, args.workers)
    # evaluator_1 = Evaluator(model_1)
    # _, mAP_1 = evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    # display set
    train_display_loader = get_display_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train, mesh_dir=args.mesh_dir)
    # train_display_loader = get_display_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
    #                                           mesh_dir=args.mesh_dir)

    # generate data
    if args.mode == 'recon':
        save_path = osp.join(output_path, args.mode)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        generate_recon(trainer, train_display_loader, save_path)
    elif args.mode == 'nv':
        save_path = osp.join(output_path, args.mode, str(args.degree))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        generate_nv(trainer, train_display_loader, save_path, degree=args.degree)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MV Training")
    # data
    parser.add_argument('--dataset-target', type=str, default='market1501', choices=datasets.names())
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # model
    parser.add_argument('-a', '--arch', type=str, default='ft_net', choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)

    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--idnet-fix", action="store_true")
    parser.add_argument('--stage', type=int, default=1)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--mesh-dir', type=str, metavar='PATH', default='./examples/mesh/market/')

    # gan config
    parser.add_argument('--config', type=str, default='configs/latest.yaml', help='Path to the config file.')
    # parser.add_argument('--output_path', type=str, default='/outputs', help="generated images saving path")
    parser.add_argument('--mode', type=str, default='recon')
    parser.add_argument('--degree', type=int, default=45)

    main()
