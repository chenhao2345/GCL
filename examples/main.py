from __future__ import print_function, absolute_import

import argparse
import os
import os.path as osp
import random
import numpy as np
import sys

from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from gcl import datasets
from gcl import models
from gcl.trainer import DGNet_Trainer
from gcl.evaluators import Evaluator, extract_features
from gcl.utils.data import IterLoader
from gcl.utils.data import transforms as T
from gcl.utils.data.sampler import RandomMultipleGallerySampler
from gcl.utils.data.preprocessor import MeshPreprocessor, Preprocessor
from gcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from gcl.utils.faiss_rerank import compute_jaccard_distance
from gcl.utils.gan_utils import get_config, prepare_sub_folder, write_loss, display_images
from torch.utils.tensorboard import SummaryWriter
start_epoch = best_mAP = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, index=False, mesh_dir=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    img_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    mesh_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.5],
                    std=[0.5])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(MeshPreprocessor(train_set, root=dataset.images_dir, transform=img_transformer,
                                    mesh_dir=mesh_dir+'train/', mesh_transform=mesh_transformer,
                                    index=index),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

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
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    img_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    mesh_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        T.Normalize(mean=[0.5],
                    std=[0.5])
    ])

    if (testset is None):
        testset = dataset.gallery
        mesh_dir = mesh_dir + 'test/'
    else:
        mesh_dir = mesh_dir + 'train/'

    test_loader = DataLoader(
        MeshPreprocessor(testset, root=dataset.images_dir, transform=img_transformer,
                         mesh_dir=mesh_dir,
                         mesh_transform=mesh_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True)

    return test_loader


def create_model(args):
    model_1 = models.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=0)

    if args.init == '':
        print('No idnet init.')
    else:
        checkpoint = load_checkpoint(args.init)
        model_1.load_state_dict(checkpoint, strict=False) # JVTC
        # model_1.load_state_dict(checkpoint['state_dict'], strict=False) # MMCL, ACT


    model_1.cuda()
    # model_1 = nn.DataParallel(model_1)
    # model_1_ema = nn.DataParallel(model_1_ema)

    return model_1


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    config = get_config(args.config)

    # sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size*8, args.workers)

    cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size*8, args.workers,
                                     testset=dataset_target.train)

    train_display_loader = get_display_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
                                              testset=dataset_target.train, mesh_dir=args.mesh_dir)
    test_display_loader = get_display_loader(dataset_target, args.height, args.width, args.batch_size, args.workers,
                                             testset=None, mesh_dir=args.mesh_dir)

    display_size = config['display_size']
    train_display_images = torch.stack([train_display_loader.dataset[i][0] for i in range(display_size)]).cuda()
    train_display_meshes = torch.stack([train_display_loader.dataset[i][1] for i in range(display_size)]).cuda()
    train_display_meshes_nv = torch.stack([train_display_loader.dataset[i][2] for i in range(display_size)]).cuda()

    test_display_images = torch.stack([test_display_loader.dataset[i][0] for i in range(display_size)]).cuda()
    test_display_meshes = torch.stack([test_display_loader.dataset[i][1] for i in range(display_size)]).cuda()
    test_display_meshes_nv = torch.stack([test_display_loader.dataset[i][2] for i in range(display_size)]).cuda()


    if args.stage == 3:
        train_writer = SummaryWriter(osp.join(args.output_path + "/logs", args.name, 'stage3'))
    else:
        train_writer = SummaryWriter(osp.join(args.output_path + "/logs", args.name))
    output_directory = osp.join(args.output_path + "/outputs", args.name)

    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

    checkpoint_directory_stage3 = osp.join(checkpoint_directory, 'stage3')
    os.makedirs(checkpoint_directory_stage3, exist_ok=True)

    # Create model
    model_1 = create_model(args)

    num_gpu = torch.cuda.device_count()

    # Trainer
    idnet_freeze = True if args.stage == 2 else False
    trainer = DGNet_Trainer(config, model_1, idnet_freeze).cuda()
    iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if args.resume else 0

    # Evaluator
    evaluator_1 = Evaluator(trainer.id_net)

    # Init memory
    if args.stage == 3:
        dict_f1, _ = extract_features(trainer.id_net, cluster_loader, print_freq=50)
        cf = torch.stack(list(dict_f1.values()))
        trainer.memory.features = F.normalize(cf, dim=1).cuda()
        print('Memory initialized.')

    _, mAP_1 = evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

    for epoch in range(args.epochs):

        if args.stage == 2:
            train_loader_target = get_train_loader(dataset_target, args.height, args.width, args.batch_size,
                                             args.workers, num_instances=0, iters=None, trainset=None, mesh_dir=args.mesh_dir)
        else:
            cf = trainer.memory.features.clone()

            rerank_dist = compute_jaccard_distance(cf, k1=args.k1, k2=6)

            tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2
            tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
            tri_mat = np.sort(tri_mat, axis=None)
            top_num = np.round(args.rho * tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()
            print('eps in cluster: {:.3f}'.format(eps))
            print('Clustering and labeling...')
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            labels = cluster.fit_predict(rerank_dist)

            trainer.memory.labels = torch.from_numpy(labels).cuda()

            num_ids = len(set(labels)) - (1 if -1 in labels else 0)

            # change pseudo labels
            pseudo_labeled_dataset = []
            pseudo_outlier_dataset = []
            pseudo_outlier_index = []
            outliers = 0
            for i, ((fname, _, cid), label) in enumerate(zip(dataset_target.train, labels)):
                if label == -1:
                    pseudo_outlier_dataset.append((fname, label.item(), cid))
                    pseudo_outlier_index.append(i)
                    outliers += 1
                else:
                    pseudo_labeled_dataset.append((fname, label.item(), cid, i))
            print('Epoch {} has {} labeled samples of {} ids and {} unlabeled samples'.
                  format(epoch, len(pseudo_labeled_dataset), num_ids, len(pseudo_outlier_dataset)))
            train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                           args.batch_size, args.workers, num_instances=0, iters=None,
                                           trainset=pseudo_labeled_dataset, index=True, mesh_dir=args.mesh_dir)

        train_loader_target.new_epoch()

        for it in range(len(train_loader_target)):
            x_img, x_mesh, x_mesh_nv, fname, pid, camid, index = train_loader_target.next()
            x_img, x_mesh, x_mesh_nv = x_img.cuda(), x_mesh.cuda(), x_mesh_nv.cuda()

            pid = pid.cuda()
            index = index.cuda()

            x_recon, x_nv, x_nv2recon, feat, feat_recon, feat_nv, feat_nv2recon, \
            f, f_recon, f_nv, f_nv2recon = trainer.forward(x_img, x_mesh, x_mesh_nv)

            if num_gpu > 1:
                trainer.module.dis_update(x_img, x_recon, x_nv, x_nv2recon, config, num_gpu)
                if iterations % int(config['gen_iters']) == 0:
                    trainer.module.gen_update(x_img, x_recon, x_nv, x_nv2recon, feat, feat_recon, feat_nv,
                                              feat_nv2recon, f, f_recon, f_nv, f_nv2recon, pid, index, config, iterations, num_gpu)
            else:
                trainer.dis_update(x_img, x_recon, x_nv, x_nv2recon, config, num_gpu=1)
                if iterations % int(config['gen_iters']) == 0:
                    trainer.gen_update(x_img, x_recon, x_nv, x_nv2recon, feat, feat_recon, feat_nv, feat_nv2recon, f,
                                       f_recon, f_nv, f_nv2recon, pid, index, config, iterations, num_gpu=1)
            torch.cuda.synchronize()

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Epoch {} Iteration {}".format(epoch, iterations + 1))

            if num_gpu == 1:
                write_loss(iterations, trainer, train_writer)
            else:
                write_loss(iterations, trainer.module, train_writer)

            # Write images
            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    if num_gpu > 1:
                        test_image_outputs = trainer.module.sample(test_display_images, test_display_meshes,
                                                                   test_display_meshes_nv)
                    else:
                        test_image_outputs = trainer.sample(test_display_images, test_display_meshes,
                                                            test_display_meshes_nv)

                display_images(test_display_images, test_display_meshes_nv, test_image_outputs, 'test',
                               train_writer, iterations + 1)
                del test_image_outputs

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    if num_gpu > 1:
                        image_outputs = trainer.module.sample(train_display_images, train_display_meshes,
                                                              train_display_meshes_nv)
                    else:
                        image_outputs = trainer.sample(train_display_images, train_display_meshes,
                                                       train_display_meshes_nv)

                display_images(train_display_images, train_display_meshes_nv, image_outputs, 'train', train_writer,
                               iterations + 1)
                del image_outputs
            iterations += 1

            # Save network weights
            if args.stage == 2:
                if (iterations + 1) % config['snapshot_save_iter'] == 0:
                    if num_gpu > 1:
                        trainer.module.save(checkpoint_directory, iterations)
                    else:
                        trainer.save(checkpoint_directory, iterations)

        if (args.stage==3):
            trainer.update_learning_rate()
            _, mAP_1 = evaluator_1.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
            # if num_gpu > 1:
            #     trainer.module.save(checkpoint_directory_stage3, iterations)
            # else:
            #     trainer.save(checkpoint_directory_stage3, iterations)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MV Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='dukemtmc-reid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='ft_net',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    # training configs
    parser.add_argument('--init', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--stage', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    # parser.add_argument('--mesh-dir', type=str, metavar='PATH',
    #                     default=osp.join(working_dir, '/data/stars/user/yaowang/data/reid/market/'))
    parser.add_argument('--mesh-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '/data/stars/user/yaowang/data/reid/DukeMTMC/'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    # cluster
    parser.add_argument('--rho', type=float, default=1.6e-3,
                        help='rho percentage')
    parser.add_argument('--k1', type=int, default=20,
                        help="k1 for re-ranking")
    # gan config
    parser.add_argument('--config', type=str, default='configs/latest.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest', help="outputs path")

    main()
