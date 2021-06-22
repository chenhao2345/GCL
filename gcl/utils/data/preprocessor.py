from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import random
import math
from PIL import Image
import torchvision.transforms.functional as TF
from gcl.utils.data import transforms as T


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        img2 = img.copy()

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img2)
        else:
            raise NotImplementedError

        return img1, img2, fname, pid, camid, index

class MutualPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform1=None,  transform2=None):
        super(MutualPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.mutual = self.transform2 is not None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform1 is not None:
            img = self.transform1(img)

        return img, fname, pid, camid, index

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        # index = i
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')
        img2 = img.copy()

        if self.transform1 is not None and self.transform2 is not None:
            img1 = self.transform1(img)
            img2 = self.transform2(img2)
        else:
            raise NotImplementedError

        return img1, img2, fname, pid, camid, index


class MeshPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mesh_dir=None, mesh_transform=None, index=False):
        super(MeshPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mesh_transform = mesh_transform
        self.mesh_dir = mesh_dir
        if 'msmt' in mesh_dir:
            self.imgs_dir = root + '/' + osp.join(*dataset[0][0].split('/')[:-2]) + '/'  # msmt
        else:
            self.imgs_dir = '/'+osp.join(*dataset[0][0].split('/')[:-1])+'/'  # market, duke
        self.index = index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_mesh_item(indices)

    def get_mesh(self, img_path, deg):
        mesh_org_dir = self.mesh_dir + 'render/'
        mesh_nv_dir = self.mesh_dir + 'render_%d/' % deg
        mesh_org_path = img_path.replace(self.imgs_dir, mesh_org_dir)
        mesh_nv_path = img_path.replace(self.imgs_dir, mesh_nv_dir)

        mesh_org = Image.open(mesh_org_path).convert('L')
        mesh_nv = Image.open(mesh_nv_path).convert('L')

        return mesh_org, mesh_nv

    def _get_mesh_item(self, index):
        if self.index:
            fname, pid, camid, i = self.dataset[index]
            index = i
        else:
            fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        deg = np.random.randint(low=1, high=7) * 45  # select a new view from 45 deg ~ 315 deg
        mesh_org, mesh_nv = self.get_mesh(fpath, deg)
        mesh_org = self.mesh_transform(mesh_org)
        mesh_nv = self.mesh_transform(mesh_nv)

        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)

        return img, mesh_org, mesh_nv, fname, pid, camid, index


class AllMeshPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mesh_dir=None, mesh_transform=None, index=False):
        super(AllMeshPreprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mesh_transform = mesh_transform
        self.mesh_dir = mesh_dir
        if 'msmt' in mesh_dir:
            self.imgs_dir = root + '/' + osp.join(*dataset[0][0].split('/')[:-2]) + '/'  # msmt
        else:
            self.imgs_dir = '/'+osp.join(*dataset[0][0].split('/')[:-1])+'/'  # market, duke
        self.index = index
        # print(self.imgs_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_mesh_item(indices)

    def get_mesh(self, img_path):
        mesh_org_dir = self.mesh_dir + 'render/'
        mesh_org_path = img_path.replace(self.imgs_dir, mesh_org_dir)
        mesh_org = Image.open(mesh_org_path).convert('L')
        all_mesh_nv=[]
        for deg in [45, 90, 135, 180, 225, 270, 315]:
            mesh_nv_dir = self.mesh_dir + 'render_%d/' % deg
            mesh_nv_path = img_path.replace(self.imgs_dir, mesh_nv_dir)
            all_mesh_nv.append(Image.open(mesh_nv_path).convert('L'))
        return mesh_org, all_mesh_nv

    def _get_mesh_item(self, index):
        if self.index:
            fname, pid, camid, i = self.dataset[index]
            index = i
        else:
            fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        # deg = np.random.randint(low=1, high=7) * 45  # select a new view from 45 deg ~ 315 deg
        mesh_org, all_mesh_nv = self.get_mesh(fpath)

        img = Image.open(fpath).convert('RGB')
        img = self.transform(img)
        mesh_org = self.mesh_transform(mesh_org)
        all_mesh_nv = [self.mesh_transform(mesh_nv) for mesh_nv in all_mesh_nv]
        return img, mesh_org, all_mesh_nv, fname, pid, camid, index
