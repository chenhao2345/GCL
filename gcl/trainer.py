"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from gcl.models.gan import AdaINGen, MsImageDis
from gcl.models.memory import Memory
from gcl.utils.gan_utils import get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import copy
import os
import cv2
import numpy as np
import random
import yaml
import torch.nn.functional as F

######################################################################
# Load model

class DGNet_Trainer(nn.Module):
    def __init__(self, hyperparameters, id_net, idnet_freeze, gpu_ids=[0]):
        super(DGNet_Trainer, self).__init__()
        lr_g = hyperparameters['lr_g']
        lr_d = hyperparameters['lr_d']
        if idnet_freeze:
            lr_g = hyperparameters['lr_g']
            lr_d = hyperparameters['lr_d']

        # Initiate the networks
        self.gen = AdaINGen(hyperparameters['input_dim'], hyperparameters['gen'],
                            fp16=False)
        self.dis = MsImageDis(3, hyperparameters['dis'], fp16=False)
        self.id_net = id_net
        self.idnet_freeze = idnet_freeze

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis.parameters())  # + list(self.dis_b.parameters())
        gen_params = list(self.gen.parameters())  # + list(self.gen_b.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr_d, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr_g, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])

        # id params
        id_params = list(self.id_net.parameters())
        lr2 = hyperparameters['lr_id']
        self.id_opt = torch.optim.SGD([
            {'params': id_params, 'lr': lr2},
        ], weight_decay=hyperparameters['weight_decay'], momentum=0.9, nesterov=True)

        # Memory bank
        self.memory = Memory(num_features=2048, num_samples=12936, temp=hyperparameters['temperature'],
                                   momentum=hyperparameters['momentum'], K=hyperparameters['K'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)
        self.id_scheduler = get_scheduler(self.id_opt, hyperparameters)
        self.id_scheduler.gamma = hyperparameters['gamma2']

    def recon_criterion(self, input, target):
        diff = input - target.detach()
        return torch.mean(torch.abs(diff[:]))

    def forward(self, x_img, x_mesh, x_mesh_nv):
        if self.idnet_freeze:
            self.id_net.eval()
        else:
            self.id_net.train()

        # encode
        s_org = self.gen.encode(x_mesh)
        s_nv = self.gen.encode(x_mesh_nv)
        feat, f = self.id_net(x_img, mode='fix')

        # decode
        x_recon = self.gen.decode(s_org, feat)
        x_nv = self.gen.decode(s_nv, feat)

        # encode again
        feat_recon, f_recon = self.id_net(x_recon, mode='fix')
        feat_nv, f_nv = self.id_net(x_nv, mode='fix')

        # decode again
        x_nv2recon = self.gen.decode(s_org, feat_nv)
        feat_nv2recon, f_nv2recon = self.id_net(x_nv2recon, mode='fix')

        return x_recon, x_nv, x_nv2recon, feat, feat_recon, feat_nv, feat_nv2recon, f, f_recon, f_nv, f_nv2recon

    def gen_update(self, x, x_recon, x_nv, x_nv2recon, feat, feat_recon, feat_nv, feat_nv2recon, f, f_recon,
                   f_nv, f_nv2recon, pid, index, hyperparameters, iterations, num_gpu):

        self.gen_opt.zero_grad()
        self.id_opt.zero_grad()

        # auto-encoder image reconstruction
        self.loss_gen_recon_x = self.recon_criterion(x_recon, x)
        self.loss_gen_cycrecon_x = self.recon_criterion(x_nv2recon, x)

        # feature reconstruction
        self.loss_gen_recon_f = self.recon_criterion(feat_recon, feat)
        self.loss_gen_nv2recon_f = self.recon_criterion(feat_nv2recon, feat)

        # ID loss AND Tune the Generated image
        if self.idnet_freeze:
            self.memory_loss_id = 0
        else:
            self.memory_loss_id = self.memory(F.normalize(f), F.normalize(f_nv), index)

        # adv loss
        if num_gpu > 1:
            self.loss_gen_adv_recon = self.dis.module.calc_gen_loss(self.dis, x_recon)
            self.loss_gen_adv_nv = self.dis.module.calc_gen_loss(self.dis, x_nv)
            self.loss_gen_adv_nv2recon = self.dis.module.calc_gen_loss(self.dis, x_nv2recon)
        else:
            self.loss_gen_adv_recon = self.dis.calc_gen_loss(self.dis, x_recon)
            self.loss_gen_adv_nv = self.dis.calc_gen_loss(self.dis, x_nv)
            self.loss_gen_adv_nv2recon = self.dis.calc_gen_loss(self.dis, x_nv2recon)

        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_recon + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_nv + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_nv2recon + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x + \
                              hyperparameters['recon_f_w'] * self.loss_gen_recon_f + \
                              hyperparameters['recon_f_w'] * self.loss_gen_nv2recon_f + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x + \
                              hyperparameters['memory_id_w'] * self.memory_loss_id

        self.loss_gen_total.backward()
        self.gen_opt.step()
        if not self.idnet_freeze:
            self.id_opt.step()
            if iterations % 10 == 0:
                print('LR_id:{}\t'
                      'L_memory_id:{:.3f}\t'
                      .format(self.id_opt.param_groups[0]['lr'],
                              hyperparameters['memory_id_w'] * self.memory_loss_id))

    def sample(self, x_img, x_mesh, x_mesh_nv):
        self.eval()
        # encode
        s_org = self.gen.encode(x_mesh)
        s_nv = self.gen.encode(x_mesh_nv)
        feat = self.id_net(x_img, mode='display')

        # decode
        x_recon = self.gen.decode(s_org, feat)
        x_nv = self.gen.decode(s_nv, feat)

        # encode again
        feat_nv = self.id_net(x_nv, mode='display')

        # decode again
        x_nv2recon = self.gen.decode(s_org, feat_nv)

        self.train()

        return x_recon, x_nv, x_nv2recon

    def sample_recon(self, x_img, x_mesh):
        self.eval()
        # encode
        s_org = self.gen.encode(x_mesh)
        feat = self.id_net(x_img, mode='display')

        # decode
        x_recon = self.gen.decode(s_org, feat)

        return x_recon

    def sample_nv(self, x_img, x_mesh_nv):
        self.eval()
        # encode
        s_nv = self.gen.encode(x_mesh_nv)
        feat = self.id_net(x_img, mode='display')

        # decode
        x_nv = self.gen.decode(s_nv, feat)
        return x_nv

    def dis_update(self, x, x_recon, x_nv, x_nv2recon, hyperparameters, num_gpu):
        self.dis_opt.zero_grad()
        # D loss
        if num_gpu > 1:
            self.loss_dis_recon, _ = self.dis.module.calc_dis_loss(self.dis, x_recon.detach(), x)
            self.loss_dis_nv, _ = self.dis.module.calc_dis_loss(self.dis, x_nv.detach(), x)
            self.loss_dis_nv2recon, _ = self.dis.module.calc_dis_loss(self.dis, x_nv2recon.detach(), x)
        else:
            self.loss_dis_recon, _ = self.dis.calc_dis_loss(self.dis, x_recon.detach(), x)
            self.loss_dis_nv, _ = self.dis.calc_dis_loss(self.dis, x_nv.detach(), x)
            self.loss_dis_nv2recon, _ = self.dis.calc_dis_loss(self.dis, x_nv2recon.detach(), x)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_recon + \
                              hyperparameters['gan_w'] * self.loss_dis_nv2recon + \
                              hyperparameters['gan_w'] * self.loss_dis_nv

        self.loss_dis_total.backward()

        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.id_scheduler is not None:
            self.id_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict['dis'])
        # Load ID dis
        last_model_name = get_model_list(checkpoint_dir, "id")
        state_dict = torch.load(last_model_name)
        self.id_net.load_state_dict(state_dict['id'])
        # Load optimizers
        try:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])
            self.id_opt.load_state_dict(state_dict['id'])
        except:
            pass
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, num_gpu=1):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        id_name = os.path.join(snapshot_dir, 'id_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': self.gen.state_dict()}, gen_name)
        # if num_gpu > 1:
        #     torch.save({'dis': self.dis.module.state_dict()}, dis_name)
        # else:
        torch.save({'dis': self.dis.state_dict()}, dis_name)
        torch.save({'id': self.id_net.state_dict()}, id_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'id': self.id_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

