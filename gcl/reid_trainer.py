from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F
from .loss import NCESoftmaxLoss
from torch.nn import MSELoss, CrossEntropyLoss
from gcl.loss import CrossEntropyLabelSmooth, SoftEntropy, TripletLoss, SoftTripletLoss, NNLoss, TrackletContrastiveLoss, MultiLabelContrastiveLoss
from .utils.meters import AverageMeter
# from mmt.loss import CrossEntropyLabelSmooth, SoftEntropy, TripletLoss, SoftTripletLoss
from .evaluation_metrics import accuracy
from gcl.models.memory import PretrainMemory

class MTTrainer(object):
    def __init__(self, model_1, model_1_ema, num_cluster=500, alpha=0.999):
        super(MTTrainer, self).__init__()
        self.model_1 = model_1
        self.num_cluster = num_cluster

        self.model_1_ema = model_1_ema
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_target,
            optimizer, ce_soft_weight=0.5, tri_soft_weight=0.5, print_freq=1, train_iters=200, centers=None):
        self.model_1.train()
        self.model_1_ema.train()

        centers = centers.cuda()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = [AverageMeter(),AverageMeter()]
        losses_tri = [AverageMeter(),AverageMeter()]
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_1, inputs_2, targets = self._parse_data(target_inputs)

            # forward
            _, f_out_t1 = self.model_1(inputs_1)
            with torch.no_grad():
                _, f_out_t1_ema = self.model_1_ema(inputs_2)

            p_out_t1 = torch.matmul(f_out_t1, centers.transpose(1, 0)) / 0.5
            # # print(torch.softmax(p_out_t1_ema,dim=1))
            # # print(torch.argmax(torch.softmax(p_out_t1_ema,dim=1),dim=1))
            #
            # # print(targets_u)
            # # print(torch.argmax(targets_u,dim=1))
            # # dis = []
            # # for j in range(len(fnames)):
            # #     moving_avg_pred[fnames[j].split('/')[-1]] = p_out_t1_ema.data.cpu()*0.1 + moving_avg_pred[fnames[j].split('/')[-1]]*0.9
            # #     if p_out_t1_ema.argmax(dim=1)[j] != moving_avg_pred[fnames[j].split('/')[-1]].argmax(dim=1)[j]:
            # #         dis.append(j)
            # # print(dis)
            #
            loss_ce_1 = self.criterion_ce(p_out_t1, targets)

            loss_tri_1 = self.criterion_tri(f_out_t1, f_out_t1, targets)

            # loss_ce_soft = self.criterion_ce_soft(p_out_t1, p_out_t1_ema)
            # loss_tri_soft = self.criterion_tri_soft(f_out_t1, f_out_t1_ema, targets)
            #
            # loss_mse = self.criterion_mse(torch.softmax(p_out_t1_u, dim=1), targets_u)*0.1
            # # loss_mse = self.criterion_mse(p_out_t1_u,p_out_t1_ema_u)*0.1
            #
            # loss = (loss_ce_1)*(1-ce_soft_weight) + \
            #          (loss_tri_1)*(1-tri_soft_weight) + \
            #          loss_ce_soft*ce_soft_weight + loss_tri_soft*tri_soft_weight

            loss = loss_ce_1 + loss_tri_1

            # # loss_ce_soft = self.criterion_ce_soft(mixed_p_out_t1, mixed_target)
            #
            # # loss = loss_ce_1 + loss_tri_1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch * len(data_loader_target) + i)

            prec_1, = accuracy(p_out_t1.data, targets.data)

            losses_ce[0].update(loss_ce_1.item())
            losses_tri[0].update(loss_tri_1.item())
            # losses_ce_soft.update(loss_ce_soft.item())
            # losses_tri_soft.update(loss_tri_soft.item())
            precisions[0].update(prec_1[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} / {:.3f}\t'
                      'Loss_tri {:.3f} / {:.3f}\t'
                      'Loss_ce_soft {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Prec {:.2%} / {:.2%}\t'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce[0].avg, losses_ce[1].avg,
                              losses_tri[0].avg, losses_tri[1].avg,
                              losses_ce_soft.avg, losses_tri_soft.avg,
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, _, pids, _, index = inputs
        # imgs_1, imgs_2, pids = inputs
        inputs_1 = imgs_1.cuda()
        inputs_2 = imgs_2.cuda()
        targets = pids.cuda()
        index = index.cuda()
        return inputs_1, inputs_2, targets


class SpCLTrainer_USL(object):
    def __init__(self, encoder, memory):
        super(SpCLTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, _, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)

            # compute loss with the hybrid memory
            loss = self.memory(f_out, indexes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)