from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'ft_net']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1,
            # resnet.relu,
            resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        # if not pretrained:
        #     self.reset_params()

    def forward(self, x, mode='test'):
        bs = x.size(0)
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if (self.training is False):
            bn_x = F.normalize(bn_x)
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return bn_x

        return x, prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNet.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.maxpool.state_dict())
        self.base[3].load_state_dict(resnet.layer1.state_dict())
        self.base[4].load_state_dict(resnet.layer2.state_dict())
        self.base[5].load_state_dict(resnet.layer3.state_dict())
        self.base[6].load_state_dict(resnet.layer4.state_dict())

def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)


# # for UDATP and ACT
# class ft_net(nn.Module):
#     def __init__(self, **kwargs):
#         super(ft_net, self).__init__()
#         self.base = torchvision.models.resnet50(pretrained=True)
#         self.base.layer4[0].conv2.stride = (1, 1)
#         self.base.layer4[0].downsample[0].stride = (1, 1)
#
#         self.part = 4
#         self.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.feat_bn = nn.BatchNorm1d(2048)
#         # self.classifier = resnet50.classifier
#
#     def forward(self, x, mode=None):
#         for name, module in self.base._modules.items():
#             if name == 'avgpool': break
#             x = module(x)
#         f = self.partpool(x)
#         f = f.view(f.size(0), f.size(1) * self.part)  # 8 * 2048 4*1
#         if mode == 'display':
#             return f
#         x = self.avgpool(x)  # 8 * 2048 1*1
#         x = x.view(x.size(0), x.size(1))
#         bn_x = self.feat_bn(x)
#         # x = self.classifier(x)
#         if mode == 'fix' and (not self.training):
#             return f, bn_x
#
#         if not self.training:
#             return F.normalize(bn_x)
#         return f, bn_x

# # for MLC
# class ft_net(nn.Module):
#     def __init__(self, **kwargs):
#         super(ft_net, self).__init__()
#         self.base = torchvision.models.resnet50(pretrained=True)
#         self.base.layer4[0].conv2.stride = (1, 1)
#         self.base.layer4[0].downsample[0].stride = (1, 1)
#
#         self.part = 4
#         self.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.bn = nn.BatchNorm1d(2048)
#         # self.classifier = resnet50.classifier
#
#     def forward(self, x, mode=None):
#         for name, module in self.base._modules.items():
#             if name == 'avgpool': break
#             x = module(x)
#         f = self.partpool(x)
#         f = f.view(f.size(0), f.size(1) * self.part)  # 8 * 2048 4*1
#         if mode == 'display':
#             return f
#         x = self.avgpool(x)  # 8 * 2048 1*1
#         x = x.view(x.size(0), x.size(1))
#         bn_x = self.bn(x)
#         # x = self.classifier(x)
#         if mode == 'fix' and (not self.training):
#             return f, bn_x
#
#         if not self.training:
#             return F.normalize(bn_x)
#         return f, bn_x


# for JVCT
class ft_net(nn.Module):
    def __init__(self, **kwargs):
        super(ft_net, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

        self.part = 4
        self.partpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feat = nn.Linear(2048, 512)
        self.feat_bn = nn.BatchNorm1d(512)
        # self.classifier = resnet50.classifier

    def forward(self, x, mode=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        f = self.partpool(x)
        f = f.view(f.size(0), f.size(1) * self.part)  # 8 * 2048 4*1
        if mode == 'display':
            return f
        x = self.avgpool(x)  # 8 * 2048 1*1
        x = x.view(x.size(0), x.size(1))
        x = self.feat(x)
        bn_x = self.feat_bn(x)
        # x = self.classifier(x)
        if mode == 'fix' and (not self.training):
            return f, bn_x

        if not self.training:
            return F.normalize(bn_x)
        return f, bn_x
