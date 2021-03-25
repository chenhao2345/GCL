import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd


class Memory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.07, momentum=0.2, K=8192):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.K = K

        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, f, f_nv, indexes):
        # inputs: B*2048, features: L*2048
        batchSize = f.size(0)
        k = self.features.clone()
        l_pos = torch.bmm(f_nv.view(batchSize, 1, -1), k[indexes].view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        labels = self.labels
        batch_labels = labels[indexes]
        # print(batch_labels)
        mat = torch.matmul(f, k.transpose(0, 1))
        mat_sim = torch.matmul(f_nv, k.transpose(0, 1))
        sample_size = self.K
        positives_wogan = []
        positives = []
        negatives = []
        negatives_wogan = []
        for i in range(batchSize):
            pos_labels = (labels == batch_labels[i])
            pos = mat[i, pos_labels]
            perm = torch.randperm(pos.size(0))
            idx = perm[:1]
            positives_wogan.append(pos[idx])

            pos_labels = (labels==batch_labels[i])
            pos = mat_sim[i, pos_labels]
            perm = torch.randperm(pos.size(0))
            idx = perm[:1]
            positives.append(pos[idx])

            neg_labels = (labels!=batch_labels[i])
            neg = mat_sim[i, neg_labels]
            perm = torch.randperm(neg.size(0))
            idx = perm[:sample_size]
            negatives.append(neg[idx])

            # neg_labels = (labels != batch_labels[i])
            # neg = mat[i, neg_labels]
            # perm = torch.randperm(neg.size(0))
            # idx = perm[:sample_size]
            # negatives_wogan.append(neg[idx])
        positives_wogan = torch.stack(positives_wogan)
        # negatives_wogan = torch.stack(negatives_wogan)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)

        self_out = torch.cat((l_pos, negatives), dim=1) / self.temp
        inter_out = torch.cat((positives, negatives), dim=1) / self.temp
        inter_out_wogan = torch.cat((positives_wogan, negatives), dim=1) / self.temp

        targets = torch.zeros([batchSize]).cuda().long()
        memory_loss = self.criterion(self_out, targets)+self.criterion(inter_out, targets)+self.criterion(inter_out_wogan, targets)

        # # update memory
        with torch.no_grad():
            weight_pos = torch.index_select(self.features, 0, indexes.view(-1))
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(f, 1 - self.momentum))
            updated_weight = F.normalize(weight_pos)
            self.features.index_copy_(0, indexes, updated_weight)

        return memory_loss


