import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcModule(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale=64,
        margin=0.5,
        easy_margin=False,
        ls_eps=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = torch.tensor(math.cos(math.pi - margin))
        self.mm = torch.tensor(math.sin(math.pi - margin) * margin)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # cos_th = cos_th.float()
        # cos_th_m = cos_th_m.float() #TODO: IF YOU CHANGE TO FLOAT, THE LOSS BECOMES HIGH FOR SOME REASON.

        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.scale
        return outputs