import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        # print(self.W.shape)
        # nn.init.kaiming_uniform_(self.W)
        nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        # Step 1:
        x_norm = F.normalize(x)
        # print(x_norm)
        W_norm = F.normalize(self.W, dim=0)
        # W_norm = F.normalize(self.W)
        # Step 2:
        ArcFaceLogits = x_norm @ W_norm
        # ArcFaceLogits = F.linear(x_norm, W_norm)
        return ArcFaceLogits


class ArcFaceLoss(nn.Module):
    def __init__(
        self,
        scale=30.0,
        margin=0.50,
        easy_margin=False,
        ls_eps=0.0,
    ):
        super().__init__()

        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps  # label smoothing
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # self.th = torch.tensor(math.cos(math.pi - margin))
        # self.mm = torch.tensor(math.sin(math.pi - margin) * margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self, cosine_theta_logits, label
    ):  # corresponds to cosine_theta_logits = ArcFaceCosineLogits

        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine_theta_logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            cosine = cosine.float()  # WHEN USE AMP MUST FLOAT
            phi = phi.float()  # WHEN USE AMP MUST FLOAT
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device="cuda")
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        # print(output)
        # cross_entropy_loss = nn.CrossEntropyLoss()(output, label)
        return output  # cross_entropy_loss #TODO: dont put cross entropy here, very hard to keep track.

        # # cos_th = F.linear(inputs, F.normalize(self.weight))
        # cos_th = cosine_theta_logits
        # cos_th = (
        #     cos_th.float()
        # )  ###TODO: do not put this below as it will cause numerical instability.
        # cos_th = cos_th.clamp(-1, 1)
        # sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        # cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m

        # cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        # cond_v = cos_th - self.th
        # cond = cond_v <= 0
        # cos_th_m[cond] = (cos_th - self.mm)[cond]

        # if labels.dim() == 1:
        #     labels = labels.unsqueeze(-1)
        # onehot = torch.zeros(cos_th.size()).cuda()
        # labels = labels.type(torch.LongTensor).cuda()
        # onehot.scatter_(1, labels, 1.0)
        # outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        # outputs = outputs * self.scale
        # return outputs