import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceNormalize(nn.Module):
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


class VanillaArcFace(nn.Module):
    def __init__(
        self,
        num_classes,
        scale=1,
        margin=0.4,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, cosine_theta_logits, label):

        # this prevents nan when a value slightly crosses 1.0 due to numerical error
        cosine_theta_logits = cosine_theta_logits.clip(-1 + 1e-7, 1 - 1e-7)
        # Step 3:
        arcosine = cosine_theta_logits.arccos()
        # Step 4:
        arcosine += F.one_hot(label, num_classes=self.num_classes) * self.margin
        # Step 5:
        final_cosine_logits = arcosine.cos()
        # Step 6:
        final_cosine_logits *= self.scale
        return final_cosine_logits