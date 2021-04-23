import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from loss import ArcModule


class SHOPEE_HIRE_ME_MODEL(nn.Module):
    def __init__(
        self,
        num_classes=11014,
        dropout=0.3,
        embedding_size=512,
        backbone="vgg16",
        pretrained=True,
    ):
        super(SHOPEE_HIRE_ME_MODEL, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        self.in_features = self.backbone.head.fc.in_features
        self.margin = ArcModule(
            in_features=self.embedding_size, out_features=self.num_classes, s=64, m=0.5
        )
        # print("margin", self.margin)
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features, self.embedding_size)
        self.bn2 = nn.BatchNorm1d(self.embedding_size)
        print(self.num_classes, self.in_features)

    def forward(self, x, labels=None):
        print(x.shape)
        print(labels.shape)
        features = self.backbone.forward_features(x)
        features_2 = self.backbone.forward(x)
        # print(features_2.shape)
        print(features.shape)
        # print("features at backbone", features.shape)
        features = self.bn1(features)
        print(features.shape)
        print(features)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        print(features.shape)

        # print("features at view", features.shape)
        features = self.fc1(features)
        # print(features.shape)
        features = self.bn2(features)
        # print(features.shape)
        features = F.normalize(features)
        print(features.shape)
        # print(features, features.shape)
        if labels is not None:
            MARGIN = self.margin(features, labels)
            # print("Margin | {} | {}".format(MARGIN, MARGIN.shape))
            return self.margin(features, labels)
        return features