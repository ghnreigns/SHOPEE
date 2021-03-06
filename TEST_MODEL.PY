import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from loss import ArcModule
from config import CONFIG


class SHOPEE_EfficientNetB4(nn.Module):
    def __init__(
        self,
        num_classes=11014,
        dropout=0.3,
        embedding_size=512,
        backbone="vgg16",
        pretrained=True,
    ):
        super(SHOPEE_EfficientNetB4, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")
        self.adaptive_pooling = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.in_features = self.backbone.num_features

        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)

        self.fc1 = nn.Linear(self.in_features, self.embedding_size)
        self.bn2 = nn.BatchNorm1d(self.embedding_size)
        self.ArcMargin = ArcModule(
            in_features=self.embedding_size,
            out_features=self.num_classes,
            **CONFIG["ArcFace"]
        )

    def forward(self, x, labels=None):

        features = self.backbone.forward_features(x)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.adaptive_pooling(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)

        if labels is not None:
            arcfaceLogits = self.ArcMargin(features, labels)
            return arcfaceLogits
        return features


class TEST_MODEL(nn.Module):
    def __init__(
        self,
        channel_size,
        out_feature,
        dropout=0.2,
        backbone="eca_nfnet_l1",
        pretrained=True,
    ):
        super(TEST_MODEL, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.channel_size = channel_size
        self.out_feature = out_feature
        # self.in_features = self.backbone.head.fc.in_features
        self.in_features = self.backbone.classifier.in_features
        print(self.in_features)
        self.margin = ArcModule(
            in_features=self.channel_size,
            out_features=self.out_feature,
            **CONFIG["ArcFace"]
        )
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # print("margin", self.margin)
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)

        self.fc1 = nn.Linear(self.in_features, self.channel_size)  # * 16 * 16
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):

        features = self.backbone.forward_features(x)

        print("features at backbone", features.shape)
        features = self.bn1(features)
        print("after bn1", features.shape)
        features = self.dropout(features)
        print("after dropout2d", features.shape)
        features = self.adaptive_avg_pool(features)
        features = features.view(features.size(0), -1)

        print("features at view", features.shape)
        features = self.fc1(features)
        print("after fc1", features.shape)
        features = self.bn2(features)
        features = F.normalize(features)
        # print(features, features.shape)
        if labels is not None:
            MARGIN = self.margin(features, labels)
            # print("Margin | {} | {}".format(MARGIN, MARGIN.shape))
            return self.margin(features, labels)
        return features