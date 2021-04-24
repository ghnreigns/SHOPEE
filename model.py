import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from loss import ArcModule
from config import CONFIG


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

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")

        self.in_features = self.backbone.num_features

        self.BN_DR_FC_BN = torch.nn.Sequential(  # Now since the classifier head is reset, we replace it with                                                                             #
            torch.nn.BatchNorm1d(
                self.in_features
            ),  # BN_DR_FC_BN - the proposed layer architecture by ArcFace.
            torch.nn.Dropout(
                p=0.1
            ),  # Applies Batch Normalization over a 2D or 3D input and since
            torch.nn.Linear(
                self.in_features, self.embedding_size
            ),  # the previous layer output is [16, 4096, 1] we ise BN_1D
            torch.nn.BatchNorm1d(
                self.embedding_size
            ),  # and lastly connect with Dropout, FC and BN again. where
        )  # fc_dim is the embeddings we want, which is proposed to be 512.

        self.ArcMargin = ArcModule(
            in_features=self.embedding_size,
            out_features=self.num_classes,
            **CONFIG["ArcFace"]
        )

    def forward(self, x, labels=None):

        features = self.backbone.forward(x)

        features = self.BN_DR_FC_BN(features)
        if labels is not None:
            arcfaceLogits = self.ArcMargin(features, labels)
            # print("Margin | {} | {}".format(MARGIN, MARGIN.shape))
            return arcfaceLogits
        return features


class SHOPEE_HIRE_ME_MODEL_IDKSHAPEERROR(nn.Module):
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
            in_features=self.embedding_size,
            out_features=self.num_classes,
            **CONFIG["ArcFace"]
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


class HN_NFNET_l0(nn.Module):
    def __init__(
        self,
        channel_size,
        out_feature,
        dropout=0.2,
        backbone="eca_nfnet_l1",
        pretrained=True,
    ):
        super(HN_NFNET_l0, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.channel_size = channel_size
        self.out_feature = out_feature
        self.in_features = self.backbone.head.fc.in_features
        print(self.in_features)
        self.margin = ArcModule(
            in_features=self.channel_size,
            out_features=self.out_feature,
            **CONFIG["ArcFace"]
        )
        # print("margin", self.margin)
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features * 16 * 16, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):

        features = self.backbone.forward_features(x)

        # print("features at backbone", features.shape)
        features = self.bn1(features)
        features = self.dropout(features)
        print(features.shape)
        features = features.view(features.size(0), -1)
        # print("features at view", features.shape)
        features = self.fc1(features)
        # print(features.shape)
        features = self.bn2(features)
        features = F.normalize(features)
        # print(features, features.shape)
        if labels is not None:
            MARGIN = self.margin(features, labels)
            # print("Margin | {} | {}".format(MARGIN, MARGIN.shape))
            return self.margin(features, labels)
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
        self.in_features = self.backbone.head.fc.in_features
        print(self.in_features)
        self.margin = ArcModule(
            in_features=self.channel_size,
            out_features=self.out_feature,
            **CONFIG["ArcFace"]
        )
        # print("margin", self.margin)
        self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc1 = nn.Linear(self.in_features, self.channel_size)
        self.bn2 = nn.BatchNorm1d(self.channel_size)

    def forward(self, x, labels=None):

        features = self.backbone.forward_features(x)

        print("features at backbone", features.shape)
        features = self.bn1(features)
        print("after bn1", features.shape)
        features = self.dropout(features)
        print("after dropout2d", features.shape)
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