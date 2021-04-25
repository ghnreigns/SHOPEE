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


class SHOPEE_PLEASE_HIRE_US(nn.Module):
    def __init__(
        self,
        num_classes=11014,
        dropout=0.3,
        embedding_size=512,
        backbone="vgg16",
        pretrained=True,
    ):
        super(SHOPEE_PLEASE_HIRE_US, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.adaptive_pooling = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")

        in_features = self.backbone.num_features

        self.BN_DR_FC_BN = torch.nn.Sequential(  # Now since the classifier head is reset, we replace it with                                                                             #
            torch.nn.BatchNorm1d(
                self.in_features
            ),  # BN_DR_FC_BN - the proposed layer architecture by ArcFace.
            torch.nn.Dropout(
                p=0.0
            ),  # Applies Batch Normalization over a 2D or 3D input and since
            torch.nn.Linear(
                self.in_features, self.embedding_size
            ),  # the previous layer output is [16, 4096, 1] we ise BN_1D
            torch.nn.BatchNorm1d(
                self.embedding_size
            ),  # and lastly connect with Dropout, FC and BN again. where
        )  # fc_dim is the embeddings we want, which is proposed to be 512.
        self._init_params()

        self.ArcMargin = ArcModule(
            in_features=self.embedding_size,
            out_features=self.num_classes,
            **CONFIG["ArcFace"]
        )

    def _init_params(self):
        # CUSTOM INIT note that try to think if there is a more
        # suggestive way to init the layer weights instead of using [1]

        nn.init.constant_(self.BN_DR_FC_BN[0].weight, 1)
        nn.init.constant_(self.BN_DR_FC_BN[0].bias, 0)
        nn.init.xavier_normal_(self.BN_DR_FC_BN[2].weight, 1)
        nn.init.constant_(self.BN_DR_FC_BN[2].bias, 0)
        nn.init.constant_(self.BN_DR_FC_BN[3].weight, 1)
        nn.init.constant_(self.BN_DR_FC_BN[3].bias, 0)

    def forward(self, x, labels=None):
        batch_size = x.shape[0]
        features = self.backbone.forward_features(x)
        features = self.adaptive_pooling(features)
        features = features.view(batch_size, -1)
        features = self.bn1(features)
        features = self.dropout(features)
        features = self.adaptive_pooling(features)
        features = features.view(features.size(0), -1)
        features = self.BN_DR_FC_BN(features)

        if labels is not None:
            arcfaceLogits = self.ArcMargin(features, labels)
            return arcfaceLogits
        return features