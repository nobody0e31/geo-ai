from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torchvision import models


class AttentionGate(nn.Module):
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = self.relu(theta_x + phi_g)
        psi = self.sigmoid(self.psi(f))
        upsampled = torch.nn.functional.interpolate(
            psi, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return x * upsampled


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResUNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.att3 = AttentionGate(1024, 2048, 512)
        self.att2 = AttentionGate(512, 1024, 256)
        self.att1 = AttentionGate(256, 512, 128)

        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(self.pool(enc0))
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        gate3 = self.att3(enc3, enc4)
        gate2 = self.att2(enc2, enc3)
        gate1 = self.att1(enc1, enc2)

        dec4 = self.decoder4(enc4, gate3)
        dec3 = self.decoder3(dec4, gate2)
        dec2 = self.decoder2(dec3, gate1)
        dec1 = self.decoder1(dec2, enc0)
        return self.final(dec1)


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        backbone = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class ModelBundle:
    segmentation: nn.Module
    classification: nn.Module


def build_models(num_seg_classes: int = 4, num_cls_classes: int = 6) -> ModelBundle:
    return ModelBundle(
        segmentation=ResUNet(num_classes=num_seg_classes),
        classification=EfficientNetClassifier(num_classes=num_cls_classes),
    )


def build_mask_rcnn(num_classes: int = 4) -> nn.Module:
    model = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, targets)


def classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, targets)


def metrics_from_logits(logits: torch.Tensor) -> Dict[str, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)
    conf, preds = probs.max(dim=1)
    return {"confidence": conf.mean(), "predictions": preds}
