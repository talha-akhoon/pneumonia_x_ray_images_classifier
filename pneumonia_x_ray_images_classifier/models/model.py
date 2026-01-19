import torch
from torch import nn
from torchvision import models


class PneumoniaClassifierModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.0, freeze_backbone=True, unfreeze_last_n: int = 0):
        super(PneumoniaClassifierModel, self).__init__()

        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')

        for p in backbone.parameters():
            p.requires_grad = False

        if not freeze_backbone and unfreeze_last_n > 0:
            for block in backbone.features[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
        elif not freeze_backbone:
            for p in backbone.features.parameters():
                p.requires_grad = True

        self.features = backbone.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
