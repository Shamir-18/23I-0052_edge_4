"""students.py - the three small STUDENT networks: MobileNetV2, SqueezeNet, EfficientNet-Lite0.
Every student has the same interface:
    logits           = student(x)
    logits, features = student(x, return_feat=True)     # features = last feature map (used for feature-based KD)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torchvision

STUDENT_NAMES = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]


def _tv(builder, pretrained, label):
    try:
        return builder(weights="IMAGENET1K_V1" if pretrained else None)
    except Exception as e:
        print(f"[WARNING] could not download pretrained {label} ({e}) -> using RANDOM weights")
        return builder(weights=None)


class StudentNet(nn.Module):
    def __init__(self, name, num_classes=100, pretrained=True):
        super().__init__()
        self.name = name
        self.kind = "tv"
        if name == "mobilenetv2":
            m = _tv(torchvision.models.mobilenet_v2, pretrained, name)
            self.body = m.features                                         # -> 1280 channels
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Dropout(0.2), nn.Linear(1280, num_classes))
            self.feat_channels = 1280
        elif name == "squeezenet":
            m = _tv(torchvision.models.squeezenet1_1, pretrained, name)
            self.body = m.features                                         # -> 512 channels
            final = nn.Conv2d(512, num_classes, kernel_size=1)
            nn.init.normal_(final.weight, std=0.01)
            nn.init.zeros_(final.bias)
            self.head = nn.Sequential(nn.Dropout(0.2), final, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.feat_channels = 512
        elif name == "efficientnet_lite0":
            import timm
            self.kind = "timm"
            try:
                self.net = timm.create_model("efficientnet_lite0", pretrained=pretrained, num_classes=num_classes)
            except Exception as e:
                print(f"[WARNING] could not download pretrained efficientnet_lite0 ({e}) -> using RANDOM weights")
                self.net = timm.create_model("efficientnet_lite0", pretrained=False, num_classes=num_classes)
            self.feat_channels = self.net.num_features                     # 1280
        else:
            raise ValueError(f"unknown student {name}")

    def forward(self, x, return_feat=False):
        if self.kind == "timm":
            f = self.net.forward_features(x)
            logits = self.net.forward_head(f)
        else:
            f = self.body(x)
            logits = self.head(f)
        return (logits, f) if return_feat else logits
