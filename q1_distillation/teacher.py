"""teacher.py - the big TEACHER network: ResNet-50 (ImageNet-pretrained backbone, new 100-class head)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torchvision


class TeacherNet(nn.Module):
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        try:
            m = torchvision.models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        except Exception as e:
            print(f"[WARNING] could not download pretrained ResNet-50 ({e}) -> using RANDOM weights")
            m = torchvision.models.resnet50(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        self.feat_channels = 2048

    def forward(self, x, return_feat=False):
        f = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))   # last feature map (intermediate representation)
        logits = self.fc(self.pool(f).flatten(1))
        return (logits, f) if return_feat else logits
