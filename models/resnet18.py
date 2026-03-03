import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes=10, pretrained=True):
    model = models.resnet18(pretrained=pretrained) #pretrained=True：使用 ImageNet 预训练权重（迁移学习）
    # 修改最后一层全连接，适配 CIFAR-10 的 10 类
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model