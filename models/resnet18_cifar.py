import torch
import torch.nn as nn
import torchvision.models as models

def resnet18_cifar(num_classes=10):

    model = models.resnet18(weights=None)

    # 修改第一层卷积
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,#CIFAR-10图像较小，使用3x3卷积核，步长为1，保持特征图尺寸不变
        bias=False#去掉偏置项，因为后面有批归一化层会有偏置项，避免冗余
    )

    # 去掉maxpool
    model.maxpool = nn.Identity()

    # 修改分类层
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model