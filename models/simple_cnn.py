import torch
import torch.nn as nn
import torch.nn.functional as F

#模型定义，继承nn.Module，包含特征提取部分和分类器部分
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):#模型最终输出10类
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(#定义特征提取部分，包含3个卷积层，每个卷积层后面跟着批归一化、ReLU激活和最大池化
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),#批归一化层，帮助加速训练和提高模型稳定性
            nn.ReLU(inplace=True),#ReLU激活函数，增加模型的非线性表达能力
            nn.MaxPool2d(2),#最大池化层，减少特征图的尺寸，提取主要特征，核2x2，步长默认2，将特征图宽高缩小为1/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 定义分类器部分，包含两个全连接层和一个输出层，每个全连接层后面跟着ReLU激活和Dropout
        self.classifier = nn.Sequential(#分类层，将特征转化为类别概率
            nn.Dropout(0.5),#Dropout层，随机丢弃神经元，防止过拟合
            #全连接层（线性层），将卷积层输出的特征图展平后输入，128个通道，每个通道4x4的特征图，输入维度为128*4*4=2048，输出维度为256
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),#ReLU激活函数，增加模型的非线性表达能力
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    #forward前向传播函数定义数据在模型中的流动路径,在初始化模型时被调用
    def forward(self, x):
        x = self.features(x)#特征提取层，得到128×4×4的特征图
        x = torch.flatten(x, 1)#展开张量，得到128×4×4的特征图
        # 注：flatten(x, 1) 表示从第1维开始展平（第0维是批量大小，保留）
        x = self.classifier(x)#展平后传入分类器，得到最终的类别概率输出
        return x#返回每个类的原始得分（未经过softmax处理）