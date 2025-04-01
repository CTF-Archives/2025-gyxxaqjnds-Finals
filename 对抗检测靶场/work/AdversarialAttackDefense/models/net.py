import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class ResNetModel(nn.Module):
    def __init__(self, architecture, num_classes=10):
        """
        定义 ResNet 模型。

        参数:
        - architecture: 使用的 ResNet 架构（如 'resnet18', 'resnet34', 'resnet50' 等）。
        - num_classes: 分类任务中的类别数量。
        """
        super(ResNetModel, self).__init__()
        # 根据用户选择的架构加载相应的 ResNet 模型
        if architecture == 'res18':
            self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif architecture == 'res34':
            self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif architecture == 'res50':
            self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # 获取最后一个全连接层的输入特征数
        in_features = self.resnet.fc.in_features

        # 替换最后的全连接层，以适应特定的类别数
        self.resnet.fc = nn.Linear(in_features, num_classes)
         # 添加全连接层之后的 BatchNorm1d 层
        self.bn_fc = nn.BatchNorm1d(num_classes)
    def forward(self, x):
        """
        定义前向传播过程。

        参数:
        - x: 输入张量，形状为 (batch_size, channels, height, width)

        返回:
        - 输出张量，形状为 (batch_size, num_classes)
        """
        x = self.resnet(x)
        x = self.bn_fc(x)  # 在全连接层后应用 BatchNorm
        return x