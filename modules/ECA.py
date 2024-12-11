import torch
import torch.nn as nn

class ECALayer(nn.Module):
    def __init__(self, in_channels, k_size=3):
        """
        ECA模块，基于一维卷积的高效通道注意力机制。

        参数:
        in_channels (int): 输入特征图的通道数。
        k_size (int): 一维卷积的核大小，默认值为3。
        """
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 一维卷积层
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, x):
        """
        前向传播函数。

        参数:
        x (Tensor): 输入张量，形状为 (batch_size, channels, height, width)。

        返回:
        Tensor: 经过ECA模块调整后的输出张量。
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)  # 全局平均池化，并调整形状为 (batch_size, channels, 1)
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)  # 调整维度后通过一维卷积处理，然后恢复维度
        y = self.sigmoid(y).view(b, c, 1, 1)  # 通过Sigmoid激活，并调整形状为 (batch_size, channels, 1, 1)
        return x * y.expand_as(x)  # 对输入张量进行通道加权

# 测试 ECA 模块
if __name__ == "__main__":
    x = torch.randn(4, 64, 32, 32)  # 生成一个随机张量，模拟输入：batch size = 4, channels = 64, height = width = 32
    eca = ECALayer(in_channels=64, k_size=3)  # 创建一个 ECA 实例，通道数为64，卷积核大小为3
    y = eca(x)  # 通过 ECA 模块处理输入特征
    print(y.shape)  # 打印输出张量的形状，应该与输入相同：torch.Size([4, 64, 32, 32])
