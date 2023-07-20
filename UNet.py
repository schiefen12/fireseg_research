import torch
import torch.nn as nn
from torchvision.models import resnet50

#UNet is simple encoder/decoder With two 3x3 convolutions with ReLU activation after each one for each layer
#Each layer is followed by a max pooling operation
#Skip-connections are also used in the decoder to bring in learned features from the corresponding encoder layer

#This model is an implementation of the model from the "U-Net: Convolutional Networks for Biomedical" paper.

Image Segmentation

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(UNet, self).__init__()

        self.contracting1 = DoubleConv(in_channels, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting2 = DoubleConv(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting3 = DoubleConv(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting4 = DoubleConv(256, 512)

        self.expanding1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.expanding2 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.expanding3 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.expanding4 = nn.ConvTranspose2d(128, num_classes, kernel_size=1)

        #self.se1 = SELayer(64)
        #self.se2 = SELayer(128)
        #self.se3 = SELayer(256)
        #self.se4 = SELayer(512)

    def forward(self, x):
        x1 = self.contracting1(x)
        #x1 = self.se1(x1)
        x2 = self.contracting2(self.maxpool1(x1))
        #x2 = self.se2(x2)
        x3 = self.contracting3(self.maxpool2(x2))
        #x3 = self.se3(x3)
        x4 = self.contracting4(self.maxpool3(x3))
        #x4 = self.se4(x4)

        x = self.expanding1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.expanding2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.expanding3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.expanding4(x)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        return x

"""
class SELayer(nn.Module):
    def __init__(self, channel, reduction=32):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
"""

