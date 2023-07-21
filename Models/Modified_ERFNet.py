# Modified Version of ERFNet
# July 2023
# Samantha Schiefen
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#Current modifications to this architecture include halving the number of non_bottleneck_1d layers throughout the model
#and replacing the 1x3 convolutions in the bottleneck with 1x1 convolutions.

class DownsamplerBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels-in_channels, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.depthwise_conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), groups=chann, bias=True)
        self.pointwise_conv1x1_1 = nn.Conv2d(chann, chann, 1, stride=1, padding=0, bias=True) #Instead of doing the 1x3 convolution, we replaced it with a 1x1 convolution to improve segmentation speed and reduce parameters

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.depthwise_conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), groups=chann, dilation=(dilated, 1), bias=True)
        self.pointwise_conv1x1_2 = nn.Conv2d(chann, chann, 1, stride=1, padding=0, bias=True) #Instead of doing the 1x3 convolution, we replaced it with a 1x1 convolution to improve segmentation speed and reduce parameters

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.depthwise_conv3x1_1(input)
        output = F.relu(output)
        output = self.pointwise_conv1x1_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.depthwise_conv3x1_2(output)
        output = F.relu(output)
        output = self.pointwise_conv1x1_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList() #Initialize and empty list of modules

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 2):    #Instead of adding 5 layers, we added 2
            self.layers.append(non_bottleneck_1d(64, 0.1, 1))  

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 1):    #Instead of adding this section twice, we only added it once
            self.layers.append(non_bottleneck_1d(128, 0.1, 2))
            self.layers.append(non_bottleneck_1d(128, 0.1, 4))
            self.layers.append(non_bottleneck_1d(128, 0.1, 8))
            self.layers.append(non_bottleneck_1d(128, 0.1, 16))
            
        #only for encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)
        
        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1)) #Original architecture had two of these layers, but we only used 1

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1)) #Original architecture had two of these layers, but we only used 1

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class ERFNet2(nn.Module):
    def __init__(self, num_classes, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder

        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input) 
            return self.decoder.forward(output)
