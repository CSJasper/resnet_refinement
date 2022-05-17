import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# con1x1 (shrink the dimension)
def conv1x1(in_channels, out_channels, stride=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


# cov3x3
def conv3x3(in_channels, out_channels, stride=1, dilation=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=1, bias=False, dilation=dilation)

# custom resnet34 module (final layer is global average pooling)

# Conv blocks

class ConvBlock(nn.Module):
        expansion = 1
        def __init__(self, in_channels, out_channels, stride=1, downsample=None, base_width=64):
                super(ConvBlock, self).__init__()
                self.conv1 = conv3x3(in_channels, out_channels, stride)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(out_channels, out_channels)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.downsample = downsample
                self.stride = stride
        
        def forward(self, x):
                identity = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                
                if self.downsample is not None:
                        identity = self.downsample(x)

                out += identity
                out = self.relu(out)
                
                return out


class Bottleneck(nn.Module):
        expansion = 4
        def __init__(self, in_channels, out_channels, stride=1, downsample=None, base_width=64):
                super(Bottleneck, self).__init__()
                width = int(in_channels * (base_width / 64.0))
                
                self.conv1 = conv1x1(in_channels, width)
                self.bn1 = nn.BatchNorm2d(width)
                self.conv2 = conv3x3(width, width, stride)
                self.bn2 = nn.BatchNorm2d(width)
                self.conv3 = conv1x1(width, out_channels * 4)
                self.bn3 = nn.BatchNorm2d(out_channels * 4)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride
                
        def forward(self, x):
                identity = x
                
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)
                
                out = self.conv3(out)
                out = self.bn3(out)
                
                if self.downsample is not None:
                        identity = self.downsample(x)
                
                out += identity
                out = self.relu(out)
                
                return out


class Resnet(nn.Module):
        def __init__(self, block, layers, num_classes=47, width=64):
                super(Resnet, self).__init__()
                self.in_channels = 64
                self.width = 64
                
                self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)  # why padding is 3
                self.bn1 = nn.BatchNorm2d(self.in_channels)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, num_classes, layers[2], stride=2)  # actually 256 in pure ResNet
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # this layer will be discarded
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512 * block.expansion, num_classes)  # should be modified as Global Average pooling (will be discared)
                
                
        
        def _make_layer(self, block,  out_channels, blocks, stride=1):
                downsample = None
                
                if stride != 1 or self.in_channels != out_channels * block.expansion:
                        downsample = nn.Sequential(
                                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                                nn.BatchNorm2d(out_channels * block.expansion)
                        )
                
                layers = []
                layers.append(block(self.in_channels, out_channels, stride, downsample, self.width))
                self.in_channels = out_channels * block.expansion
                
                for _ in range(1, blocks):
                        layers.append(block(self.in_channels, out_channels, base_width=self.width))
                        
                return nn.Sequential(*layers)
        
        # make global average pool
        def GlobalAvgPool(self, x):
                x = self.avgpool(x)
                x = torch.mean(x, dim=2)
                return x
        
        def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                #x = self.layer4(x)
                x = self.GlobalAvgPool(x)
                x = x.view((x.shape[0], x.shape[1]))
                
                return x


def _resnet(arch, block, layers, progress, **kwargs):
        model = Resnet(block, layers, **kwargs)
        return model

def custom_resnet34(progress=True, **kwargs):
        return _resnet('resnet34_custom', ConvBlock, [3, 4, 6, 3], progress, **kwargs)

def custom_resnet18(progress=True, **kwargs):
        return _resnet('resnet18_custom', ConvBlock, [2, 2, 2, 2], progress, **kwargs)

def custom_resnet50(progress=True, **kwargs):
        return _resnet('resnet50_custom', Bottleneck, [3, 4, 6, 3], progress, **kwargs)

def custom_resnet152(progress=True, **kwargs):
        return _resnet('resnet152_custom', Bottleneck, [3, 8, 36, 3], progress, **kwargs)


class simple_block(nn.Module):
        def __init__(self, num_classes=47, transform=None):
                super(simple_block, self).__init__()
                self.num_classes = num_classes
                self.transform = transform
                
                self.conv = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2)
                )
                
                self.fc1 = nn.Sequential(
                        nn.Linear(32 *13 * 13, 512),
                        nn.ReLU()
                )
                
                self.fc2 = nn.Sequential(
                        nn.Linear(512, num_classes),
                )
                
                
        
        def forward(self, x):
                if self.transform is not None:
                        x = self.transform(x)
                x = self.conv(x)
                x = torch.flatten(x)
                x = self.fc1(x)
                x = self.fc2(x)
                
                return x


def simple(num_classes=47, transform=None):
        return simple_block(num_classes, transform)


'''class simple_lowdim(nn.Module):
        def __init__(self, num_classes=47, transform=None):
                super(simple_lowdim, self).__init__()
                self.num_classes = num_classes
                self.transform = transform
                '''