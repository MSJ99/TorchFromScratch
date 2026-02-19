import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        L1 = self.relu1(self.bn1(self.conv1(x)))
        L2 = self.bn2(self.conv2(L1))
        output = self.relu2(x + L2)

        return output
    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 7, 2)
        
        self.conv2_1 = nn.MaxPool2d(3, 2)
        self.conv2_2 = Block(64, 64)
        self.conv2_3 = Block(64, 64)
        self.conv2_4 = Block(64, 64)
        
        self.conv3_1 = Block(64, 128, stride=2)
        self.conv3_2 = Block(128, 128)
        self.conv3_3 = Block(128, 128)
        self.conv3_4 = Block(128, 128)
        
        self.conv4
        self.conv5