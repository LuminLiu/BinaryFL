#This net use tanh as the activation function
import torch.nn as nn
import torchvision.transforms as transforms
import math
from .binarized_modules import BinarizeLinear, BinarizeConv2d

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = BinarizeConv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.tanh1 = nn.Tanh()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = BinarizeConv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.tanh2 = nn.Tanh()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = BinarizeLinear(7*7*16, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.tanh3 = nn.Tanh()
        self.fc2 = BinarizeLinear(100, 10)
        self.bn4 = nn.BatchNorm1d(10)
        self.logsoftmax=nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.maxpool2(x)
        x = x.view(-1,7*7*16)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.tanh3(x)
        x = self.fc2(x)
        x = self.bn4(x)
        return self.logsoftmax(x)


def mnistnet_binary(**model_config):
    return Net()
