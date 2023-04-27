import torch
import torch.nn as nn
from torch.autograd import Function

def Binarize(tensor,quant_mode = 'det'):
    if quant_mode == 'det':
        return tensor.sign()
    else:
        return tensor.add_(1).div(2).add(tensor.rand(tensor.size(), device=device)).add(-0.5).clamp_(0,1).round().mul_(2).add(-1)

class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input) :
        if input.size(1) != 784: #
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = self.weight.org.sign()
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = self.bias.org.sign()
            out += self.bias.view(1 , -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):
    def __init__(self,*kargs, **kwargs):
        super(BinarizeConv2d,self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3: #3 for CIFAR10, 1 for mnist #dont binarize the input
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=self.weight.org.sign()
        # self.weight.data=Ternarize(self.weight.org)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            self.bias.data=self.bias.org.sign()
            # self.bias.data=Ternarize(self.bias.org)
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out