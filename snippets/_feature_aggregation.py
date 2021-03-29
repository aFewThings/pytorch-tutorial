import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class Conv(nn.Module):
    def __init__(self, in_chan, out_chan, k=3, s=1, bn=False, relu=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, k, s, 
                                padding=(k-1)//2, bias=True)
        self.bn = None
        self.relu = None
        
        if bn == True:
            self.bn = nn.BatchNorm2d(out_chan)
        if relu == True:
            self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv(3, 3, k=3, s=1, bn=True, relu=True)
        self.conv2 = Conv(3, 3, k=3, s=1, bn=True, relu=True)


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)

        i = random.randint(0, 2)
        out = out1[:, i, :, :] + out2[:, i, :, :]
        
        return out.unsqueeze(1)

net = Net()
inp = torch.randn(2, 3, 64, 64)

print(net.conv1.conv.weight.grad)
print(net.conv2.conv.weight.grad)

pred = net(inp)

target = torch.randn(2, 1, 64, 64)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

loss = criterion(pred, target)

loss.backward()

print(net.conv1.conv.weight.grad)
print(net.conv2.conv.weight.grad)
