"""
pytorch의 model summary는 __init__에서 정의된 속성들을 출력할 수 있다.
출력되는 속성들은 반드시 nn.Module을 상속받은 클래스이어야 한다. 
또한, __init__에서 정의되지 않고 (혹은 등록되지 않은) 곧바로 forward()에서 사용되는 클래스들은 출력에 나타나지 않는다.
이러한 클래스들의 파라미터들은 학습시킬 수 없기 때문에, 학습시키고자 하는 레이어들은 꼭 __init__에서 정의해주어야 한다.

+ pytorch의 model summary는 단순히 __init__에서 정의된 nn.Module들을 출력하므로 forward() 과정의 순차적인 layers를 출력할 수 없다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        

class Temp(nn.Module):
    def __init__(self):
        super(Temp, self).__init__()
        self.conv1 = Conv(3, 3, k=3, s=1, bn=True, relu=True)
        self.conv2 = Conv(3, 3, k=3, s=1, bn=True, relu=True)
        self.w = torch.randn(3, 3, 1, 1)


    def forward(self, x):
        out = self.conv1(x)
        out = F.conv2d(out, self.w)

        return out

tmp = Temp()
print(tmp)