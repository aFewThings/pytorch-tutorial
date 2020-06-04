import torch
import copy

class DynamicNet(torch.nn.Module):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.lin1 = torch.nn.Linear(1, 1, bias=False)
        self.lin2 = copy.deepcopy(self.lin1)

    def forward(self, x):
        out = self.lin1(x)
        out = self.lin2(out) # w2 * w1x

        return out

net = DynamicNet()

x = torch.ones((1, 1)) # 1
print('x:', x)

for name, param in net.named_parameters():
    print(name, param)

y_hat = net(x)
print('y_hat:', y_hat)

for para in net.parameters():
    print(para.grad) # None

y_hat.backward()

for para in net.parameters():
    print(para.grad) # w1x + w2x
