# Define NN
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel(grayscale), 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight
print(net.conv1.weight.size())

input = torch.randn(1, 1, 32, 32) # BCHW
out = net(input)
print(net.forward(input))
print(out)

net.zero_grad() # 모든 매개변수의 변화도 버퍼(gradient buffer)를 0으로 설정, 무작위 값으로 역전파 합니다.
out.backward(torch.randn(1, 10))

# Loss Function
output = net.forward(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

"""
.grad_fn 속성을 사용하여 loss를 역방향에서 따라가다보면, 이러한 모습의 연산 그래프를 볼 수 있습니다.
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
"""
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad() # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# gradient update
# 가중치(wiehgt) = 가중치(weight) - 학습율(learning rate) * 변화도(gradient)

# 간단한 업데이트 과정
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# optimizer 사용해서 업데이트하기
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad() # 수동으로 변화도 버퍼를 0으로 설정해주지 않으면 변화도가 계속 누적된다.
output = net.forward(input)
loss = criterion(output, target)
loss.backward() # compute loss
optimizer.step() # update trainable variables
