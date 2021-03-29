import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1, 3, bias=False)
        self.fc2 = nn.Linear(3, 3, bias=False)
        self.fc3 = nn.Linear(3, 1, bias=False)

    def forward(self, x):
        out = []
        out.append(self.fc1(x))
        out.append(self.fc2(out[0]))
        out.append(self.fc3(out[1]))
        
        return out


model = Net()
x = torch.ones((1, 1))
print('x', x)

target1 = torch.ones((1, 1)) * 5
print('target', target1)

target2 = torch.ones((1, 3)) * 2

optimizer = optim.SGD(model.parameters(), lr=0.1)

criterion = nn.MSELoss()

for data in model.parameters():
    print('weights grad', data.grad) #None


optimizer.zero_grad()

output = model(x)
loss1 = criterion(output[2], target1)
print('loss', loss1)

loss1.backward(retain_graph=True)

i = 0
for data in model.parameters():
    i += 1
    print(str(i) + '_weights grad', data.grad)
    

#optimizer.zero_grad()

loss2 = criterion(output[0], target2)
loss2.backward()
optimizer.step() # loss1과 loss2는 .grad에 누적된다. 

i = 0
for data in model.parameters():
    i += 1
    print(str(i) + '_weights grad', data.grad)

# loss = loss1 + loss2
# loss.backward()도 동일하게 작동함.

### 단, loss1을 backward한 뒤, optimizer.step()과 같이 가중치를 업데이트하게 되면, loss2를 구한 시점이 가중치 업데이트의 전인지, 후인지에 따라 결과가 달라짐.
### 일반적으로 loss1 -> update -> loss2 -> update 보다는, loss1 + loss2 -> update가 훈련 과정이나 중간 감독이라는 의도에 적합할 것임.