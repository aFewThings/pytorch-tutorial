import torch
import torch.nn as nn
import torch.optim as optim


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, hidden_c, out_c):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, hidden_c, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_c, hidden_c, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_c, out_c, kernel_size=1, stride=1, padding=0)

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(hidden_c * 2, out_c)
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        coords = self.argmax(out)
        att_raw = self.mlp(coords)
        score = torch.sigmoid(att_raw).unsqueeze(2).unsqueeze(3).expand_as(x) # BxCxHxW
        
        out = self.conv3(out) * score

        return out


    def argmax(self, inp):
        with torch.no_grad():
            inp_re = inp.reshape(inp.size(0), inp.size(1), -1) # 1, 5, 64*64
            xy = torch.argmax(inp_re, dim=-1) # 1, 5

            xi = (xy % inp.size(3)).float()
            yi = (xy // inp.size(3)).float()
            print(xi, yi)
            #print(xi.grad, yi.grad) # None, None

        return torch.cat([torch.unsqueeze(xi, 2), torch.unsqueeze(yi, 2)], dim=2) # BC2


model = Net(5, 3)
x = torch.randn(1, 3, 64, 64)
target = torch.zeros(1, 3, 64, 64) 

optimizer = optim.SGD(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

output = model(x)

optimizer.zero_grad()
loss = criterion(output, target)
loss.backward()
optimizer.step()
