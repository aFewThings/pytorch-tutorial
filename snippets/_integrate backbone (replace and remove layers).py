'''
1. freeze pretrained model
2. replace and remove some layers in pretrained model
3. include (modified) pretrained model to other model as a backbone and train integrated model
'''

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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
        

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = Conv(3, 3, k=3, s=1, bn=True, relu=True)
        self.conv2 = Conv(3, 3, k=3, s=1, bn=True, relu=True)
        self.layers = nn.Sequential(
            Conv(3, 1, k=1),
            Flatten() # replace this
        )
        self.final_layer = nn.Linear(64*64, 2) # remove this

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.layers(out)
        out = self.final_layer(out)

        return out


class Net(nn.Module):
    def __init__(self, backbone):
        super(Net, self).__init__()
        self.backbone = self._modify_backbone(backbone, freeze_layers=True)
        self.new_final_layer = nn.Linear(3*64*64, 10)

    def _modify_backbone(self, backbone, freeze_layers=False):
        
        if freeze_layers:
            for name, param in backbone.named_parameters():
                param.requires_grad = False

        # removing a specific layer can be implemented by replacing it with an identity layer
        backbone._modules['final_layer'] = nn.Identity()
        # it also can be deleted by __delattr__ but, it would raise an error as the forward function uses self.final_layer
        # backbone.__delattr__('final_layer')
        
        # Module.children() (none recursive) vs Module.modules() (recursive)
        # [reference](https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/4)
        for name, module in backbone.named_children():
            if isinstance(module, nn.Sequential):
                for n, m in module.named_modules():
                    if isinstance(m, Flatten):
                        module.__delattr__(n)
                        module.add_module(n, Conv(1, 3, k=1))
                        module.add_module(str(int(n)+1), Flatten())

        print(backbone)
        return backbone

    def forward(self, x):
        out = self.backbone(x)
        out = self.new_final_layer(out)

        return out


inp = torch.randn(2, 3, 64, 64)

backbone = Backbone()
pred1 = backbone(inp)
target1 = torch.randn(2, 2)

optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)
criterion = nn.MSELoss()
loss = criterion(pred1, target1)

# train backbone
optimizer.zero_grad()
print('backbone.conv1.conv.weight.data', backbone.conv1.conv.weight.data)
loss.backward()
optimizer.step()
print('backbone.conv1.conv.weight.data', backbone.conv1.conv.weight.data)

new_model = Net(backbone)
pred2 = new_model(inp)
target2 = torch.randn(2, 10)

optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
loss = criterion(pred2, target2)

# train new model
optimizer.zero_grad()
print('backbone.conv1.conv.weight.data', backbone.conv1.conv.weight.data) # frozen layer to be fixed
print('new_model.new_final_layer.weight.data', new_model.new_final_layer.weight.data) # target layer to be trained
loss.backward()
optimizer.step()
print('backbone.conv1.conv.weight.data', backbone.conv1.conv.weight.data)
print('new_model.new_final_layer.weight.data', new_model.new_final_layer.weight.data)