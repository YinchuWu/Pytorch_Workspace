import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, arg):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1, padding=0,
                               dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0,
                               dilation=1, groups=1, bias=True)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, 10, bias=True)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # If the size is a square you can only specify a single numbe
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, stride=None, padding=0,
                         dilation=1, ceil_mode=False, return_indices=False)
        x = F.max_pool2d(F.relu(self.conv2(x), inplace=False), 2, stride=None,
                         padding=0, dilation=1, ceil_mode=False, return_indices=False)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x), inplace=False)
        x = F.relu(self.fc2(x), inplace=False)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

#==========Define_net=========
net = Net(nn.Module)
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())
#=============================

#==========Forward============
input = torch.randn(1, 32, 32)
input = input.unsqueeze(0)
output = net(input)
print(out)
# net.zero_grad()
# out.backward(torch.randn(1, 10))
target = torch.arange(1, 11)
target = target.view(1, -1)
criterion = nn.MSELoss(size_average=True, reduce=True)

loss = criterion(output, target)
#=============================

#==========Backward===========
net.zero_grad()
print('conv1.bias.grad before backward')
# print(net.conv1.weight.grad)
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#=============================
# print(net.conv1.weight.grad)


#=========hand_update=========
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    print(f)
#=============================


#=========auto_update=========
optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                      momentum=0, dampening=0, weight_decay=0, nesterov=False)
optimizer.zero_grad()
optimizer.step(closure=None)
#=============================
