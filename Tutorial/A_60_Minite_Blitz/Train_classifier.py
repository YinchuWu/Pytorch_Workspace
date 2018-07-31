import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# == == == == =Visualization == == == == == == ==


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# #Random choose some samples
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))
# plt.show()
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# del dataiter
# == == == == == == == == == == == == == == == == == == =


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=0,
                               dilation=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0,
                               dilation=1, groups=1, bias=True)
        self.pool = nn.MaxPool2d(
            2, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=True)
        self.fc2 = nn.Linear(120, 84, bias=True)
        self.fc3 = nn.Linear(84, 10, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x), inplace=False))
        x = self.pool(F.relu(self.conv2(x), inplace=False))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x), inplace=False)
        x = F.relu(self.fc2(x), inplace=False)
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss(
    weight=None, size_average=True, ignore_index=-100, reduce=True)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9,
                      dampening=0, weight_decay=0, nesterov=False)
#==============Train==================

# for epoch in range(2):
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data

#         optimizer.zero_grad()

#         outputs = net(inputs)

#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step(closure=None)

#         running_loss += loss.item()
#         if i % 2000 == 0:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0
# print('finished')
#======================================


#==============Train_GPU==================

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

net = net.to(device)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step(closure=None)

        running_loss += loss.item()
        if i % 2000 == 0:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0
print('finished')
#======================================

#============Test_single============

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# imshow(torchvision.utils.make_grid(images, nrow=8, padding=2,
#                                    normalize=False, range=None, scale_each=False, pad_value=0))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# # plt.show()

# del dataiter

# out = net(images)

# aa, predicted = torch.max(out, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))
#===================================

#=============Test==================
# correct = 0
# total = 0

# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         output = net(images)
#         _, predicted = torch.max(output, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
#====================================

# ==========Test_per_class============
class_correct = list(0.0 for i in range(10))
class_total = list(0 for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        output = net(images)
        _, predicted = torch.max(output, 1)
        d = (predicted == labels)
        for i in range(4):
            label = labels[i]
            class_correct[label] += d[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' %
          (classes[i], 100 * class_correct[i] / class_total[i]))
# ====================================
