import torchvision as tv
import torchvision.transforms as transforms
import torch as t
from torchvision.transforms import ToPILImage
import torchvision
import numpy as np

dev = t.device('cuda:0')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
                                ])
trainset=tv.datasets.CIFAR10(root='/tmp/cifar10/train',
                             train=True,
                             download=True,
                             transform=transform)

trainloader=t.utils.data.DataLoader(trainset,
                                    batch_size=4,
                                    shuffle=True,
                                    num_workers=0)
testset=tv.datasets.CIFAR10(root='/tmp/cifar10/test',
                             train=False,
                             download=True,
                             transform=transform)

testloader=t.utils.data.DataLoader(testset,
                                   batch_size=4,
                                   shuffle=True,
                                   num_workers=0)


classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

(data,label)=trainset[100]
print(classes[label])

dataiter = iter(trainloader)
images, labels = dataiter.next()
images = images.to(dev)
labels = labels.to(dev)
print(images.size())


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net=Net().to(dev)
print(net)

from torch import optim
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)

from torch.autograd import Variable
import time

start_time = time.time()
for epoch in range(2):
    running_loss=0.0
    for i,data in enumerate(trainloader,0):
        inputs,labels=data
        inputs,labels=Variable(inputs).to(dev),Variable(labels).to(dev)
        optimizer.zero_grad()

        outputs=net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('finished training')
end_time = time.time()
print("Spend time:", end_time - start_time)

