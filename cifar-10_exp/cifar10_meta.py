# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms

import logging
logging.basicConfig(filename='cifar-10-meta-log',
                    level=logging.DEBUG,
                    filemode='a',
                    format='%(levelname)s: %(message)s')
                    #format='\n%(asctime)s: \n -- %(pathname)s \n --[line:%(lineno)d]\n -- %(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)
import datetime
logger.info("############# experiment {} ################".format(datetime.datetime.now()))
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False,drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
#dataiter = iter(trainloader)
#images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# logger.info labels
#logger.info(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F
import resnet

resnet18=resnet.resnet18(pretrained=False,num_classes=10)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
### CNN
#net = torch.nn.DataParallel(Net()).cuda()
#### ResNet50
net = torch.nn.DataParallel(resnet18).cuda()
net=net.train()
logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1000000.0))


import torch.optim as optim

criterion = nn.CrossEntropyLoss(reduce=False)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

import magic
import time
from tensorboardX import SummaryWriter

log_path='log/metalog'
writer_dict = {
        'writer': SummaryWriter(log_dir=log_path),
        'train_global_steps': 0,
        'test_global_steps': 0,
    }

def test_accuracy(net,classes,writer_dict):
    net=net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()

    logger.info('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    writer = writer_dict['writer']
    global_steps = writer_dict['test_global_steps']
    writer.add_scalar('meta_test_accuarcy', 100 * correct / total, global_steps)
    
    ########################################################################
    # That looks waaay better than chance, which is 10% accuracy (randomly picking
    # a class out of 10 classes).
    # Seems like the network learnt something.
    #
    # Hmmm, what are the classes that performed well, and the classes that did
    # not perform well:

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels.cuda()).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        logger.info('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        writer.add_scalar('class_{}_accuarcy'.format(classes[i]), 100 * class_correct[i] / class_total[i], global_steps)
    
    writer_dict['test_global_steps'] = global_steps + 1

batchsize=16
total_epoch=70
M_C=magic.MetaData_Container(len(trainset),batchsize,total_epoch)
trainset_meta=M_C.Add_Meta_To_Trainset(trainset)

#引入随机标签
rand=0.3
trainset_meta_randY=M_C.Random_Label_To_Trainset(trainset_meta,rand)
logger.info('==> Label Y is changed randomly by {} possibility'.format(rand))
trainloader = torch.utils.data.DataLoader(trainset_meta_randY, batch_size=batchsize,
                                          shuffle=True,drop_last=True)
logger.info('==> batchsize = {}'.format(batchsize))
logger.info('==> total_epoch = {}'.format(total_epoch))
logger.info('==> dataset size = {}'.format(len(trainset)))
for epoch in range(total_epoch):  # loop over the dataset multiple times
 
    begin = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        
        inputs, labels, meta = data
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs.cuda(), labels.cuda())
        #loss=loss.mean(dim=0)
        #logger.info(loss.mean(dim=0))
        # make sure the loss is divided into each example in minibatch
        meta_loss=M_C.API_LOSS(loss,meta,epoch)
        #loss.backward()
        meta_loss.backward()
        #logger.info(meta_loss)
        optimizer.step()
        
        
        # logger.info statistics
        running_loss += meta_loss.item()
        if i % 1000 == 999:    # logger.info every 2000 mini-batches
            end=time.time()
            logger.info('[%d, %5d] loss: %.3f Time consuming: %.3f'  %
                  (epoch + 1, i + 1, running_loss / 1000,(end-begin)/1000))
            
            begin=time.time()
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('meta_train_loss', running_loss / 1000, global_steps*1000)
            writer_dict['train_global_steps'] = global_steps + 1
            running_loss = 0.0
    
    test_accuracy(net,classes,writer_dict)

logger.info('Finished Training')


writer_dict['writer'].close()