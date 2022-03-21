import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from os import listdir
from os.path import isfile, join
from ChinaDataset import ChinaDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    transform2 = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform = transforms.Compose([ #fra resnet.
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 4

    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
     #                                       download=True, transform=transform)
    #trainset = ChinaDataset(csv_file = 'ChinaData.csv', root_dir = '../ChinaSet_AllFiles/CXR_png', transform = transform)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
     #                                       shuffle=True, num_workers=2)

    #testset = torchvision.datasets.CIFAR10(root='./data', train=False,
      #                                  download=True, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
     #                                       shuffle=False, num_workers=2)



    #fra jakob
    dataset = ChinaDataset(csv_file = 'ChinaData2.csv', root_dir = '../ChinaSet_AllFiles/CXR_png', transform = transform) #fra transforms.ToTensor()

    trainset, testset = torch.utils.data.random_split(dataset, [561, 100])

    batch_size = 4
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)
    classes = ('normal', 'tubercolosis')

    

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    class Net(nn.Module):
        def __init__(self):
            self.PATH = './resting2.pth'
            super().__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 53 * 53, 120) #5 rettet til 53, pga. 224 - 4 (kernel size) = 220. / 2 (pool) = 110 - 4 (kernel) = 106 / 2 = 53
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 2) #fra 84, 2 (fordi vi kun har 2 labels.)
            #self.fc1 = nn.Linear(16 * 5 * 5, 120)
            #self.fc2 = nn.Linear(120, 84)
            #self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x2 = self.conv1(x)
            x22 = self.pool(F.relu(x2))
            x3 = self.conv2(x22)
            x33 = self.pool(F.relu(x3))
            x4 = torch.flatten(x33, 1) # flatten all dimensions except batch
            x5 = F.relu(self.fc1(x4))
            x6 = F.relu(self.fc2(x5))
            x7 = self.fc3(x6)
            return x7

        def train(self):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


            for epoch in range(2):  # loop over the dataset multiple times

                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:    # print every 2000 mini-batches
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                        running_loss = 0.0

            print('Finished Training')
            torch.save(net.state_dict(), self.PATH)

        def evaluatePerformance(self):
            self.evaluateEachClass()

            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        
        def evaluateEachClass(self):
            # prepare to count predictions for each class
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}

            # again no gradients needed
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    outputs = net(images)
                    _, predictions = torch.max(outputs, 1)
                    # collect the correct predictions for each class
                    for label, prediction in zip(labels, predictions):
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1


            # print accuracy for each class
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        def test(self):
            dataiter = iter(testloader)
            images, labels = dataiter.next()

            # print images
            print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)

            print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                        for j in range(4)))

            imshow(torchvision.utils.make_grid(images))


    net = Net()
    net.load_state_dict(torch.load(net.PATH))
    net.train()
    net.test()
    net.evaluatePerformance()

    