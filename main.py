#Defining convolutional neural network
import torch.nn as nn
#For defining loss function and optimizer
import torch.optim as optim
#import wandb
from matplotlib.path import Path
from matplotlib.patches import PathPatch
#Imports from other local files
from model import Net
from test import testing
from train import training
from load_data import load_data

#Log in to wandb here
#wandb.login()

#wandb.init(
#    project="my-test-project", 
#    entity="simoneliasen", 
#    config={
#        "learning_rate": 0.001,
#        "momentum": 0.9,
#        "epochs": 3
#        }
#    )

#config = wandb.config
# Pass config.epochs to training, also wandb.log(loss)


#load data
transform, batch_size, trainset, trainloader, testset, testloader, classes = load_data()

#Convolutional Neural network + Optimizer and Loss function
net = Net()
criterion = nn.CrossEntropyLoss() #Can this be defined within model? (check flatland)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Path to save and load model
PATH = './cifar_net.pth'

#Training
training(trainloader, optimizer, net, criterion, PATH)

#Testing
testing(testloader, classes, PATH)

#wandb.finish()


