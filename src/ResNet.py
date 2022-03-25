import torch
from torch import nn
from torch import optim
from torchvision import models
from Methods.Train import KfoldTrain

class ResNet():
    def __init__(self):
        self.model_path = './resnet.pth'
        self.data_dir = "../PP_data/" #skal være opdelt i TB_positive og TB_negative
        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model = models.resnet50(pretrained=False)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loadModelSettings()

    def loadModelSettings(self):
        #vi freezer de pretrained layers:
        #for param in self.model.parameters():
        #    param.requires_grad = False

        #og så redefine the final fully connnected layer (the one we want to train)
        self.model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))

        #og vælger bare loss function og optimizer
        self.criterion = nn.NLLLoss()
        #husk at skriv at vi ikke kun vil tærne på model.fc laget.
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)
        #scheduler her bliver kun brugt i den gamle train.
        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.1, patience=5)
        self.model.to(self.device) #load cuda hvis muligt

net = ResNet()
KfoldTrain(net)