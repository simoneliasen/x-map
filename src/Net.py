import torch
from torch import nn, relu
from torch import optim
from torchvision import models
import glob 
import os
import copy
from Methods.wandb import wandb_initialize
from Methods.parser import get_arguments

args = get_arguments()
print(args)

#i høj grad inspireret fra https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 

class Net():
    def __init__(self, model_name, pretrained = True, feature_extract = False, num_classes = 2, checkpoint = False): #features = tb positive/negative
        self.model_path = './resnet.pth'
        self.model_name = model_name
        self.data_dir = args.data_path #skal være opdelt i TB_positive og TB_negative
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_inception = False
        self.input_size = 224
        print(self.device)
        self.load_model(model_name, pretrained, num_classes)
        
        self.model.eval()
        self.model.to(self.device)

        #start træningen:
        if args.wandb:
            wandb_initialize(self)
        else:
            self.set_hyperparameters()
            from Methods.Train import KfoldTrain
            KfoldTrain(self)

    def set_baselineparameters(self):
        self.criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

        #baseline, vgg19, fra paper
        params = {
            'batch_size': 32,
            'lr': 0.001,
        }
        print('custom baseline hyperpameters!', params)

        self.batch_size = params['batch_size']
        self.optimizer = optim.SGD(self.model.parameters(), lr=params['lr'], momentum=0.9)
        args.scheduler = False
        self.scheduler = None

    def set_hyperparameters(self, params=None):
        self.criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

        if params is None or args.custom_config: #altså ingen wandb
            #vores bedste sweep, vgg19, crimson sweep 1:
            custom_params = {
                'batch_size': 32,
                'dropout_rate': 0.05233710200284458,
                'exponential_scheduler': 0.028372597647743948,
                'lr': 0.002987017391774973,
                'optimizer': 'sgd',
                'weight_decay': 0.00009916691342646931,
            }

            if args.batch_size is not None:
                custom_params['batch_size'] = args.batch_size
            
            params = custom_params
            print('custom bedste sweep hyperpameters!', params)

        #Og load params:
        self.batch_size = params['batch_size']

        if 'dropout_rate' in params:
            self.set_dropout(params['dropout_rate'])

        if params['optimizer'] == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=params['weight_decay'])
        elif params['optimizer'] == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=params['weight_decay'])
        
        if args.scheduler:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=params['exponential_scheduler'])

        

    def set_dropout(self, dropout_rate):
        #densenet og resnext har ikke dropout. Bør vi selv implementere det?
        if self.model_name == "vgg":
            self.model.classifier[2] = nn.Dropout(p=dropout_rate, inplace=False)
            self.model.classifier[5] = nn.Dropout(p=dropout_rate, inplace=False)
        
        elif self.model_name == "inception":
            self.model.dropout = nn.Dropout(p=dropout_rate, inplace=False)

        elif self.model_name == "efficientnet":
            self.model.classifier[0] = nn.Dropout(p=dropout_rate, inplace=True)

    def load_model(self, model_name, pretrained, num_classes):
        v = args.model_version

        if model_name == "densenet":
            if v == 0:
                self.model = models.densenet121(pretrained=pretrained)
            elif v == 1:
                self.model = models.densenet161(pretrained=pretrained)
            elif v == 2:
                self.model = models.densenet201(pretrained=pretrained)
            
            num_features = self.model.classifier.in_features 
            self.model.classifier = nn.Linear(num_features,num_classes)

        elif model_name == "resnext":
            if v == 0:
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
            elif v == 1:
                self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
            
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features,num_classes)

        elif model_name == "vgg":
            if v == 0:
                self.model = models.vgg11_bn(pretrained=pretrained)
            elif v == 1:
                self.model = models.vgg16_bn(pretrained=pretrained)
            elif v == 2:
                #self.model = models.vgg19(pretrained=pretrained)
                self.model = models.vgg19_bn(pretrained=pretrained)
            
            print(self.model.classifier)
            self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

        elif model_name == "inception":
            self.is_inception = True

            if v == 0:
                self.model = models.inception_v3(pretrained=pretrained)
            
            self.model.AuxLogits.fc = nn.Linear(768, num_classes)
            self.model.fc = nn.Linear(2048, num_classes)
            self.input_size = 299

        elif model_name == "efficientnet":
            if v == 0:
                self.model = models.efficientnet_b0(pretrained=pretrained)
                print(self.model.classifier)
                self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
            elif v == 1:
                self.model = models.efficientnet_b4(pretrained=pretrained)
                print(self.model.classifier)
                self.model.classifier[1] = nn.Linear(in_features=1792, out_features=num_classes, bias=True)
            elif v == 2:
                self.model = models.efficientnet_b7(pretrained=pretrained)
                print(self.model.classifier)
                self.model.classifier[1] = nn.Linear(in_features=2560, out_features=num_classes, bias=True)
            

            
            

model_names = ["densenet", "resnext", "vgg", "inception", "efficientnet"]

if args.model == None:
    net = Net("resnext") #skriv navnet på den du vil bruge.
else:
    net = Net(args.model)

#husk grayscale ting!!
#og husk det med at se på hvor sikker man er i sin prediction.