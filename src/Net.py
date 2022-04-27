import torch
from torch import nn, relu
from torch import optim
from torchvision import models
from Methods.wandb import wandb_initialize
from Methods.parser import get_arguments

args = get_arguments()
print(args)

#i høj grad inspireret fra https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 

class Net():
    def __init__(self, model_name, pretrained = True, feature_extract = False, num_classes = 2): #features = tb positive/negative
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

    def set_hyperparameters(self, params=None):
        self.criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

        if params is None: #altså ingen wandb
            self.batch_size = 4
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        else:
            self.batch_size = params['batch_size']
            self.exponential_scheduler = params['exponential_scheduler']
            self.set_dropout(params['dropout_rate'])

            if params['optimizer'] == "sgd":
                self.optimizer = optim.SGD(self.model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=params['weight_decay'])
            elif params['optimizer'] == "rmsprop":
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=params['lr'], momentum=0.9, weight_decay=params['weight_decay'])

        

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
        if model_name == "densenet":
            self.model = models.densenet201(pretrained=pretrained)
            num_features = self.model.classifier.in_features 
            self.model.classifier = nn.Linear(num_features,num_classes)

        elif model_name == "resnext":
            self.model = models.resnext101_32x8d(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features,num_classes)

        elif model_name == "vgg":
            self.model = models.vgg19_bn(pretrained=pretrained)
            self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)

        elif model_name == "inception":
            self.is_inception = True
            self.model = models.inception_v3(pretrained=pretrained)
            self.model.AuxLogits.fc = nn.Linear(768, num_classes)
            self.model.fc = nn.Linear(2048, num_classes)
            self.input_size = 299
            
        elif model_name == "efficientnet":
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

            
            

model_names = ["densenet", "resnext", "vgg", "inception", "efficientnet"]

if args.model == None:
    net = Net("densenet") #skriv navnet på den du vil bruge.
else:
    net = Net(args.model)



#husk grayscale ting!!
#og husk det med at se på hvor sikker man er i sin prediction.