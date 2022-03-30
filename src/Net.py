import torch
from torch import nn
from torch import optim
from torchvision import models
from Methods.Train import KfoldTrain

#i høj grad inspireret fra https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 

class Net():
    def __init__(self, model_name, pretrained = True, feature_extract = False, num_classes = 2): #features = tb positive/negative
        self.model_path = './resnet.pth'
        self.data_dir = "../PP_data/" #skal være opdelt i TB_positive og TB_negative
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        
        self.load_model(model_name, pretrained, num_classes)

        self.model.eval()
        self.set_parameter_requires_grad(self.model, feature_extract, self.last_layer_name)

        #self.criterion = nn.NLLLoss() #virkede ikke med densenet. Spørg mig ikke hvorfor.
        self.criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
        self.set_optimizer(optim.Adam,feature_extract,self.last_layer_name, lr=0.003)
        #self.print_parems_to_update(feature_extract)
        self.model.to(self.device)

    def set_parameter_requires_grad(self, model, feature_extracting, last_layer_name):
        #altså hvis vi kun vil træne det sidste layer (classifier) så freezer vi alle andre layers.
        #dette kaldes også for feature_extracting.
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

            #Og så vælger vi bare at vi kun vil have grad på sidste layer:
            last_layer = model.fc if last_layer_name == "fc" else model.classifier
            for parem in last_layer.parameters():
                parem.requires_grad = True
    
    def set_optimizer(self, optimizer, feature_extract, last_layer_name, lr):
        #optimizer = en metode, dvs. higher order function. Kurt havde været stolt.
        target = self.model
        if feature_extract:
            target = self.model.fc if last_layer_name == "fc" else self.model.classifier

        if optimizer == optim.Adam:
            self.optimizer = optimizer(target.parameters(), lr=lr)
        else: 
            raise NotImplementedError("Den ønskede optimizer skal lige implementeres dynamisk først <3")

    def print_parems_to_update(self, feature_extract):
        params_to_update = self.model.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

    def load_model(self, model_name, pretrained, num_classes):
        if model_name == model_names[0]: #densenet
            self.model = models.densenet121(pretrained=pretrained)
            num_features = self.model.classifier.in_features 
            #densenet har 1024 in_features hvilket jo skal angives i classifieren - num_features.
            self.model.classifier = nn.Linear(num_features,num_classes)
            self.last_layer_name = "classifier"

        elif model_name == model_names[1]: #resnet
            self.model = models.resnet50(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features,num_classes)
            self.last_layer_name = "fc"
            

model_names = ["densenet", "resnet"]
net = Net(model_names[0])
KfoldTrain(net)