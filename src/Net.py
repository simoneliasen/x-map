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
        self.data_dir = args.data_path #skal være opdelt i TB_positive og TB_negative
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_inception = False
        self.input_size = 224
        print(self.device)
        
        self.load_model(model_name, pretrained, num_classes)

        self.model.eval()
        self.set_parameter_requires_grad(self.model, feature_extract, self.last_layer_name)
        self.model.to(self.device)

        #start træningen:
        if args.wandb:
            wandb_initialize(self)
        else:
            self.set_hyperparameters()
            self.batch_size = 256
            print("batch size bør nu være 256")
            from Methods.Train import KfoldTrain
            KfoldTrain(self)

    def set_hyperparameters(self, parameters=None):
        self.criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

        if parameters is None: #altså ingen wandb
            self.batch_size = 4
            self.lr=0.001
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            print("parameters er:")
            print(parameters)
            self.batch_size = parameters['batch_size']
            self.exponential_scheduler = parameters['exponential_scheduler']
            self.lr = parameters['lr']
            self.weight_decay = parameters['weight_decay']
            self.dropout_rate = parameters['dropout_rate']

            if parameters['optimizer'] == "sgd":
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            elif parameters['optimizer'] == "rmsprop":
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
            

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
        elif model_name == model_names[2]: #squeezenet
            #virker nok ikke.
            self.model = models.squeezenet1_1(pretrained=pretrained)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            )
            self.last_layer_name = "classifier"
        elif model_name == model_names[3]: #chexnet
            self.model = models.densenet121(pretrained=pretrained)
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
            )
            self.last_layer_name = "classifier"
        elif model_name == model_names[4]: #vgg19
            #self.model = models.vgg19(pretrained=pretrained)
            self.model = models.vgg11_bn(pretrained=pretrained)
            self.model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
            self.last_layer_name = "classifier"
        elif model_name == model_names[5]: #densenet201
            self.model = models.densenet201(pretrained=pretrained)
            num_features = self.model.classifier.in_features 
            #densenet har 1024 in_features hvilket jo skal angives i classifieren - num_features.
            self.model.classifier = nn.Linear(num_features,num_classes)
            self.last_layer_name = "classifier"
        elif model_name == model_names[6]: #inception v3
            self.is_inception = True
            self.model = models.inception_v3(pretrained=pretrained)
            self.model.AuxLogits.fc = nn.Linear(768, num_classes)
            self.model.fc = nn.Linear(2048, num_classes)
            self.last_layer_name = "fc"
            self.input_size = 299
            
            

model_names = ["densenet", "resnet", "squeezenet", "chexnet", "vgg", "densenet201", "inception"]
net = Net(model_names[4])

#husk grayscale ting!!
#og husk det med at se på hvor sikker man er i sin prediction.