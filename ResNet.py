import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from load_data import label_files, load_and_transform_data


class ResNet():
    def __init__(self):
        self.model_path = './resnet.pth'
        self.data_dir = "../PP_data/" # "../ChinaSet_AllFiles/CXR_png/"
        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model = models.resnet50(pretrained=True) #altså den rigtige
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #1 kanal da det billedet er grayscale.
        self.model.eval() #?
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loadModelSettings()

    def load_split_train_test(self, datadir, valid_size = .2):
        train_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        ])
        test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        ])
        train_data = datasets.ImageFolder(datadir,       
                        transform=train_transforms)
        test_data = datasets.ImageFolder(datadir,
                        transform=test_transforms)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        train_idx, test_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        trainloader = torch.utils.data.DataLoader(train_data,
                    sampler=train_sampler, batch_size=64)
        testloader = torch.utils.data.DataLoader(test_data,
                    sampler=test_sampler, batch_size=64)
        return trainloader, testloader

    def loadModelSettings(self):
        #self.model.load_state_dict(torch.load(self.model_path)) #load den trænede model.
        self.trainloader, self.testloader = self.load_split_train_test(self.data_dir, .2)
        print(self.trainloader.dataset.classes)

        #vi freezer de pretrained layers:
        for param in self.model.parameters():
            param.requires_grad = False

        #og så redefine the final fully connnected layer (the one we want to train)
        self.model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 10),
                                 nn.LogSoftmax(dim=1))

        #og vælger bare loss function og optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.003)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.1, patience=5)
        self.model.to(self.device)

    def train(self):
        EPOCHS = 2 #fra 200
        for epoch in range(EPOCHS):
            losses = []
            running_loss = 0
            for i, inp in enumerate(self.trainloader):
                
                inputs, labels = inp
                if torch.cuda.is_available():
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    
                self.optimizer.zero_grad()
            
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                losses.append(loss.item())

                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                if i%1 == 0 and i > 0:
                    print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 1)
                    running_loss = 0.0

            avg_loss = sum(losses)/len(losses)
            self.scheduler.step(avg_loss)
                    
        print('Training Done')
        torch.save(self.model.state_dict(), self.model_path)
        self.test()

    def KfoldTrain(self):
        transform_ting = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.Grayscale(1),
                                        transforms.ToTensor(),
                                        ])
            
        dataset = datasets.ImageFolder(self.data_dir,       
                        transform=transform_ting)

        len_dataset = len(dataset)
        torch.manual_seed(42)

        num_epochs=2 # fra 10
        batch_size=64 #fra 128
        k = 10
        splits=KFold(n_splits=k,shuffle=True,random_state=42)
        foldperf={}

        

        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len_dataset))):

            print('Fold {}'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

            for epoch in range(num_epochs):
                train_loss, train_correct=self.train_epoch(self.model,device,train_loader,self.criterion,self.optimizer)
                test_loss, test_correct=self.valid_epoch(self.model,device,test_loader,self.criterion)

                train_loss = train_loss / len(train_loader.sampler)
                train_acc = train_correct / len(train_loader.sampler) * 100
                test_loss = test_loss / len(test_loader.sampler)
                test_acc = test_correct / len(test_loader.sampler) * 100

                print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                    num_epochs,
                                                                                                                    train_loss,
                                                                                                                    test_loss,
                                                                                                                    train_acc,
                                                                                                                    test_acc))
                history['train_loss'].append(train_loss)
                history['test_loss'].append(test_loss)
                history['train_acc'].append(train_acc)
                history['test_acc'].append(test_acc)

            foldperf['fold{}'.format(fold+1)] = history  

    def train_epoch(self, model,device,dataloader,loss_fn,optimizer):
        train_loss,train_correct=0.0,0
        model.train()
        for images, labels in dataloader:

            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        return train_loss,train_correct
  
    def valid_epoch(self, model,device,dataloader,loss_fn):
        valid_loss, val_correct = 0.0, 0
        model.eval()
        for images, labels in dataloader:

            images,labels = images.to(device),labels.to(device)
            output = model(images)
            loss=loss_fn(output,labels)
            valid_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            val_correct+=(predictions == labels).sum().item()

        return valid_loss,val_correct

    def test(self):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.testloader:
                
                images, labels = data
                if torch.cuda.is_available():
                    images, labels = images.to('cuda'), labels.to('cuda')
                    
                outputs = self.model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy on 10,000 test images: ', 100*(correct/total), '%')
        print('Correct: ', correct, ' Total: ', total)

    def sampleExecution(self):
        filename = Path("../ChinaSet_AllFiles/CXR_png/CHNCXR_0001_0.png")

        input_image = Image.open(filename).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities)

        # Read the categories
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())


net = ResNet()
#net.sampleExecution()
net.KfoldTrain()