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


class ResNet():
    def __init__(self):
        self.model_path = './resnet.pth'
        self.data_dir = "./PP_data/" # "../ChinaSet_AllFiles/CXR_png/"
        #self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model = models.resnet50(pretrained=True) #altså den rigtige
        #self.model = torch.load(self.model_path)
        self.model.eval() #?
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loadModelSettings()

    def load_split_train_test(self, datadir, valid_size = .2):
        train_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        ])
        test_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        ])
        dataset = datasets.ImageFolder(datadir,       
                        transform=train_transforms)
        net.k_Fold_Cross_Validation(self, dataset)

        test_data = datasets.ImageFolder(datadir,
                        transform=test_transforms)
        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))
        np.random.shuffle(indices)
        from torch.utils.data.sampler import SubsetRandomSampler
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
    
    def k_Fold_Cross_Validation(self, dataset):

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits = 10, shuffle=True)

         # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')
            
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)




net = ResNet()
#net.sampleExecution()
#net.train()
