from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from Methods.wandb import wandb_log
from Methods.parser import get_arguments

args = get_arguments()
#kræver at nettet har self.model, self.criterion, self.optimizer. Evt. brug interface?
def KfoldTrain(net):
        transform_ting = transforms.Compose([
            transforms.Resize(net.input_size + 32), #fordi 224 + 32 = 256.
            transforms.CenterCrop(net.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
            
        data_dir = net.data_dir # husk opdelt i TP_positive, TP_positive
        dataset = datasets.ImageFolder(data_dir,       
                        transform=transform_ting)
        #imagefolder konverterer vidst selv til RGB.

        torch.manual_seed(42)

        num_epochs=2 # fra 10
        batch_size=net.batch_size #fra 128
        k = 10 # dvs. hver fold er 1/10.
        splits=KFold(n_splits=k,shuffle=True,random_state=42) #random state randomizer, men med det samme resultat. (seed)
        foldperf={}

        
        #train_idx = ca. 90 % af træningssættet. Eks: [1,3,4,5......]
        #val_idx = ca. 10 % af træningssættet. Eks: [2, 13......]
        #og de storer bare indexer. 
        #og for hver fold skifter værdierne for train_idx og val_idx.
        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            print('Fold {}'.format(fold + 1))

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

            for epoch in range(num_epochs):
                train_loss, train_correct=train_epoch(net.model,device,train_loader,net.criterion,net.optimizer, net.is_inception)
                test_loss, test_correct=valid_epoch(net.model,device,test_loader,net.criterion, net.is_inception)

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
                if args.wandb:
                    wandb_log(train_loss, test_loss, train_acc, test_acc)

            foldperf['fold{}'.format(fold+1)] = history  

def train_epoch(model,device,dataloader,loss_fn,optimizer, is_inception):
        train_loss,train_correct=0.0,0
        model.train()
        for images, labels in dataloader:

            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()

            if is_inception:
                output, aux_outputs = model(images)
                loss1 = loss_fn(output, labels)
                loss2 = loss_fn(aux_outputs, labels)
                loss = loss1 + 0.4*loss2
            else:
                output = model(images)
                loss = loss_fn(output,labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        return train_loss,train_correct
  
def valid_epoch(model,device,dataloader,loss_fn, is_inception):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        #lav

        output = model(images)
        loss = loss_fn(output,labels)

        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct
