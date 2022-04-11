from pickletools import float8
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
import torch
from torch.utils.data.sampler import SubsetRandomSampler 
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime
import random


            
#kræver at nettet har self.model, self.criterion, self.optimizer. Evt. brug interface?
def KfoldTrain(net, model_name, checkpoint_every):
        transform_ting = transforms.Compose([
            transforms.Resize(net.input_size + 32), #fordi 224 + 32 = 256.
            transforms.CenterCrop(net.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

                           
        data_dir = "../PP_data/" # husk opdelt i TP_positive, TP_positive
        

        dataset = datasets.ImageFolder(data_dir,       
                        transform=transform_ting)

        #imagefolder konverterer vidst selv til RGB.

        torch.manual_seed(42)
        random.seed(42)
        


        batch_size=2 #fra 128
        k = 10 # dvs. hver fold er 1/10.
        splits=KFold(n_splits=k,shuffle=True,random_state=42) #random state randomizer, men med det samme resultat. (seed)
        foldperf={}

        
        #train_idx = ca. 90 % af træningssættet. Eks: [1,3,4,5......]
        #val_idx = ca. 10 % af træningssættet. Eks: [2, 13......]
        #og de storer bare indexer. 
        #og for hver fold skifter værdierne for train_idx og val_idx.

        #variable to get the correct number of steps or image processed in tensorboard
        total_steps = 0


        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(len(dataset) * 0.8))
        np.random.shuffle(indices)
        test_idx, kfold_idx = indices[split:], indices[:split]
        print(len(kfold_idx),len(test_idx))
        KFold_sampler = SubsetRandomSampler(kfold_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        #KFold_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=KFold_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

       #print(len(KFold_loader),len(test_loader))

        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(KFold_sampler)))):

            print('Fold {}'.format(fold + 1))

            #Creates eventlog files for tensorboard, saved in runs map, names the file as the modelname/datetime
            model_name_string = str(model_name)
            now = datetime.now()
            time_string = str(now.strftime("%d.%m.%Y.%H.%M.%S"))
            writter = SummaryWriter("runs/{}_{}_{}".format(fold + 1,model_name_string,time_string))

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

      
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}

             # Early stopping
            last_loss = 100
            patience = 2
            trigger_times = 0
            epoch = 0
            
            while epoch < 500:
                epoch += 1 
                
                train_loss, train_correct=train_epoch(net.model,device,train_loader,net.criterion,net.optimizer, net.is_inception)
                train_loss = train_loss / len(train_loader.sampler)
                train_acc = train_correct / len(train_loader.sampler) * 100

                #history['train_loss'].append(train_loss)            
                #history['train_acc'].append(train_acc)
                
                print(len(train_loader.sampler))
                total_steps += len(train_loader.sampler)

                #Her ændres hvor tit man vil tage checkpoint, 1 = hver gang
                if epoch % checkpoint_every == 0:
                    save_checkpoint(model_name, net, epoch)

                    val_loss, val_correct=valid_epoch(net.model,device,val_loader,net.criterion, net.is_inception)               
                    val_loss = val_loss / len(val_loader.sampler)
                    val_acc = val_correct / len(val_loader.sampler) * 100 
                    
                    #writes to the SummaryWritten tensorboard object. 
                    writter.add_scalar('TraniningLoss', train_loss, total_steps)
                    writter.add_scalar('TraniningAccuracy', train_acc, total_steps)
                    writter.add_scalar('ValLoss', val_loss, total_steps)
                    writter.add_scalar('ValAccuracy', val_acc, total_steps)



                    print("Epoch:{} AVG Training Loss:{:.8f} AVG Val Loss:{:.8f} AVG Training Acc {:.8f} % AVG Val Acc {:.8f} %".format(epoch,
                                                                                                                        train_loss,
                                                                                                                        val_loss,
                                                                                                                        train_acc,
                                                                                                                        val_acc))
                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    history['train_acc'].append(train_acc)
                    history['val_acc'].append(val_acc)

                    print('The Current Loss:', val_loss)

                    early_stopping_val_loss = val_loss
                    early_stopping_val_loss = format(early_stopping_val_loss, '.4f')
                    early_stopping_last_loss = last_loss
                    early_stopping_last_loss = format(early_stopping_last_loss, '.4f')
    
                    if early_stopping_val_loss > early_stopping_last_loss:
                        trigger_times += 1
                        print('Trigger Times:', trigger_times)
    
                        if trigger_times >= patience:
                            epoch = 505
                            print('Early stopping!\nStart to test process.')
                            foldperf['fold{}'.format(fold+1)] = history

                            test_loss, test_correct= test_method(net.model,device,test_loader,net.criterion, net.is_inception)
                            test_correct = test_correct/ len(test_loader.sampler) * 100
                            print(test_loss,test_correct)
                            
                    else:
                        print('trigger times: 0')
                        trigger_times = 0
                    last_loss = val_loss
                else:
                    print("Epoch:{} AVG Training Loss:{:.8f} AVG Training Acc {:.8f} %".format(epoch,
                                                                                            train_loss,                                                                                       
                                                                                            train_acc))

            #foldperf['fold{}'.format(fold+1)] = history  

def save_checkpoint(model_name, net, epoch):
        print("CHECKPOINT")
        now = datetime.now()
        epoch_string = str(epoch)
        model_name_string = str(model_name)
        time_string = str(now.strftime("%d.%m.%Y.%H.%M.%S"))
        model_file_string = f"./checkpoints/{model_name_string}/model/modelcheckpoint-{epoch_string}-{model_name_string}-{time_string}.pth"
        optimizer_file_string = f"./checkpoints/{model_name_string}/optimizer/optimizercheckpoint-{epoch_string}-{model_name_string}-{time_string}.pth"
        model_FILE = model_file_string
        optimizer_FILE = optimizer_file_string
        torch.save(net.model.state_dict(), model_FILE)
        torch.save(net.optimizer.state_dict(), optimizer_FILE)

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


def test_method(model,device,dataloader,loss_fn, is_inception):
    test_loss, test_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        #lav

        output = model(images)
        loss = loss_fn(output,labels)

        test_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        test_correct+=(predictions == labels).sum().item()

    return test_loss,test_correct
