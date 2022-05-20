from copy import deepcopy
from pickletools import float8
from pyexpat import model
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold, cross_validate
import torch
from torch.utils.data.sampler import SubsetRandomSampler 
import numpy as np
from torch.utils.tensorboard import SummaryWriter 
from datetime import datetime
import random
import copy
import os
import glob 
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy

from Methods.wandb import wandb_log, wandb_log_folds_avg
from Methods.parser import get_arguments
from Methods.ImageFolderWithPaths import ImageFolderWithPaths

args = get_arguments()
#kræver at nettet har self.model, self.criterion, self.optimizer. Evt. brug interface?
def KfoldTrain(net):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(net.model_name)
        #Bruges til at lave en kopi af parameterne før model er kørt
        #De bruges til næste fold, så modellen bliver reset.
        parameterDefault = copy.deepcopy(net.model.state_dict())
        optimizerDefault = copy.deepcopy(net.optimizer.state_dict())            
        
        train_data_dir = f"{net.data_dir}/train/" # husk opdelt i TP_positive, TP_positive
        test_data_dir = f"{net.data_dir}/test/"

        #billederne hvis der flippes, roteres osv.
        train_dataset = ImageFolderWithPaths(train_data_dir,       
                        transform=net.train_transform)   

        #andet end train, da vi ikke vil rotere og flipper billerne osv.
        val_dataset = ImageFolderWithPaths(train_data_dir,       
                        transform=net.val_transform)

        #andre billeder. Specifikt udvalgt til test.
        test_dataset =  ImageFolderWithPaths(test_data_dir,       
                        transform=net.val_transform)    
        
        torch.manual_seed(42)

        batch_size=net.batch_size
        k = 3 # dvs. hver fold er 1/3.
        
        splits=KFold(n_splits=k,shuffle=True,random_state=42) #random state randomizer, men med det samme resultat. (seed)
        foldperf={}
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        cumu_val_acc = 0.0
        cumu_val_loss = 0.0
        cumu_test_acc = 0.0

        #region To get final test result for all folds
        Total_Test_Avg_Loss = 0
        Total_Test_Avg_Acc = 0
        Total_Test_Avg_Sensitivity = 0
        Total_Test_Avg_precision = 0
        Total_Test_Avg_specificity = 0
        Total_Test_Avg_FalseNegativeRate = 0
        Total_Test_Avg_FalsePositiveRate = 0
        #endregion

        for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
            #Load parameter and optimizer fra inden modellen er kørt
            net.model.load_state_dict(parameterDefault)
            net.optimizer.load_state_dict(optimizerDefault)

            print('Fold {}'.format(fold + 1))

            #Creates eventlog files for tensorboard, saved in runs map, names the file as the modelname/datetime
            model_name_string = str(net.model_name)
            now = datetime.now()
            time_string = str(now.strftime("%d.%m.%Y.%H.%M.%S"))
            writter = SummaryWriter("runs/{}_{}_{}".format(fold + 1,model_name_string,time_string))
            #variable to get the correct number of steps or image processed in tensorboard
            total_steps = 0

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, pin_memory=True)

            history = {'train_loss': [], 'val_loss': [],'train_acc':[],'val_acc':[]}

            # Early stopping
            best_val_loss = 9999999999999999999.9
            best_val_loss_acc = 0.0
            patience = 5 #bare til hyper tuning.
            trigger_times = 0
            epoch = 0

            val_loss, CMVAL=valid_epoch(net.model,device,val_loader,net.criterion, net.is_inception)    
            
            while epoch < 500: 
                epoch += 1
                if epoch < 4: #lidt nemmere debug
                    print('Epoch: ', epoch) 
                train_loss, CMTRAIN=train_epoch(net.model,device,train_loader,net.criterion,net.optimizer, net.is_inception)
                train_loss = train_loss / len(train_loader.sampler)
                train_acc = (np.sum(np.diag(CMTRAIN)/np.sum(CMTRAIN))*100)
          
                if net.scheduler is not None:
                    net.scheduler.step()

                #region cmting
                tn=CMTRAIN[0][0]
                tp=CMTRAIN[1][1]
                fp=CMTRAIN[0][1]
                fn=CMTRAIN[1][0]

                only_negative_guesses = True if tp + fp == 0 else False

                train_sensitivity= (tp/(tp+fn))*100
                train_precision= 0 if only_negative_guesses else (tp/(tp+fp))*100
                train_specificity= (tn/(tn+fp))*100
                train_FalseNegativeRate= (1-(tp/(tp+fn)))*100
                train_FalsePositiveRate = (1-(tn/(tn+fp)))*100
               

                #history['train_loss'].append(train_loss)            
                #history['train_acc'].append(train_acc)
                #endregion
                
                total_steps += len(train_loader.sampler)

                #Her ændres hvor tit man vil tage checkpoint, 1 = hver gang
                if epoch % 3 == 0:
                    

                    val_loss, CMVAL=valid_epoch(net.model,device,val_loader,net.criterion, net.is_inception)               
                    val_loss = val_loss / len(val_loader.sampler)
                    val_acc = (np.sum(np.diag(CMVAL)/np.sum(CMVAL))*100)

                    if math.isnan(val_loss): #hvis resultatet er helt ude i hampen, bliver det nan. = nicht gud
                        val_loss = 9999999999999999999.9
                    
                    #region Description
                    tn1=CMVAL[0][0]
                    tp1=CMVAL[1][1]
                    fp1=CMVAL[0][1]
                    fn1=CMVAL[1][0]

                    only_negative_guesses1 = True if tp1 + fp1 == 0 else False

                    Val_sensitivity= (tp1/(tp1+fn1))*100
                    val_precision=  0 if only_negative_guesses1 else (tp1/(tp1+fp1))*100
                    val_specificity= (tn1/(tn1+fp1))*100
                    val_FalseNegativeRate= (1-(tp1/(tp1+fn1)))*100
                    val_FalsePositiveRate = (1-(tn1/(tn1+fp1)))*100

                    
                    #writes to the SummaryWritten tensorboard object. 
                    writter.add_scalar('TraniningLoss', train_loss, total_steps)
                    writter.add_scalar('TraniningAccuracy', train_acc, total_steps)
                    writter.add_scalar('TraniningSensitivity', train_sensitivity, total_steps)
                    writter.add_scalar('TraniningPrecision', train_precision, total_steps)
                    writter.add_scalar('TraniningSpecificity', train_specificity, total_steps)
                    writter.add_scalar('TraniningFalseNegativeRate', train_FalseNegativeRate, total_steps)
                    writter.add_scalar('TraniningFalsePositiveRate', train_FalsePositiveRate, total_steps)
                    writter.add_scalar('ValLoss', val_loss, total_steps)
                    writter.add_scalar('ValAccuracy', val_acc, total_steps)
                    writter.add_scalar('ValSensitivity', Val_sensitivity, total_steps)
                    writter.add_scalar('ValPrecision', val_precision, total_steps)
                    writter.add_scalar('ValSpecificity', val_specificity, total_steps)
                    writter.add_scalar('ValFalseNegativeRate', val_FalseNegativeRate, total_steps)
                    writter.add_scalar('ValFalsePositiveRate', val_FalsePositiveRate, total_steps)

                    



                    print("Epoch:{} AVG Training Loss:{:.8f} AVG Val Loss:{:.8f} AVG Training Acc {:.2f} % AVG Val Acc {:.2f} %".format(epoch,
                                                                                                                        train_loss,
                                                                                                                        val_loss,
                                                                                                                        train_acc,
                                                                                                                        val_acc))
                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    history['train_acc'].append(train_acc)
                    history['val_acc'].append(val_acc) 
                    #endregion   

                    if args.wandb:
                        wandb_log("train", train_loss, train_acc, train_sensitivity, train_precision, train_specificity, train_FalseNegativeRate, train_FalsePositiveRate)
                        wandb_log("val", val_loss, val_acc, Val_sensitivity, val_precision, val_specificity, val_FalseNegativeRate, val_FalsePositiveRate)           

                    #print('The Current Loss:', val_loss)
                    val_loss_rounded = float(format(val_loss, '.4f'))

                    if val_loss_rounded >= best_val_loss: #ergo: den nye val er dårligere eller ligeså god

                        trigger_times += 1
                        print('Trigger Times:', trigger_times)
                        
                        if trigger_times >= patience:
                            #og så træn på valideringsfolden i forrige_epochs/2 antal epochs, da vi antager en nogenlunde lineær korrelation.
                            total_epoch_for_val_training = math.floor(epoch / 2)

                            print("TRÆNER PÅ VAL SÆT!!!")
                            for epoch in range(total_epoch_for_val_training):
                                train_loss, CMTRAIN=train_epoch(net.model,device,val_loader,net.criterion,net.optimizer, net.is_inception) #bemærk: val loader!
                            epoch = 505
                            print('Early stopping!\nStarting the test process.')
                            foldperf['fold{}'.format(fold+1)] = history
                            # x  = 0
                            # avg_test_loss = 0
                            # avg_test_acc = 0

                            checkpoint_file_model = load_checkpoint(net, string = "model")
                            checkpoint_file_optimizer = load_checkpoint(net, string = "optimizer")
                            net.model.load_state_dict(torch.load(checkpoint_file_model))
                            net.optimizer.load_state_dict(torch.load(checkpoint_file_optimizer))
                            test_loss, CMTEST= test_method(net, net.model,device,test_loader,net.criterion, net.is_inception)
                            test_loss = test_loss / len(test_loader.sampler)
                            test_correct = (np.sum(np.diag(CMTEST)/np.sum(CMTEST))*100)

                            #region cmtest
                            tn2=CMTEST[0][0]
                            tp2=CMTEST[1][1]
                            fp2=CMTEST[0][1]
                            fn2=CMTEST[1][0]
                            test_sensitivity= (tp2/(tp2+fn2))*100
                            test_precision= (tp2/(tp2+fp2))*100
                            test_specificity= (tn2/(tn2+fp2))*100
                            test_FalseNegativeRate= (1-(tp2/(tp2+fn2)))*100
                            test_FalsePositiveRate = (1-(tn2/(tn2+fp2)))*100
                    
                            Total_Test_Avg_Loss += test_loss
                            Total_Test_Avg_Acc += test_correct
                            Total_Test_Avg_Sensitivity += test_sensitivity
                            Total_Test_Avg_precision += test_precision
                            Total_Test_Avg_specificity += test_specificity
                            Total_Test_Avg_FalseNegativeRate += test_FalseNegativeRate
                            Total_Test_Avg_FalsePositiveRate += test_FalsePositiveRate
                            writter.add_scalar('TestLoss', test_loss, fold)
                            writter.add_scalar('TestAccuracy', test_correct, fold)
                            writter.add_scalar('TestSensitivity', test_sensitivity, fold)
                            writter.add_scalar('TestPrecision', test_precision, fold)
                            writter.add_scalar('TestSpecificity', test_specificity, fold)
                            writter.add_scalar('TestFalseNegativeRate', test_FalseNegativeRate, fold)
                            writter.add_scalar('TestFalsePositiveRate', test_FalsePositiveRate, fold)
                            #endregion
                            if args.wandb:
                                wandb_log("test", test_loss, test_correct, test_sensitivity, test_precision, test_specificity, test_FalseNegativeRate, test_FalsePositiveRate)
                            print("Test Loss:{:.8f}, Test Acc:{:.8f} %, Test Sensitivity:{:.8f} %, Test Precision:{:.8f} % ".format(test_loss, test_correct, test_sensitivity, test_precision))
                            cumu_val_acc += best_val_loss_acc
                            cumu_val_loss += best_val_loss
                            print('cumu val loss: ' ,cumu_val_loss)
                                
                    else: #ergo: den nye val er bedre
                        save_checkpoint(net, fold)
                        trigger_times = 0
                        best_val_loss = val_loss_rounded
                        best_val_loss_acc = val_acc

            break #KUN 1 FOLD

        avg_val_acc = cumu_val_acc / float(k)
        avg_val_loss = cumu_val_loss / float(k)
        print('avg_val_acc er: ', avg_val_acc)
        wandb_log_folds_avg(avg_val_acc, avg_val_loss)

        #region log ting
        Total_Test_Avg_Loss = Total_Test_Avg_Loss / k
        Total_Test_Avg_Acc = Total_Test_Avg_Acc / k
        Total_Test_Avg_Sensitivity = Total_Test_Avg_Sensitivity / k
        Total_Test_Avg_precision = Total_Test_Avg_precision / k
        Total_Test_Avg_specificity = Total_Test_Avg_specificity / k
        Total_Test_Avg_FalseNegativeRate = Total_Test_Avg_FalseNegativeRate / k
        Total_Test_Avg_FalsePositiveRate = Total_Test_Avg_FalsePositiveRate / k
        print("Total Avg Test Loss:{:.8f}, Total Avg Test Acc:{:.8f} %".format(Total_Test_Avg_Loss, Total_Test_Avg_Acc))
        writter.add_scalar('TotalAVGTestLoss', Total_Test_Avg_Loss, fold)
        writter.add_scalar('TotalAVGTestAccuracy', Total_Test_Avg_Acc, fold)
        writter.add_scalar('TotalAVGTestSensitivity', Total_Test_Avg_Sensitivity, fold)
        writter.add_scalar('TotalAVGTestPrecision', Total_Test_Avg_precision, fold)
        writter.add_scalar('TotalAVGTestSpecificity', Total_Test_Avg_specificity, fold)
        writter.add_scalar('TotalAVGTestFalseNegativeRate', Total_Test_Avg_FalseNegativeRate, fold)
        writter.add_scalar('TotalAVGTestFalsePositiveRate', Total_Test_Avg_FalsePositiveRate, fold)
        #endregion

def roc_auc(net, scores, labels):
        # yolo = roc_auc_score(labels.cpu(), scores.cpu())
        # print(yolo)
        new_scores = []
        new_lables = []

        for x in labels:
            new_lables.append(x.cpu().numpy())

        for x in scores:
            new_scores.append(x.cpu().numpy())

        scores2 = numpy.array(new_scores).flatten()
        labels2 = numpy.array(new_lables).flatten()
        fpr1, tpr1, thresh1 = roc_curve(labels2, scores2)
        print(thresh1)
        xx = auc(fpr1,tpr1)
        print(xx)

        #print("aucscore: {}".format(yolo))
        plt.style.use('seaborn')
        # plot roc curves
        plt.plot(fpr1, tpr1, linestyle='--',color='orange', label= net.model_name)
        plt.xlim([0, 0.14])
        plt.ylim([0.8, 1])
        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        timenow = datetime.now()
        time_string1 = str(timenow.strftime("%d.%m.%Y.%H.%M.%S"))
        plt.savefig('/content/{}_{}.png'.format(net.model_name, time_string1),dpi=300, transparent=True)

def save_checkpoint(net, fold):
        fold_string = str(fold)
        model_name_string = str(net.model_name)
        model_file_string = f"/content/x-map/src/checkpoints/{model_name_string}/model/modelcheckpoint-{fold_string}-{model_name_string}.pth"
        optimizer_file_string = f"/content/x-map/src/checkpoints//{model_name_string}/optimizer/optimizercheckpoint-{fold_string}-{model_name_string}.pth"
        model_FILE = model_file_string
        optimizer_FILE = optimizer_file_string
        torch.save(net.model.state_dict(), model_FILE)
        torch.save(net.optimizer.state_dict(), optimizer_FILE)

def fp_similar_diseases(paths, predictions, similar_diseases_fp_dict):

    disease_name_identifiers = {
        'pneumonia': 'person', 
        'lung_cancer': 'jpcln', 
        'covid': 'covid',
    }

    normal_name_identifiers = ['normal', 'jpcnn', '_0.png', 'hejmor']

    for i in range(len(paths)):
        path = paths[i].lower()

        for disease in disease_name_identifiers:
            if disease_name_identifiers[disease] in path:
                predict_status = 'tn' if predictions[i] == 0 else 'fp'
                similar_diseases_fp_dict[disease][predict_status] += 1

        for identifier in normal_name_identifiers:
            if identifier in path:
                predict_status = 'tn' if predictions[i] == 0 else 'fp'
                similar_diseases_fp_dict['normal'][predict_status] += 1


def train_epoch(model,device,dataloader,loss_fn,optimizer, is_inception):
        train_loss=0.0
        CMTRAIN = 0
        model.train()
        for images, labels, paths in dataloader:
            #if args.wandb_data_augmentation:
                #plt.imshow(  images[0].permute(1, 2, 0)  )
                #plt.show()

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
            CMTRAIN+=confusion_matrix(labels.cpu(), predictions.cpu(), labels =[0,1])           

        return train_loss,CMTRAIN
  
def valid_epoch(model,device,dataloader,loss_fn, is_inception):
    valid_loss=0.0
    CMVAL = 0
    model.eval()

    with torch.no_grad(): #undgå cuda out of memoery fejl?
        for images, labels, paths in dataloader:

            images,labels = images.to(device),labels.to(device)
            #lav

            output = model(images)
            loss = loss_fn(output,labels)

            valid_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            CMVAL+=confusion_matrix(labels.cpu(), predictions.cpu(), labels =[0,1]) 

    return valid_loss,CMVAL


def test_method(net, model,device,dataloader,loss_fn, is_inception):
    similar_diseases_fp_dict = {
        'normal': {
            'fp': 0,
            'tn': 0,
        },
        'pneumonia': {
            'fp': 0,
            'tn': 0,
        },
        'lung_cancer': {
            'fp': 0,
            'tn': 0,
        },
        'covid': {
            'fp': 0,
            'tn': 0,
        },
    }

    SigmoidScores = []
    labelsAUROC = []

    test_loss=0.0
    CMTEST = 0
    model.eval()

    with torch.no_grad(): #undgå cuda out of memoery fejl?
        for images, labels, paths in dataloader:

            images,labels = images.to(device),labels.to(device)
            #lav

            output = model(images)
            loss = loss_fn(output,labels)
            sigmoidpred= torch.softmax(output, dim=1)

            test_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            fp_similar_diseases(paths, predictions, similar_diseases_fp_dict)
            CMTEST+=confusion_matrix(labels.cpu(), predictions.cpu(), labels =[0,1]) 
            for x in sigmoidpred:
                scoresSig = x.data[1]
                SigmoidScores.append(scoresSig)
            for x in labels:
                labelsAUROC.append(x)

    roc_auc(net, SigmoidScores, labelsAUROC)
    print(args.name, ' fp dict: ')
    print(similar_diseases_fp_dict)
    return test_loss,CMTEST

def load_checkpoint(net, string):
    #load model
    if string == "model":
        path_string = f"/content/x-map/src/checkpoints/{net.model_name}/model/*.pth"
        list_of_files = glob.glob(f'{path_string}') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print("You have loaded the modelcheckpoint: " + latest_file)
        FILE = f"{latest_file}"
        #checkpoint = self.model.load_state_dict(torch.load(FILE)) # it takes the loaded dictionary, not the path file itself
        #load optimizer
       
    if string == "optimizer":
        path_string = f"/content/x-map/src/checkpoints/{net.model_name}/optimizer/*.pth"
        list_of_files = glob.glob(f'{path_string}') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print("You have loaded the optimizercheckpoint: " + latest_file)
        FILE = f"{latest_file}"
        #checkpoint = self.model.load_state_dict(torch.load(FILE)) # it takes the loaded dictionary, not the path file itself
   
    return FILE


