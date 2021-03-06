from pkgutil import get_data
from numpy import string_
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets
import shutil
import os
from sklearn.model_selection import KFold

def label_files(img_dir):
    mypath = 'PP_data'
    for f in os.listdir(img_dir):
        # move if TB positive
        if f.endswith('1.png'):
            shutil.move(os.path.join(img_dir, f), os.path.join(mypath,"TB_Positive"))
        # move if TB negative
        elif f.endswith('0.png'):
            shutil.move(os.path.join(img_dir, f), os.path.join(mypath, "TB_Negative"))
        # notify if something else
        else:
            print('Could not categorize file with name %s' % f)


def get_dataloader(img_dir, batch_size):
    label_files(img_dir)

    normalize = transforms.Compose([
                        transforms.Resize(256),
                        transforms.Grayscale(1),
                        transforms.CenterCrop(256),
                        transforms.ToTensor()])

    normalized_dataset = datasets.ImageFolder(root = "PP_data", transform = normalize)

    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.Grayscale(1),
                        transforms.CenterCrop(256),
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                        transforms.RandomHorizontalFlip(p=0.3),
                        transforms.RandomAutocontrast(p=0.3),
                        transforms.RandomAffine(degrees = (0,30), translate = (0.3,0.1)),
                        transforms.ToTensor()])



    transformed_dataset = datasets.ImageFolder(root = "PP_data", transform = transform)

    concat_dataset = ConcatDataset([normalized_dataset, transformed_dataset])
  
    # list_of_classes=list(map(str, list(transformed_dataset.classes)) )

    # print(list_of_classes)
    # for idx, (sample, target) in enumerate(concat_dataset):
    #     print(sample, list_of_classes[target] )

    dl_cds = DataLoader(concat_dataset, batch_size = batch_size, shuffle=True)
    return dl_cds


#Kims Path: 'C:/Users/Monkk/OneDrive/Dokumenter/AAU/CS/CS2/01.P8/Data/ChinaSet_AllFiles/CXR_png'
#Dennis Station??r: D:/Downloads/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png
#Dennis laptop: C:/Users/Dennis/Downloads/xrayTB/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png'


#usage:
#img_dir ='C:/Users/Dennis/Downloads/xrayTB/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png'
#batchsize = 5
#get_dataloader(img_dir, batchsize)

