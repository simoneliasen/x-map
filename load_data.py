import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, datasets
import pandas as pd
from sklearn.model_selection import KFold

normalized_dataset = datasets.ImageFolder(root = "PP_data", transform = transforms.Compose([transforms.ToTensor()]))

mean = img_tr.mean([1,2])
transform = transforms.Compose([
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.RandomAutocontrast(p=0.3),
                    transforms.RandomAffine(degrees = (0,30), translate = (0.3,0.1)),
                    transforms.Grayscale(1),
                    transforms.ToTensor()])



transformed_dataset = datasets.ImageFolder(root = "PP_data", transform = transform)

concat_dataset = ConcatDataset([normalized_dataset, transformed_dataset])

train_dataset = DataLoader(transformed_dataset, batch_size = 5, shuffle=True)
#test_dataset = DataLoader(concat_dataset, batch_size = 5, shuffle=True)

for i in range(1):
    for x in normalized_dataset:
        print(x)

# dataset = ConcatDataset([train_dataset, test_dataset])

# #k-fold cross validation

# kfold = KFold(n_splits = 10, shuffle = True, random_state = 32)

# # enumerate splits
# for train, test in kfold.split(dataset):
# 	print('train: %s, test: %s' % (train, test))
#classes with corresponding id. (tb_negative 0, tb_positive 1)
#print(transformed_dataset.class_to_idx)

#Paths of all pictures and corresponding labels
#print(transformed_dataset.imgs)
