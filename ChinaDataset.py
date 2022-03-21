from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transfrom
import urllib
import os
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class ChinaDataset(Dataset):
     def __init__(self, csv_file, root_dir, transform=None):
         self.annotations = pd.read_csv(csv_file)
         self.root_dir = root_dir
         self.transform = transform

     def __len__(self):
         return len(self.annotations)

     def __getitem__(self,index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path) #.convert("RGB")
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


 #Load Data
#dataset = ChinaDataset(csv_file = 'ChinaData.csv', root_dir = '../ChinaSet_AllFiles/CXR_png', transform = transforms.ToTensor())

#train_set, test_set = torch.utils.data.random_split(dataset, [562, 100])

#batch_size = 4
#train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)