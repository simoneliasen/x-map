import torch
from torchvision import transforms, datasets
import pandas as pd

batch_size=5
normalized_dataset = datasets.ImageFolder(root = "PP_data")
transform = { 'train': transforms.Compose([
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.RandomAutocontrast(p=0.3),
                    transforms.RandomAffine(degrees = (0,30), translate = (0.3,0.1)),
                    transforms.ToTensor()]),
                'test': transforms.Compose([
                    #
                ])
}

transformed_dataset = datasets.ImageFolder(root = "PP_data", transform = transform)


#classes with corresponding id. (tb_negative 0, tb_positive 1)
print(transformed_dataset.class_to_idx)

#Paths of all pictures and corresponding labels
#print(transformed_dataset.imgs)
