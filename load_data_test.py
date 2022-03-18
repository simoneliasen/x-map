import torch
from torchvision import transforms, datasets
import pandas as pd

batch_size=5
dataset = datasets.ImageFolder(root = "/PP_data")
train_loader = torch.utils.data.Dataloader(
   dataset, transform=transforms.Compose([
                      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                      transforms.RandomHorizontalFlip(p=0.3),
                      transforms.RandomAutocontrast(p=0.3),
                      transforms.RandomAffine(degrees = (0,30), translate = (0.3,0,1)),
                      transforms.ToTensor()
                  ]), batch_size=batch_size, shuffle=True)