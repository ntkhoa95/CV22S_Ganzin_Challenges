import torchvision
import numpy as np
import os, time, torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from datasets.GANZIN_dataset import *

data_path = "./ganzin_dataset_final/train"

transform_img = transforms.Compose([
    transforms.ToTensor(),
])

image_data = torchvision.datasets.ImageFolder(
  root=data_path, transform=transform_img
)

image_data_loader = DataLoader(
  image_data, 
  batch_size=len(image_data), 
  shuffle=False, 
  num_workers=8
)

image_data_loader = DataLoader(
  image_data, 
  # batch size is whole datset
  batch_size=len(image_data), 
  shuffle=False, 
  num_workers=8)

def mean_std(loader):
    images, labels = next(iter(loader))
    # shape of images = [b,c,w,h]
    mean, std = images.mean([0,2,3]), images.std([0,2,3])
    return mean, std

mean, std = mean_std(image_data_loader)
print("mean and std: \n", mean, std)