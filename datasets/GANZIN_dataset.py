import os, torch, pickle
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.augmentation import *
import torchvision.transforms as T
import matplotlib.pyplot as plt

class GANZIN_dataset(Dataset):
    def __init__(self, data_path, phase, resize_h=480, resize_w=640, transform=True):
        """
        Args:
            data_path: directory to the dataset
            phase: train, val, test session
            resize_h: target resizing image in height
            resize_w: target resizing image in width
            transform: transformation to image
            experiment_name: setting the name for the training session
        """
        super(GANZIN_dataset, self).__init__()
        assert phase in ["train", "val", "test"]
        
        self.indexes = list(range(len(os.listdir(os.path.join(data_path, phase, "images")))))
        self.phase = phase

        self.resize_h = resize_h
        self.resize_w = resize_w
        self.transform = transform
        self.data_path = data_path

    def __getitem__(self, index):
        index = self.indexes[index]
        image_name = str(index) + ".jpg"
        label_name = str(index) + ".png"
        input_image = np.array(Image.open(os.path.join(self.data_path, self.phase, "images", image_name)).convert("RGB"))
        label = np.array(Image.open(os.path.join(self.data_path, self.phase, "binary_labels", label_name)))

        if self.transform:
            transform_functions = [RandomFlip(prob=0.5),\
                                    RandomCrop(crop_rate=0.1, prob=1.0), \
                                        RandomCropOut(crop_rate=0.2, prob=1.0), \
                                        RandomBrightness(bright_range=0.15, prob=0.9), \
#                                         RandomNoise(noise_range=5, prob=0.9),
                                ]
            for transform_func in transform_functions:
                input_image, label = transform_func(input_image, label)

        input_image = np.asarray(Image.fromarray(input_image).resize((self.resize_w, self.resize_h)))
        input_image = input_image.astype('float32')
        input_image = np.transpose(input_image, (2, 0, 1)) / 255.0
        label       = np.asarray(Image.fromarray(label).resize((self.resize_w, self.resize_h), resample=Image.NEAREST))
        label       = label.astype('int64')

        return torch.FloatTensor(input_image), torch.tensor(label), image_name

    def __len__(self):
        return len(self.indexes)