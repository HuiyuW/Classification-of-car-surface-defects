import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.label_file = pd.read_csv(annotations_file)
        self.img_labels = list(self.label_file.iloc[:, 2] - 1)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.annotations_file = annotations_file

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        file = pd.read_csv(self.annotations_file) # 'label0-896.csv'
        img_name = str(file.iloc[idx, 0]) + '_' + str(file.iloc[idx, 1]) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        # image = read_image(img_path)
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]
        # image = Image.open(image).convert('RGB')
        if self.transform:
           image = self.transform(image)      
        if self.target_transform:
            label = self.target_transform(label)
        return image, label