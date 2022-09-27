from matplotlib import pyplot as plt
from torch.utils.data import Dataset

import numpy as np

class combine_dataset(Dataset): 

    def __init__(self, dataframe,transform = None):
        super(combine_dataset, self).__init__()
        self.dataframe = dataframe #val[159,83,25,35]
        self.img_folder = "imgProcess/Resnet_ALWeb/Annotated_images/"
        self.transform = transform

    def __getitem__(self, index): 
        image_id = self.dataframe.iloc[index,1]
        if image_id == 0: # if id is 0 then its newly added images from user folder
            img_path = self.dataframe.iloc[index,0]
            img = plt.imread(img_path)
            img = img.astype(np.uint8)
            label = self.dataframe.iloc[index,2]
        else: # otherweise its original train set from Annotated images 224
            annotation_index = self.dataframe.iloc[index,0]
            image_id = self.dataframe.iloc[index,1]
            label = self.dataframe.iloc[index,2]

            img_name = self.img_folder + str(image_id) + '_' + str(annotation_index) + '.jpg'
            img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)
        label = label.squeeze()

        return img, label
    
    def count_classes(self): # count classes distribution from dataframe
        label_list = self.dataframe.iloc[:,2].values
        dict = {}
        for key in label_list:
            dict[key] = dict.get(key, 0) + 1 #count num in each classes
        # print(dict)
        return dict        



    def __len__(self):
        return self.dataframe.shape[0]