from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os






class Upload_dataset(Dataset): #crete dataloader for new folder and new img

    def __init__(self, path,transform = None):
        super(Upload_dataset, self).__init__()
        self.path = path #val[159,83,25,35]
        self.transform = transform
        imagelist = os.listdir(path) # get all imgs under this path folder
        self.imagelist = imagelist

    def __getitem__(self, index): 
        imgname = self.imagelist[index] 
        img_path = self.path + str(imgname)


        img = plt.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.imagelist)