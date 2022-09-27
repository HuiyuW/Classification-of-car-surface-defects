from matplotlib import pyplot as plt
from torch.utils.data import Dataset








class WENN_dataset(Dataset): # get img from folder we created named Annotated img 224 and get ready for dataloader

    def __init__(self, index_list,dataframe,transform = None):
        super(WENN_dataset, self).__init__()
        self.index_list = index_list
        self.transform = transform
        self.dataframe = dataframe


    def __getitem__(self, index): 
        # img_name = self.img_folder + self.filenames[index]
        csv_index = self.index_list[index]
        annotation_index = self.dataframe.iloc[csv_index,0]
        annotation_index=annotation_index.astype(int)
        image_id = self.dataframe.iloc[csv_index,1]
        label = self.dataframe.iloc[csv_index,2]

        img_path = 'imgProcess/Resnet_ALWeb/Annotated_images/' + str(image_id) + '_' + str(annotation_index) + '.jpg'

        img = plt.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)
        label = label.squeeze()

        return img, label
    def count_classes(self): # count classes distribution from index_list
        label_list = self.dataframe.iloc[self.index_list,2].values
        dict = {}
        for key in label_list:
            dict[key] = dict.get(key, 0) + 1 #count num in each classes
        # print(dict)
        return dict


    def __len__(self):
        return len(self.index_list)


