import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd
from PIL import Image

torch.cuda.set_device(0) # Set gpu number here
torch.cuda.manual_seed(0)
np.random.seed(0) # Keep val_list controlable

######################################################################################################################################### Set some basic parameters

classes = ('Dent','Other','Rim','Scratch') # Used to show each classes accuracy

label_frame = pd.read_csv("label0-896_clean.csv") # load csv clean as original dataframe
label_frame_copy = label_frame.copy()
label_frame_copy['human_label'] = label_frame_copy['human_label'].map(lambda x: x-1) # Set each label from 0-3 for nn.CrossEntropyLoss()  


csv_index_list = np.arange(label_frame_copy.shape[0]) # Split for train and test

Val_size = 600 # Here to change Val_size
val_index_ori = np.random.choice(csv_index_list, size=Val_size, replace=False) # Val is represented by index of csv

test_index_ori = [value for value in csv_index_list if value not in val_index_ori] # Test is chosen except for Val

val_dataframe = label_frame_copy.iloc[val_index_ori,:] # all info about val is saved in this dataframe 
test_dataframe = label_frame_copy.iloc[test_index_ori,:] # Dataframe will not change, but val list will change in AL process

test_index_list = np.arange(test_dataframe.shape[0])  # test_index_list will not change so it is placed outside the algorithm

####################################################################### Pick 50 images from the test to simulate user-uploaded images.

sample_test_list = test_index_list[:50] #Pick 50 images from the test to simulate user-uploaded images.
sample_test_dataframe = test_dataframe.iloc[:50,:]

def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)           
		print ("---  new folder...  ---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")

new_folder = '../Annotated_images_test/' #create folder for customer
mkdir(new_folder)     


img_folder = "../Annotated_images_224/"

for i in range(len(sample_test_list)):
    csv_index = sample_test_list[i]
    annotation_index = test_dataframe.iloc[csv_index,0]
    annotation_index=annotation_index.astype(int)
    image_id = test_dataframe.iloc[csv_index,1]
    img_path = img_folder + str(image_id) + '_' + str(annotation_index) + '.jpg'
    img = Image.open(img_path)
    saveImagepath = new_folder + str(image_id) + "_" + str(annotation_index) + ".jpg"
    img.save(saveImagepath)

sample_test_dataframe.to_csv('sample_test_dataframe.csv',index=False,header=True) #save used img in folder
print("sample_test_dataframe.csv saved")