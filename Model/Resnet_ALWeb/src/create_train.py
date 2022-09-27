import numpy as np
import torch
import os
from PIL import Image

torch.cuda.set_device(0) # Set gpu number here
torch.cuda.manual_seed(0)
np.random.seed(0) # Keep val_list controlable

######################################################################################################################################### Set some basic parameters
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)           
		print ("---  new folder...  ---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")

def create_train(train_df): # create train set folder
    train_folder = './Annotated_images_train/' #create folder for customer
    mkdir(train_folder) 
    for i in range(train_df.shape[0]):
        annotation_index = train_df.iloc[i,0]
        annotation_index=annotation_index.astype(int)
        image_id = train_df.iloc[i,1]
        img_path = './Annotated_images_224/' + str(image_id) + '_' + str(annotation_index) + '.jpg'
        img = Image.open(img_path)
        saveImagepath = train_folder + str(image_id) + "_" + str(annotation_index) + ".jpg"
        img.save(saveImagepath)


# train_df = pd.read_csv('./results/AL_15_accuracy_0.8174_images_375select_3.csv')
# create_train(train_df)