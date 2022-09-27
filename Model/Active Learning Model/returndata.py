import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.random.seed(1)               # reproduce the result
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from returnVect import image_to_vector




def returndataset(n):
   dataset = np.zeros((1, n*n))
   with open('annotated_functional_test3_fixed.json','r',encoding='utf-8') as f:   #json file should be same path
    objectDict = json.load(f)   #load json
    len_annotation = len(objectDict['annotations'])  #annotations 897
    for idx in range(len_annotation):

       image_id = objectDict['annotations'][idx]['image_id']    #get image id
       curdamagepath = "../Annotated_images/" + str(image_id) + '_' + str(idx) + '.jpg'
       img = Image.open(curdamagepath)
       img = img.resize((n,n))
       vector = image_to_vector(img, n)
       
       dataset=np.vstack((dataset,vector))
    dataset = dataset[1:]
    X = dataset
    label_frame = pd.read_csv("label0-896.csv")
    label_frame_copy = label_frame.copy()
    y = label_frame_copy['human_label'].values

    return X, y

