import os
import json
import torch
import numpy as np
from torch.utils.data import random_split


root_path=os.getcwd()

with open('annotated_functional_test3_fixed.json','r',encoding='utf-8') as f:   #json file should be same path
 objectDict = json.load(f)   #load json
 len_annotation = len(objectDict['annotations'])  #annotations 897 


torch.manual_seed(0)
test_split = 0.3 #val:test = 0.7:0.3
test_size = int(test_split * 805)
val_size = 805 - test_size
val_index_list, test_index_list = random_split(np.arange(805),[val_size, test_size]) # val set and test set are 
val_all_list = list(val_index_list)
test_list = list(test_index_list)
train_size = int(6*len(val_all_list)/7)
val_size = len(val_all_list)-train_size
val_index_list, train_index_list = random_split(val_all_list,[val_size, train_size])
val_list = list(val_index_list)
train_list = list(train_index_list)

def create_set(input_list_name,input_txt_name,key):
    sample_set_path=root_path+"/"+"sample_set_new"
    if not os.path.exists(sample_set_path):
       os.mkdir(sample_set_path)
    txt_file=open(sample_set_path+"/"+key+".txt",'w+')
    for item in eval(input_list_name):
        txt_file.write(str(item)+'\n')
    # for item in input_list_name:
    #     txt_file.write(str(item)+'\n')
    # for item in eval(input_list_name):
    #     txt_file.write(labels_path+'/'+item+'\n')
    txt_file.close()
 
key_name=['train','val','test']
for name in key_name:
    list_name=name+"_list"
    txt_name=name+".txt"
    create_set(list_name,txt_name,name)