import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .Upload_dataset import Upload_dataset
import os


def select1(probas_val): #MarginSamplingSelection choose biggest prob each line 
    sorted, indices = torch.sort(probas_val,descending=True) # sort from biggest to smallest in each line
    print("--------------sorted-------------------")
    print(sorted)
    values = sorted[:, 0] - sorted[:, 1] # tells model is uncertain to both classes
    vsorted, vindices = torch.sort(values) # get first 25 samples of this set from probas_val 
    return vindices

def select2(probas_val): #MinStdSelection
    # select the samples where the std is smallest - i.e., there is uncertainty regarding the relevant class
    # and then train on these "hard" to classify samples.
    stddd = torch.std(probas_val,dim=1,unbiased=False)
    vsorted, vindices = torch.sort(stddd) #when model is uncertain to each classes ,prob of each classes will be similar. Std will be small
    return vindices # return 25 smallest std of each line

def select3(probas_val): #Entropyselection
    entropy = torch.sum(torch.mul(-probas_val, torch.log2(probas_val)),dim=1) #get entropy of each line according to function
    vsorted, vindices = torch.sort(entropy,descending=True) #sort from biggest to smallest in each line
    return vindices

def select4(probas_val): # Randomselection for compare
    vindices = torch.randperm(len(probas_val))
    return vindices



def select_Webimg(path,model,transform, k=8,select=1): 
    
    up_dataset = Upload_dataset(path,transform=transform)
    up_dataloder = DataLoader(up_dataset, batch_size=1, num_workers=0, drop_last=True)
    softmax = nn.Softmax(dim=1)
    prob_all = torch.tensor([])
    # iter = 0
    with torch.no_grad():
        for batch_idx, (img, img_path) in enumerate(up_dataloder):
            #img = img.cuda()
            with torch.autograd.set_grad_enabled(False):
                outputs = model(img) 
                outputs = softmax(outputs)
                outputs =  outputs.cpu()   
                prob_all = torch.cat((prob_all, outputs), 0) 
            # iter +=1
            # print(iter)
    if select == 1:
        vindices = select1(prob_all)
    elif select == 2:
        vindices = select2(prob_all)
    elif select == 3:
        vindices = select3(prob_all)
    else:
        vindices = select4(prob_all)

    img_index = vindices[:k]
    img_path_list = []
    for i in range(k):
        index = img_index[i].item()
        img,img_path = up_dataset[index] #tensor to int
        img_path_list.append(img_path) # get first k uncertain img path list



    return img_path_list