import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.WENN_dataset import WENN_dataset

def select1(probas_val): #MarginSamplingSelection choose biggest prob each line 
    sorted, indices = torch.sort(probas_val,descending=True) # sort from biggest to smallest in each line
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





def play_query(val_index_list,model,dataframe,transform, k=25,select=1):    # sample best uncertain k images and return its index

    val_dataset = WENN_dataset(val_index_list,dataframe,transform=transform)
    val_dataloder = DataLoader(val_dataset, batch_size=4, num_workers=0, drop_last=True)
    softmax = nn.Softmax(dim=1)
    prob_all = torch.tensor([])
    # iter = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloder):
            data, target = data.cuda(), target.cuda() 
            preds = model(data)
            preds = preds.cpu()
            prob = softmax(preds) # return prediction of each sample
            # prob = prob.cpu()
            torch.cuda.empty_cache()
            prob_all = torch.cat((prob_all, prob), 0) # concat prob in a big one
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



    return img_index 