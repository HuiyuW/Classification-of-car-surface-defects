import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from PIL import Image
from src.model_parameter import initialize_model
from src.Upload_dataset import Upload_dataset


def main(path):

    label_dict = {'0': 'Dent', '1': 'Other', '2': 'Rim', '3': 'Scratch'} # change label num to words
    torch.cuda.set_device(0) # Set gpu number here
    torch.cuda.manual_seed(0) # keep split all in same way
    np.random.seed(0) # Keep val_list controlable



    model_name = "resnet"
    num_classes = 4
    feature_extract = True
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # print(model_ft)
    model_ft = model_ft.cuda()
    if feature_extract:
        params_to_update = []                            
        for name,param in model_ft.named_parameters():   
            if param.requires_grad == True:              
                params_to_update.append(param)           
                # print("\t",name)
    else:                                               
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                pass
                    # print("\t",name)
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    model_PATH = './results/AL_15_accuracy_0.8174images_375select_3_parameter.pkl' # load pretrain model parameters
    model_ft.load_state_dict(torch.load(model_PATH))

    model_ft.eval()
    prob_all = torch.tensor([])



    transform = transforms.Compose([transforms.ToPILImage(), #transform will not change
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])
    softmax = nn.Softmax(dim=1)




    up_dataset = Upload_dataset(path,transform=transform) # get imgs from path of uploaded folder
    up_dataloader = DataLoader(up_dataset, batch_size=2, num_workers=0, drop_last=True)
    test_predict_dataframe = pd.DataFrame()
    for inputs,img_path in up_dataloader:
        inputs = inputs.cuda()               #inputs.shape = torch.Size([32, 3, 224, 224])
        # labels = labels.cuda()
        with torch.autograd.set_grad_enabled(False):
            outputs = model_ft(inputs) 
            outputs = softmax(outputs)
            outputs =  outputs.cpu()   
            prob_all = torch.cat((prob_all, outputs), 0)        #outputs.shape = torch.Size([32, 10])
            sorted, indices = torch.sort(outputs,descending=True)
            prediction = indices[:,0].numpy()
            new_dataframe = pd.DataFrame({'img_path':img_path,'prediction':prediction}) # each iter of dataloader save img and img path in csv
            test_predict_dataframe = pd.concat([test_predict_dataframe, new_dataframe],axis=0)
    for i in range(5): # here only show 5 imgs of course you can choose how many img predictions you wanna show
        img_path = test_predict_dataframe.iloc[i,0]
        img = Image.open(img_path)
        predict_label = test_predict_dataframe.iloc[i,1]
        predict_label_name = label_dict[str(predict_label)]
        print("Our model predicts it is a {}".format(predict_label_name))
        plt.imshow(img)
        plt.show()



if __name__ == '__main__':
    path = 'C:/Users/wanghuiyu/Desktop/AMI dataset/Data/Annotated_images_test/' # upload imgs folder from web users, remember put some imgs here for test
    # path = './Annotated_images_test/' #or just use my test folder
    main(path)