from hashlib import new
import string
import numpy as np
import torch
import torch.nn as nn
import time
import copy
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from torch.utils.data import random_split
from .src.model_parameter import initialize_model
from .src.create_train import create_train
from .src.Combine_dataset import combine_dataset
from .src.WENN_dataset import WENN_dataset
from .src.creterion import ConfusionMatrix
from .src.Upload_dataset import Upload_dataset
from .src.fetch_image import dataPreprocess
from .main_select import main_select
from PIL import Image
from matplotlib import pyplot as plt


def main_train(path_dataset,num,select,label_list,img_path_list):
    #torch.cuda.set_device(0) # Set gpu number here
    #torch.cuda.manual_seed(0)

    #torch.manual_seed(0)
    np.random.seed(0) # Keep val_list controlable

    #dp = dataPreprocess(path_dataset) #create folder with all 897 annotated images in size 224*224
    #dp.fetchImages()
    
    label_dict = {'0': 'Dent', '1': 'Other', '2': 'Rim', '3': 'Scratch'}
    classes = ('Dent','Other','Rim','Scratch')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    label_frame = pd.read_csv('imgProcess/Resnet_ALWeb/Labels/label0-896_clean.csv') # load csv clean as original dataframe
    label_frame_copy = label_frame.copy()
    label_frame_copy['human_label'] = label_frame_copy['human_label'].map(lambda x: x-1)
    dataframe = label_frame_copy

    train_df = pd.read_csv('imgProcess/Resnet_ALWeb/results/AL_15_accuracy_0.8174_images_375select_3.csv') # read train csv from pretrain model

    create_train(train_df) # create train set folder


    num_classes=4
    model_name = "resnet"
    feature_extract = True   
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    #model_ft = model_ft.cuda()
    model_PATH = 'imgProcess/Resnet_ALWeb/results/AL_5_select_1accuracy_0.8672_parameter.pkl'
    model_ft.load_state_dict(torch.load(model_PATH, map_location=torch.device('cpu'))) # load pretrained model

    # print("Params to learn:")
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

    # dp = dataPreprocess(path)
    # dp.fetchImages()
    # label_frame = pd.read_csv('Labels\label0-896_clean.csv') # load csv clean as original dataframe
    # label_frame_copy = label_frame.copy()
    # label_frame_copy['human_label'] = label_frame_copy['human_label'].map(lambda x: x-1)
    # dataframe = label_frame_copy

    test_split = 0.3
    test_size = int(test_split * dataframe.shape[0])
    val_size = dataframe.shape[0] - test_size
    val_index_list, test_index_list = random_split(np.arange(dataframe.shape[0]),[val_size, test_size])# get test_index_list

    transform = transforms.Compose([transforms.ToPILImage(), #transform will not change
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])


    
    label_name = []
    for i in range(num):
        label_name.append(label_dict[str(label_list[i])]) # get manual lable list and change to words

    #print("-------------label list and label name---------------------")
    #print(label_list)
    #print(label_name)
    fake_id = [0]*num    # set id to 0 for identification
    up_data_df = pd.DataFrame({'annotation_index':img_path_list, 'image_id':fake_id, 'human_label':label_list, 'label_name' : label_name}) #form dataset of uploaded uncertrain img set
    
    combine_train_df = pd.concat([train_df, up_data_df],axis=0) # conbine  original train set and uploaded set
    limit = 20 # use new dataset to train model and get new performance
    since = time.time()
    print("-"*10)

    for it in range(limit):
##################################################################################Train
        running_loss = 0.
        running_corrects = 0.
        model_ft.train()
        train_dataset = combine_dataset(combine_train_df,transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=combine_train_df.shape[0], num_workers=0, drop_last=True)

        dict = train_dataset.count_classes() # get classes distribution
        class_sample_counts = [dict[0], dict[1], dict[2], dict[3]]
        weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights) # add train classes weights to criterion

        for inputs, labels in train_dataloader:
            labels = labels.type(torch.LongTensor) #"nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Double'
            #inputs = inputs.cuda()              
            #labels = labels.cuda()
            with torch.autograd.set_grad_enabled(True):

                outputs = model_ft(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item() * inputs.size(0)                                 
            running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()     
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects / len(train_dataloader.dataset)
        epoch_acc = round(epoch_acc,4)
        if it%5==0:
            print('Epoch',it)
            print("train Loss of each epoch: {} Acc: {}".format(epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("-"*10)
##################################################################################Test
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    running_loss = 0.
    running_corrects = 0.
    confusion = ConfusionMatrix(num_classes=4) # Matrix to get TN TF PN PF
    model_ft.eval()
    test_dataset = WENN_dataset(test_index_list,dataframe,transform=transform) # get test set from Annotated images 224
    test_dataloader = DataLoader(test_dataset, batch_size=4, num_workers=0, drop_last=True)
    for inputs, labels in test_dataloader:
        #inputs = inputs.cuda()               
        #labels = labels.cuda()
        with torch.autograd.set_grad_enabled(False):
            outputs = model_ft(inputs)              
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        ret, predictions = torch.max(outputs.data, 1) # prediction here is for ConfusionMatrix
        confusion.update(predictions.cpu().numpy(), labels.cpu().numpy()) # update CondusionMatrix after each dataloader iter
        running_loss += loss.item() * inputs.size(0)                                
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
        c = (preds == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()  # count accuracy of each classes
            class_total[label] += 1
    acc,table, table_array = confusion.summary()  # get Precision Recall Specificity and F1 from table
    
    test_epoch_loss = running_loss / len(test_dataloader.dataset)
    test_epoch_acc = running_corrects / len(test_dataloader.dataset)
    test_epoch_acc = round(test_epoch_acc,4)
    print("Test_Loss: {} Acc: {}".format(test_epoch_loss, test_epoch_acc))
    acc_list = []
    for i in range(4):
        print('Accuracy of %5s : %2d %%' %(classes[i],100*class_correct[i]/class_total[i]))
        acc_list.append(round(class_correct[i]/class_total[i],2))
    print("the model accuracy is ", acc)
    acc_list.append(acc)
    print(table)


    if test_epoch_acc > 0.8672:
        print("After learning new imgs the accuracy of our model improved from 0.8672 to {}.".format(test_epoch_acc))
    else:
        print("After learning new imgs the accuracy of our model changed from 0.8672 to {}.".format(test_epoch_acc))

    confusion.plot()
    print(combine_train_df.shape[0],'images were used')
    csv_saved_name = 'imgProcess/Resnet_ALWeb/results/new_train_AL.csv'
    combine_train_df.to_csv(csv_saved_name,index=False,header=True)
    model_parameter_saved_name = 'imgProcess/Resnet_ALWeb/results/new_train_AL.pkl'
    model_wts = copy.deepcopy(model_ft.state_dict())
    torch.save(model_wts, model_parameter_saved_name)
    print("model_parameter saved")

    return table_array, acc_list



# use it help us to directly to predict single picture with new model
def newresnet18Predict(img):
    label_dict = {'0': 'Dent', '1': 'Other', '2': 'Rim', '3': 'Scratch'} # change label num to words
    np.random.seed(0) # Keep val_list controlable

    model_name = "resnet"
    num_classes = 4
    feature_extract = True
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # print(model_ft)
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
    model_PATH = 'imgProcess/Resnet_ALWeb/results/new_train_AL.pkl' # load pretrain model parameters
    model_ft.load_state_dict(torch.load(model_PATH, map_location=torch.device('cpu')))

    model_ft.eval()
    prob_all = torch.tensor([])

    transform = transforms.Compose([transforms.ToPILImage(), #transform will not change
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])
    softmax = nn.Softmax(dim=1) # why softmax

    image_tensor = transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    outputs = model_ft(image_tensor)

    outputs = softmax(outputs)

    outputs =  outputs.cpu()   

    _, indices = torch.sort(outputs, descending=True)

    #print(indices)

    index = indices[0][0].numpy()

    #print(index)
    if index == 0:
        prediction = "Dent"
    elif index == 1:
        prediction = "Other"
    elif index == 2:
        prediction = "Rim"
    elif index == 3:
        prediction = "Scratch"

    return prediction


# use it help us to directly to predict pictures
def resnet18Predict(img):
    label_dict = {'0': 'Dent', '1': 'Other', '2': 'Rim', '3': 'Scratch'} # change label num to words
    np.random.seed(0) # Keep val_list controlable

    model_name = "resnet"
    num_classes = 4
    feature_extract = True
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # print(model_ft)
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
    model_PATH = 'imgProcess/Resnet_ALWeb/results/AL_5_select_1accuracy_0.8672_parameter.pkl' # load pretrain model parameters
    model_ft.load_state_dict(torch.load(model_PATH, map_location=torch.device('cpu')))

    model_ft.eval()
    prob_all = torch.tensor([])

    transform = transforms.Compose([transforms.ToPILImage(), #transform will not change
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])
    softmax = nn.Softmax(dim=1) # why softmax

    image_tensor = transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    outputs = model_ft(image_tensor)

    outputs = softmax(outputs)

    outputs =  outputs.cpu()   

    _, indices = torch.sort(outputs, descending=True)

    #print(indices)

    index = indices[0][0].numpy()

    print(index)
    if index == 0:
        prediction = "Dent"
    elif index == 1:
        prediction = "Other"
    elif index == 2:
        prediction = "Rim"
    elif index == 3:
        prediction = "Scratch"

    return prediction



if __name__ == '__main__':
    path = './test_imgaes/' # user upload img folder path
    path_dataset = './Annotated_images/' # WENN dataset Data path
    num = 5
    select = 1
    img_path_list = main_select(path,num,select)# as long as you get img_path_list then you can train the model
    
    for i in range(num): # here only show 5 imgs of course you can choose how many img predictions you wanna show
        img_path = img_path_list[i]
        img = Image.open(img_path)
        plt.imshow(img)
        plt.show()
    
    label_list = [0,0,3,0,3] #Suppose you have obtained labels for five images manually annotated by the user
    main_train(path_dataset,num,select,label_list,img_path_list)