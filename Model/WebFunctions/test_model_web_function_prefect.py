import numpy as np
import torch
import torch.nn as nn
import time
from copy import deepcopy
import copy
from matplotlib import pyplot as plt
from torchvision import transforms, models
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import pandas as pd
from PIL import Image
import os




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
test_index_list = test_index_list[50:]

test_sample_folder = '../Annotated_images_test/'
imagelist = os.listdir(test_sample_folder)#读取images文件夹下所有文件的名字
train_df = pd.read_csv("AL_20_accuracy_0.8293_images_461select_2.csv")
label_dict = {'0': 'Dent', '1': 'Other', '2': 'Rim', '3': 'Scratch'}

transform = transforms.Compose([transforms.ToPILImage(), #transform will not change
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])

criterion = nn.CrossEntropyLoss().cuda() # set criterion before algorithm loss
softmax = nn.Softmax(dim=1)

##################################################################################set model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False        # freeze original layer and parameters

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)       # use pretrain
        set_parameter_requires_grad(model_ft, feature_extract)      # No more renew parameters

        num_ftrs = model_ft.fc.in_features               #model_ft.fc is last layer of resnet，(fc): Linear(in_features=512, out_features=1000, bias=True)，num_ftrs is 512
        model_ft.fc = nn.Linear(num_ftrs, num_classes)   #out_features=1000 changed to num_classes=4

        input_size = 224                                 #resnet18 input is 224，also resnet34，50，101，152

    return model_ft, input_size
##################################################################################set model

model_name = "resnet"
num_classes = 4
feature_extract = True
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# print(model_ft)
model_ft = model_ft.cuda()

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
PATH = 'AL_20_accuracy_0.8293images_461select_2_parameter.pkl'
model_ft.load_state_dict(torch.load(PATH))

##################################################################################select function

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

##################################################################################dataset from new data

class new_dataset(Dataset): #crete dataloader for new folder and new img

    def __init__(self, imagelist,transform = None):
        super(new_dataset, self).__init__()
        self.imagelist = imagelist #val[159,83,25,35]
        self.transform = transform

    def __getitem__(self, index): 
        imgname = self.imagelist[index]
        img_path = test_sample_folder + str(imgname)


        img = plt.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.imagelist)

##################################################################################predict new data and save dataframe

def show_predict(test_index_list, model): 
    
    model.eval()
    prob_all = torch.tensor([])
    test_datalset = new_dataset(test_index_list,transform=transform)
    test_dataloader = DataLoader(test_datalset, batch_size=2, num_workers=0, drop_last=True)
    test_predict_dataframe = pd.DataFrame()
    for inputs,img_path in test_dataloader:
        inputs = inputs.cuda()               #inputs.shape = torch.Size([32, 3, 224, 224])
        # labels = labels.cuda()
        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs) 
            outputs = softmax(outputs)
            outputs =  outputs.cpu()   
            prob_all = torch.cat((prob_all, outputs), 0)        #outputs.shape = torch.Size([32, 10])
            sorted, indices = torch.sort(outputs,descending=True)
            prediction = indices[:,0].numpy()
            new_dataframe = pd.DataFrame({'img_path':img_path,'prediction':prediction})
            test_predict_dataframe = pd.concat([test_predict_dataframe, new_dataframe],axis=0)

    return test_predict_dataframe

##################################################################################use al methods to select img

def select_img(select, test_index_list, model): 
    
    model.eval()
    prob_all = torch.tensor([])
    test_datalset = new_dataset(test_index_list,transform=transform)
    test_dataloader = DataLoader(test_datalset, batch_size=2, num_workers=0, drop_last=True)
    for inputs,img_path in test_dataloader:
        inputs = inputs.cuda()               #
        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs) 
            outputs = softmax(outputs)
            outputs =  outputs.cpu()   
            prob_all = torch.cat((prob_all, outputs), 0)        #
    if select == 1:
        vindices = select1(prob_all)
    elif select == 2:
        vindices = select2(prob_all)
    elif select == 3:
        vindices = select3(prob_all)
    else:
        vindices = select4(prob_all)
    return vindices #return select img index of probability matrix from model output

##################################################################################get dataset from ori_train and new select

class combine_dataset(Dataset): 

    def __init__(self, dataframe,transform = None):
        super(combine_dataset, self).__init__()
        self.dataframe = dataframe #val[159,83,25,35]
        self.img_folder = "../Annotated_images_224/"
        self.transform = transform

    def __getitem__(self, index): 
        image_id = self.dataframe.iloc[index,1]
        if image_id == 0:
            img_path = self.dataframe.iloc[index,0]
            img = plt.imread(img_path)
            label = self.dataframe.iloc[index,2]
        else:
            annotation_index = self.dataframe.iloc[index,0]
            image_id = self.dataframe.iloc[index,1]
            label = self.dataframe.iloc[index,2]

            img_name = self.img_folder + str(image_id) + '_' + str(annotation_index) + '.jpg'
            img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)
        label = label.squeeze()

        return img, label

    def __len__(self):
        return self.dataframe.shape[0]

##################################################################################train combine data

def train(dataframe, model,optimizer_ft): #train for new dataset
    running_loss = 0.
    running_corrects = 0.
    model.train()
    train_dataset = combine_dataset(dataframe,transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=0, drop_last=True)
    for inputs, labels in train_dataloader:
        labels = labels.type(torch.LongTensor) #"nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Double'
        inputs = inputs.cuda()              #inputs.shape = torch.Size([32, 3, 224, 224])
        labels = labels.cuda()
        with torch.autograd.set_grad_enabled(True):

            outputs = model(inputs)
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

    return epoch_loss, epoch_acc

##################################################################################dataset for test data

class test_dataset(Dataset): 

    def __init__(self, index_list, transform = None):
        super(test_dataset, self).__init__()
        self.index_list = index_list #val[159,83,25,35]
        self.img_folder = "../Annotated_images_224/"
        self.transform = transform
        self.dataframe = test_dataframe

    def __getitem__(self, index): 
        csv_index = self.index_list[index]
        annotation_index = self.dataframe.iloc[csv_index,0]
        annotation_index=annotation_index.astype(int)
        image_id = self.dataframe.iloc[csv_index,1]
        label = self.dataframe.iloc[csv_index,2]
        img_name = self.img_folder + str(image_id) + '_' + str(annotation_index) + '.jpg'

        img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)
        label = label.squeeze()

        return img, label

    def __len__(self):
        return len(self.index_list)

##################################################################################test combine data

def test(test_index_list, model):
    
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    running_loss = 0.
    running_corrects = 0.
    model.eval()
    test_datalset = test_dataset(test_index_list,transform=transform)
    test_dataloader = DataLoader(test_datalset, batch_size=4, num_workers=0, drop_last=True)
    for inputs, labels in test_dataloader:
        inputs = inputs.cuda()               #inputs.shape = torch.Size([32, 3, 224, 224])
        labels = labels.cuda()
        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs)              #outputs.shape = torch.Size([32, 10])
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)                                
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
        c = (preds == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    epoch_loss = running_loss / len(test_dataloader.dataset)
    epoch_acc = running_corrects / len(test_dataloader.dataset)
    epoch_acc = round(epoch_acc,4)
    print("Test_Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    for i in range(4):
        print('Accuracy of %5s : %2d %%' %(classes[i],100*class_correct[i]/class_total[i]))
    return epoch_loss, epoch_acc

##################################################################################final visual

def algorithm(num, select):
    vindices = select_img(select, imagelist, model_ft)

    img_path_list = []
    label_name = []
    human_label = []
    fake_id = [0]*num

    for i in range(num):
        img_path = test_predict_dataframe.iloc[vindices.numpy()[i],0]
        predict_label = test_predict_dataframe.iloc[vindices.numpy()[i],1]
        predict_label_name = label_dict[str(predict_label)]
        img = Image.open(img_path)
        print("-"*10)
        print('label 0 -> Dent \nlabel 1 -> Other \nlabel 2 -> Rim \nlabel 3 -> Scratch.')
        print("Image path is {}".format(img_path))
        print("Our model predicts it is a {}".format(predict_label_name))
        plt.imshow(img)
        plt.show()
        label = int(input("What do u think expert? write label number here ："))     #type in label number
        human_label.append(label)
        label_name.append(label_dict[str(label)])
        img_path_list.append(img_path)

    select_data_df = pd.DataFrame({'annotation_index':img_path_list, 'image_id':fake_id, 'human_label':human_label, 'label_name' : label_name}) #AL选出图片的csv

    new_train_df = pd.concat([train_df, select_data_df],axis=0)

    limit = 25 
    since = time.time()
    print("-"*10)

    for it in range(limit):
        epoch_loss, epoch_acc = train(new_train_df, model_ft,optimizer_ft)
        if it%5==0:
            print('Epoch',it)
            print("train Loss of each epoch: {} Acc: {}".format(epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("-"*10)
    test_epoch_loss, test_epoch_acc = test(test_index_list, model_ft)

    if test_epoch_acc > 0.8293:
        print("After learning new imgs the accuracy of our model improved from 0.8293 to {}.".format(test_epoch_acc))
    else:
        print("After learning new imgs the accuracy of our model changed from 0.8293 to {}.".format(test_epoch_acc))

    fake_number = new_train_df['image_id'].drop_duplicates()
    number = fake_number-1+num
    print(len(number),'images were used')
    csv_saved_name = 'new_train_AL_' + str(num) + '_' +'select'+ '_' +str(select) + 'accuracy' + '_' + str(test_epoch_acc)+ '_' + '.csv'
    new_train_df.to_csv(csv_saved_name,index=False,header=True)
    model_parameter_saved_name = 'new_train_AL_' + str(num) + '_' +'select'+ '_' +str(select)+ 'accuracy' + '_' + str(test_epoch_acc)+ '_parameter.pkl'
    model_wts = copy.deepcopy(model_ft.state_dict())
    torch.save(model_wts, model_parameter_saved_name)
    print("model_parameter saved")

    return test_epoch_acc

##################################################################################get select data and df

test_predict_dataframe = show_predict(imagelist, model_ft) 
print('Have a see of our prediction')
for i in range(5):
    img_path = test_predict_dataframe.iloc[i,0]
    img = Image.open(img_path)
    predict_label = test_predict_dataframe.iloc[i,1]
    predict_label_name = label_dict[str(predict_label)]
    print("Our model predicts it is a {}".format(predict_label_name))
    plt.imshow(img)
    plt.show()

print("u have {} new images in total".format(test_predict_dataframe.shape[0]))
num = int(input("how many images do u want to select with AL ："))     #type in label number

##################################################################################algo

select_list = [2,3,4,1]
test_epoch_acc_list = []
since = time.time()
for i in range(len(select_list)):
    test_epoch_acc = algorithm(num, select = select_list[i])
    test_epoch_acc_list.append(test_epoch_acc)
time_elapsed = time.time() - since
print("-"*10)
for i in range(len(select_list)):
    if test_epoch_acc_list[i] > 0.8293:
        print("After learning new imgs with Al selection {}, the accuracy of our model improved from 0.8293 to {}.".format(select_list[i], test_epoch_acc_list[i]))
    else:
        print("After learning new imgs with Al selection {}, the accuracy of our model changed from 0.8293 to {}.".format(select_list[i], test_epoch_acc_list[i]))
print("All select methods compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))