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
import json
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

ori_counts = val_dataframe['label_name'].value_counts()   # count how many samples in each classes 
print('Before oversampling'+'\n',ori_counts)
datanew3 = val_dataframe[val_dataframe['human_label'] == 3]
datanew2 = val_dataframe[val_dataframe['human_label'] == 2]
datanew1 = val_dataframe[val_dataframe['human_label'] == 1]
datanew0 = val_dataframe[val_dataframe['human_label'] == 0]
frames = [val_dataframe, datanew2.iloc[:6,:],datanew0.iloc[:20,:],datanew1,datanew1,datanew1.iloc[:41,:],datanew3.iloc[:27,:]]# keep sample balanced manually

val_dataframe = pd.concat(frames,ignore_index=True) # after oversampling we got balance samples with each classes 200

aft_counts = val_dataframe['label_name'].value_counts()  # count how many samples in each classes after sampling 
print('After oversampling'+'\n',aft_counts)


transform = transforms.Compose([transforms.ToPILImage(), #transform will not change
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])

criterion = nn.CrossEntropyLoss().cuda() # set criterion before algorithm loss



# Set classification model and classes
model_name = "resnet"
num_classes = 4
feature_extract = True       #only change last layer parameter
######################################################################################################################################### model function

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

######################################################################################################################################### fetch image function changed from Zucheng Han

def imageQuery(imageid):
    with open('annotated_functional_test3_fixed.json','r',encoding='utf-8') as f:   #json file should be same path
     objectDict = json.load(f)   #load json
    lenImages = len(objectDict['annotations'])
    dataDictimages = objectDict['images'] # in file json images there are path of images

    for indexImages in range(0, lenImages): # fetch image path according to image id
        if imageid == dataDictimages[indexImages]["id"]:
            return indexImages
        elif indexImages == lenImages:
            print("Didn't find this annnoted image")

def fetchImages(annotation_index): #fetch image according annotation_index 
    with open('annotated_functional_test3_fixed.json','r',encoding='utf-8') as f:   #json file should be same path
     objectDict = json.load(f)   #load json
    curPointpos = objectDict["annotations"][annotation_index]["segmentation"]
    image_id = objectDict[annotation_index]["image_id"]
    json_images = objectDict["images"]
    image_index_injson = imageQuery(image_id)
    curImagepath = json_images[image_index_injson]["file_name"]
    curfullImagepath = "../Images/" + curImagepath
    img = Image.open(curfullImagepath)
    curRegionpos = np.array([curPointpos[0][0], curPointpos[0][1], curPointpos[0][4], curPointpos[0][5]])
    curRegion = img.crop(curRegionpos) #crop image with segmentation
    return curRegion

######################################################################################################################################### following dataset is ready for dataloader
class ready_for_dataloader(Dataset): 

    def __init__(self, index_list, phase,transform = None):
        super(ready_for_dataloader, self).__init__()
        self.index_list = index_list #val[159,83,25,35]
        # self.labels = labels
        self.img_folder = "../Annotated_images_224/"
        self.transform = transform
        
        if phase == "test":
            self.dataframe = test_dataframe

        else:
            self.dataframe = val_dataframe



    def __getitem__(self, index): 
        # img_name = self.img_folder + self.filenames[index]
        csv_index = self.index_list[index]
        annotation_index = self.dataframe.iloc[csv_index,0]
        annotation_index=annotation_index.astype(int)
        image_id = self.dataframe.iloc[csv_index,1]
        label = self.dataframe.iloc[csv_index,2]

        # with open('annotated_functional_test3_fixed.json','r',encoding='utf-8') as f:   #json file should be same path
        #  objectDict = json.load(f)   #load json
        # image_id = objectDict['annotations'][img_index]['image_id']
        img_name = self.img_folder + str(image_id) + '_' + str(annotation_index) + '.jpg'
        # img_name = self.img_folder + self.filenames[index]
        # label = self.labels[index]

        img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)
        label = label.squeeze()

        return img, label



    def __len__(self):
        return len(self.index_list)

#########################################################################################################################################





train_accur=[]


def del_rows_index(a,index):
    sorted, indices = torch.sort(index,descending=True)
    for i in range(len(sorted)):
        a = a[torch.arange(a.size(0)).cuda()!=sorted[i]]
    return a

def val(val_index_list, model):
    running_loss = 0.
    running_corrects = 0.
    model.eval()
    val_datalset = ready_for_dataloader(val_index_list,'val',transform=transform)
    val_dataloader = DataLoader(val_datalset, batch_size=4, num_workers=0, drop_last=True)
    for inputs, labels in val_dataloader:
        inputs = inputs.cuda()               #inputs.shape = torch.Size([32, 3, 224, 224])
        labels = labels.cuda()
        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs)              #outputs.shape = torch.Size([32, 10])
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)                                
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_acc = running_corrects / len(val_dataloader.dataset)
    epoch_acc = round(epoch_acc,4)

    print("val_Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def test(test_index_list, model):
    
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    running_loss = 0.
    running_corrects = 0.
    model.eval()
    test_datalset = ready_for_dataloader(test_index_list,'test',transform=transform)
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
    print("test_Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    for i in range(4):
        print('Accuracy of %5s : %2d %%' %(classes[i],100*class_correct[i]/class_total[i]))


    
    return epoch_loss, epoch_acc,class_correct,class_total

def train(train_index_list, model,optimizer_ft):
    running_loss = 0.
    running_corrects = 0.
    model.train()
    train_dataset = ready_for_dataloader(train_index_list,'train',transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=0, drop_last=True)
    for inputs, labels in train_dataloader:
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
    # train_acc_history.append(epoch_acc)
    # print("train Loss of each epoch: {} Acc: {}".format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc
    





def algorithm(test_index_list,k,select = 1): 
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

    

    val_index_list = np.random.choice(np.arange(val_dataframe.shape[0]), size=val_dataframe.shape[0], replace=False)
    max_queried = 750
    limit = 45 

    queried = k
    active_iteration = 0
    val_acc_history = []
    val_loss_history = []
    test_acc_history = []
    test_loss_history = []
    train_index_list = []
    # train_label_list = []
    best_test_acc = 0.
    # best_acc = 0.

    while queried < max_queried: 
        active_iteration += 1
        
        img_index_select = play_query(val_index_list,model_ft,k,select)
        img_index_select = img_index_select.tolist()
        img_index_select = map(int,img_index_select)
        img_index_select = list(img_index_select)
        

   
        for i in range(len(img_index_select)):
            train_index_list.append(val_index_list[img_index_select[i]])
            # train_label_list.append(val_label_list[img_index_select[i]])
        val_index_list = np.delete(val_index_list, img_index_select)
        # val_label_list = np.delete(val_label_list, img_index_select)


    
        print("-"*10)
        print ('val_set size:',  len(val_index_list))
        print ('train_set size:',  len(train_index_list))
        print ('test_set size:',  len(test_index_list))

        

        since = time.time()
        for it in range(limit):
            

            epoch_loss, epoch_acc = train(train_index_list, model_ft,optimizer_ft)
            if it%5==0:
                print('Epoch',it)
                print("train Loss of each epoch: {} Acc: {}".format(epoch_loss, epoch_acc))

        time_elapsed = time.time() - since
        print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))


        val_epoch_loss, val_epoch_acc = val(val_index_list,  model_ft)
        val_acc_history.append(val_epoch_acc)
        val_loss_history.append(val_epoch_loss)
        queried += k
        test_epoch_loss, test_epoch_acc,class_correct,class_total = test(test_index_list, model_ft)
        test_acc_history.append(test_epoch_acc)
        test_loss_history.append(test_epoch_loss)
        if test_epoch_acc > best_test_acc:
            best_test_acc = test_epoch_acc
            best_model_wts = copy.deepcopy(model_ft.state_dict())
            best_train_index_list = train_index_list
            best_class_correct = class_correct
            best_class_total = class_total
      
    print("-"*10)

    print('val_acc',val_acc_history)
    print('val_loss',val_loss_history)
    print('test_acc',test_acc_history)
    print('test_loss',test_loss_history)
    best_train_dataframe = val_dataframe.iloc[best_train_index_list,:]
    best_counts = val_dataframe['image_id'].value_counts()
    number = best_train_dataframe['image_id'].drop_duplicates()
    csv_saved_name = 'AL_' + str(k) + '_' + 'accuracy' + '_' + str(best_test_acc)+ '_' + 'images'+ '_' +str(len(number))+'select'+ '_' +str(select)+'.csv'
    
    best_train_dataframe.to_csv(csv_saved_name,index=False,header=True)
    print("best_train_dataframe.csv saved")

    print(len(number),'images were used')
    for i in range(4):
        print('Best Accuracy of %5s : %2d %%' %(classes[i],100*best_class_correct[i]/best_class_total[i]))
    for i in range(len(best_counts.values)):
        count = best_counts.values[i]
        if count > 1:
            print("image_id {} were used {} times".format(best_counts._stat_axis[i], count))
    # print('used best train samples\n',best_counts)

    model_parameter_saved_name = 'AL_' + str(k) + '_' + 'accuracy' + '_' + str(best_test_acc) + 'images'+ '_' +str(len(number))+'select'+ '_' +str(select)+'_parameter.pkl'
    torch.save(best_model_wts, model_parameter_saved_name)
    print("model_parameter saved")

    txt_name =  'AL_' + str(k) + '_' + 'accuracy' + '_' + str(best_test_acc) + 'images'+ '_' +str(len(number))+'select'+ '_' +str(select)+'.txt'
    f = open(txt_name,'w+')
    f.write('Val_acc_history'+' : '+str(val_acc_history)+'\n')
    f.write('Val_loss_history'+' : '+str(val_loss_history)+'\n')
    f.write('Test_acc_history'+' : '+str(test_acc_history)+'\n')
    f.write('Test_loss_history'+' : '+str(test_loss_history)+'\n')
    f.write(str(len(number))+' images were used'+'\n')
    for i in range(4):
        f.write('Best Accuracy of '+str(classes[i])+' : '+ str(100*best_class_correct[i]/best_class_total[i])+'\n')
    for i in range(len(best_counts.values)):
        count = best_counts.values[i]
        if count > 1:
            f.write('image_id '+str(best_counts._stat_axis[i])+ ' were used '+ str(count) +' times'+'\n')
    f.close()


    fig = plt.figure(select)
    plt.title("Validation Accuracy vs. Test Accuracy")
    plt.xlabel("AL Rounds")
    plt.ylabel("Accuracy")
    plt.plot(range(1,active_iteration+1),val_acc_history,label="Val_"+str(select))
    plt.plot(range(1,active_iteration+1),test_acc_history,label="Test_"+str(select))
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, active_iteration+1, 1.0))
    plt.legend()
    pic_acc_name = 'AL_' + str(k) +'_select_'+str(select)+'accuracy' + '_' + str(best_test_acc) +'_acc.png'
    plt.savefig(pic_acc_name,bbox_inches='tight')

    fig = plt.figure(select+10)
    plt.title("Validation loss vs. Test loss")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,active_iteration+1),val_loss_history,label="Val_"+str(select))
    plt.plot(range(1,active_iteration+1),test_loss_history,label="test_"+str(select))
    plt.ylim((0,1.5))
    plt.xticks(np.arange(1, active_iteration, 1.0))
    plt.legend()
    pic_acc_name = 'AL_' + str(k) +'_select_'+str(select) +'accuracy' + '_' + str(best_test_acc) +'_loss.png'
    plt.savefig(pic_acc_name,bbox_inches='tight')
    return active_iteration,val_acc_history,test_acc_history,val_loss_history,test_loss_history


def select1(probas_val): 

    sorted, indices = torch.sort(probas_val,descending=True) 
    values = sorted[:, 0] - sorted[:, 1] #
    vsorted, vindices = torch.sort(values) 

    return vindices

def select2(probas_val): 
    stddd = torch.std(probas_val,dim=1,unbiased=False)

    vsorted, vindices = torch.sort(stddd) 

    return vindices

# torch.log2(a)
def select3(probas_val): 
    entropy = torch.sum(torch.mul(-probas_val, torch.log2(probas_val)),dim=1)

    vsorted, vindices = torch.sort(entropy,descending=True) 

    return vindices

def select4(probas_val): 
    vindices = torch.randperm(len(probas_val))

    
    return vindices


def play_query(val_index_list,model, k=25,select=1):    

    val_dataset = ready_for_dataloader(val_index_list,'val',transform=transform)
    val_dataloder = DataLoader(val_dataset, batch_size=4, num_workers=0, drop_last=True)
    softmax = nn.Softmax(dim=1)
    prob_all = torch.tensor([])
    # iter = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloder):
            data, target = data.cuda(), target.cuda() 
            preds = model(data)
            preds = preds.cpu()
            prob = softmax(preds)
            # prob = prob.cpu()
            torch.cuda.empty_cache()
            prob_all = torch.cat((prob_all, prob), 0)
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

k = 20
since = time.time()
select_list = [2,3,4,1]
val_acc_history_all = []
test_acc_history_all = []
val_loss_history_all = []
test_loss_history_all = []
for i in range(len(select_list)):
    active_iteration, val_acc_history, test_acc_history, val_loss_history, test_loss_history = algorithm(test_index_list, k,select = select_list[i])
    val_acc_history_all.append(val_acc_history)
    test_acc_history_all.append(test_acc_history)
    val_loss_history_all.append(val_loss_history)
    test_loss_history_all.append(test_loss_history)

time_elapsed = time.time() - since
print("all compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))


fig = plt.figure(5)
plt.title("Validation Accuracy vs. Test Accuracy")
plt.xlabel("AL Rounds")
plt.ylabel("Accuracy")
for i in range(len(select_list)):
    plt.plot(range(1,active_iteration+1),val_acc_history_all[i],label="Val_"+str(select_list[i]))
    plt.plot(range(1,active_iteration+1),test_acc_history_all[i],label="Test_"+str(select_list[i]))
    
    #'AL_' + str(k) +'_select_'

pic_acc_name = 'AL_' + str(k) +'_select_'+str(select_list)+'_acc.png'
plt.ylim((0,1.))
plt.xticks(np.arange(1, active_iteration+1, 1.0))
plt.legend()
plt.savefig(pic_acc_name,bbox_inches='tight')

fig = plt.figure(6)
plt.title("Validation loss vs. Test loss")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")

for i in range(len(select_list)):
    plt.plot(range(1,active_iteration+1),val_loss_history_all[i],label="Val_"+str(select_list[i]))
    plt.plot(range(1,active_iteration+1),test_loss_history_all[i],label="Test_"+str(select_list[i]))


pic_acc_name = 'AL_' + str(k) +'_select_'+str(select_list)+'_loss.png'
plt.ylim((0,1.5))
plt.xticks(np.arange(1, active_iteration, 1.0))
plt.legend()
plt.savefig(pic_acc_name,bbox_inches='tight')