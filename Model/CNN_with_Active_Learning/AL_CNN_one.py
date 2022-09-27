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
############################################################################## Set some basic parameters
classes = ('Dent','Other','Rim','Scratch') # Used to show each classes accuracy

label_frame = pd.read_csv("label0-896_clean.csv") # load csv clean as original dataframe
label_frame_copy = label_frame.copy()
# Set each label from 0-3 for nn.CrossEntropyLoss()  
label_frame_copy['human_label'] = label_frame_copy['human_label'].map(lambda x: x-1)


csv_index_list = np.arange(label_frame_copy.shape[0]) # Split for train and test

Val_size = 600 # Here to change Val_size
val_index_ori = np.random.choice(csv_index_list, size=Val_size, replace=False) # Val is represented by index of csv
# val_label_list = label_list[:600]
test_index_ori = [value for value in csv_index_list if value not in val_index_ori] # Test is chosen except for Val

val_dataframe = label_frame_copy.iloc[val_index_ori,:] # all info about val is saved in this dataframe 
test_dataframe = label_frame_copy.iloc[test_index_ori,:] # Dataframe will not change, but val list will change in AL process

test_index_list = np.arange(test_dataframe.shape[0]) # test_index_list will not change so it is placed outside the algorithm

ori_counts = val_dataframe['label_name'].value_counts()  # count how many samples in each classes 
print('Before oversampling'+'\n',ori_counts)
datanew3 = val_dataframe[val_dataframe['human_label'] == 3]
datanew2 = val_dataframe[val_dataframe['human_label'] == 2]
datanew1 = val_dataframe[val_dataframe['human_label'] == 1]
datanew0 = val_dataframe[val_dataframe['human_label'] == 0]
# keep sample balanced manually
frames = [val_dataframe, datanew2.iloc[:6,:],datanew0.iloc[:20,:],datanew1,datanew1,datanew1.iloc[:41,:],datanew3.iloc[:27,:]]#补前三个
val_dataframe = pd.concat(frames,ignore_index=True)

aft_counts = val_dataframe['label_name'].value_counts()# after oversampling we got balance samples with each classes 200
print('After oversampling'+'\n',aft_counts)


transform = transforms.Compose([transforms.ToPILImage(),#transform will not change
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])

criterion = nn.CrossEntropyLoss().cuda() # set criterion before algorithm loss

# Set classification model and classes
model_name = "resnet"
num_classes = 4
# num_epochs = 2
feature_extract = True        #only change last layer parameter
############################################################################ model function

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False        # freeze original layer and parameters

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)       # use pretrain
        set_parameter_requires_grad(model_ft, feature_extract)      # No more renew parameters
         #model_ft.fc is last layer of resnet，(fc): Linear(in_features=512, out_features=1000, bias=True)，num_ftrs is 512
        num_ftrs = model_ft.fc.in_features               
        model_ft.fc = nn.Linear(num_ftrs, num_classes)   #out_features=1000 changed to num_classes=4

        input_size = 224                 #resnet18 input is 224，also resnet34，50，101，152

    return model_ft, input_size

######################################################### fetch image function changed from Zucheng Han

def imageQuery(imageid):
    #json file should be same path
    with open('annotated_functional_test3_fixed.json','r',encoding='utf-8') as f:   
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
    image_id = objectDict["annotations"][annotation_index]["image_id"]
    json_images = objectDict["images"]
    image_index_injson = imageQuery(image_id)
    curImagepath = json_images[image_index_injson]["file_name"]
    curfullImagepath = "../Images/" + curImagepath
    img = Image.open(curfullImagepath)
    curRegionpos = np.array([curPointpos[0][0], curPointpos[0][1], curPointpos[0][4], curPointpos[0][5]])
    curRegion = img.crop(curRegionpos) #crop image with segmentation
    return curRegion

# def del_rows_index(a,index): #algorithm written for delete line from bottom to top with index of line
#     sorted, indices = torch.sort(index,descending=True)
#     for i in range(len(sorted)):
#         a = a[torch.arange(a.size(0)).cuda()!=sorted[i]]
#     return a

##############################################################################following dataset is ready for dataloader
class ready_for_dataloader(Dataset):  # son of Dataset :)

    def __init__(self, index_list, phase,transform = None):
        super(ready_for_dataloader, self).__init__()
        self.index_list = index_list #  such as val[159,83,25,35,...]
        self.transform = transform
        
        if phase == "test": # test and val have different dataframe
            self.dataframe = test_dataframe

        else:
            self.dataframe = val_dataframe



    def __getitem__(self, index): # set for following dataloader
        csv_index = self.index_list[index] # csv index of val/test dataframe
        annotation_index = self.dataframe.iloc[csv_index,0] #first col is annotation
        annotation_index=annotation_index.astype(int)
        image_id = self.dataframe.iloc[csv_index,1] #second col is image_id
        label = self.dataframe.iloc[csv_index,2] #third col is label
        img = fetchImages(annotation_index) # use fetch function to get img after crop
        
        img = np.array(img) #changed to array for transform later

        if self.transform is not None:
            img = self.transform(img)
        label = label.squeeze() # setting for model

        return img, label



    def __len__(self):
        return len(self.index_list)

############################################################################################ 4 different selection ways
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

############################################################################################ select images from val set to train set
def play_query(val_index_list,model, k=25,select=1):     

    val_dataset = ready_for_dataloader(val_index_list,'val',transform=transform)
    val_dataloder = DataLoader(val_dataset, batch_size=4, num_workers=0, drop_last=True)
    softmax = nn.Softmax(dim=1)
    prob_all = torch.tensor([])

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_dataloder):
            data, target = data.cuda(), target.cuda() #分批上传
            preds = model(data)
            preds = preds.cpu()
            prob = softmax(preds)
            # prob = prob.cpu()
            torch.cuda.empty_cache()
            prob_all = torch.cat((prob_all, prob), 0)

    if select == 1:
        vindices = select1(prob_all)
    elif select == 2:
        vindices = select2(prob_all)
    elif select == 3:
        vindices = select3(prob_all)
    else:
        vindices = select4(prob_all)
    img_index = vindices[:k]#20中选5


    return img_index #返回的是val_list中的index，并不是csv的index

############################################################################################ Val Train and Test process

def val(val_index_list, model): #Val process
    running_loss = 0.
    running_corrects = 0.
    model.eval() # No grad
    val_datalset = ready_for_dataloader(val_index_list,'val',transform=transform)
    val_dataloader = DataLoader(val_datalset, batch_size=4, num_workers=0, drop_last=True) #val dataloader
    for inputs, labels in val_dataloader:
        inputs = inputs.cuda()              
        labels = labels.cuda()
        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs)              
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)                                 #loss have already averaged
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
    epoch_loss = running_loss / len(val_dataloader.dataset)
    epoch_acc = running_corrects / len(val_dataloader.dataset)
    epoch_acc = round(epoch_acc,4) # get accuracy
    print("val_Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc


def test(test_index_list, model): #similar as val but with different name
    
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    running_loss = 0.
    running_corrects = 0.
    model.eval()
    test_datalset = ready_for_dataloader(test_index_list,'test',transform=transform)
    test_dataloader = DataLoader(test_datalset, batch_size=4, num_workers=0, drop_last=True)
    for inputs, labels in test_dataloader:
        inputs = inputs.cuda()               
        labels = labels.cuda()
        with torch.autograd.set_grad_enabled(False):
            outputs = model(inputs)              
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)                                
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
        c = (preds == labels).squeeze()   # get accuracy for each classes learned from pytorch tutorial
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

def train(train_index_list, model,optimizer_ft): # train process
    running_loss = 0.
    running_corrects = 0.
    model.train()
    train_dataset = ready_for_dataloader(train_index_list,'train',transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=0, drop_last=True)
    for inputs, labels in train_dataloader:
        inputs = inputs.cuda()              
        labels = labels.cuda()
        with torch.autograd.set_grad_enabled(True):

            outputs = model(inputs)
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        optimizer_ft.zero_grad()
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item() * inputs.size(0)                                 
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()      # count accuracy
    epoch_loss = running_loss / len(train_dataloader.dataset)
    epoch_acc = running_corrects / len(train_dataloader.dataset)
    epoch_acc = round(epoch_acc,4)
    return epoch_loss, epoch_acc
    
################################################### Big Algorithm involved set model train val test and selection function

def algorithm(test_index_list,k,select = 1,max_queried = 40,limit = 5): 

    ###################################################set model

    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True) #set new model each time
    # print(model_ft)
    model_ft = model_ft.cuda()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []                            #parameters need to be renew
        for name,param in model_ft.named_parameters():   #list all parameters
            #If judgment statement, only true param will be judged before print#param.requires_grad, which is a unique value for each param layer.
            if param.requires_grad == True:              
                params_to_update.append(param)
                #The layer before the fully connected layer here param.requires_grad == Flase, 
                # and the fully connected layer added later param.requires_grad == True           
                # print("\t",name)
    else:                                                #Otherwise, all parameters will be updated
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                pass
                # print("\t",name)
    ###################################################set parameters

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9) #define optimizer

    val_index_list = np.random.choice(np.arange(val_dataframe.shape[0]), size=val_dataframe.shape[0], replace=False) # set val list
    # max_queried = 40          #750 tells how many samples will be loaded to AL process
    # limit = 5                 #cnn training epoch

    queried = k                # how many samples will learned in each AL round
    active_iteration = 0      #count AL rounds
    val_acc_history = []          # remember val acc of each AL round
    val_loss_history = []
    test_acc_history = []
    test_loss_history = []
    train_index_list = []

    best_test_acc = 0.      # save best test_acc for record

    ###################################################algo begins

    while queried < max_queried: # stop until AL rounds are over
        active_iteration += 1
        
        img_index_select = play_query(val_index_list,model_ft,k,select)   # select samples from val set
        img_index_select = img_index_select.tolist()
        img_index_select = map(int,img_index_select)
        img_index_select = list(img_index_select)    # change it to proper type
        

   
        for i in range(len(img_index_select)):
            train_index_list.append(val_index_list[img_index_select[i]])   # get train_list from val
            # train_label_list.append(val_label_list[img_index_select[i]])
        val_index_list = np.delete(val_index_list, img_index_select)
        # val_label_list = np.delete(val_label_list, img_index_select)
    
        print("-"*10)
        print ('Val_set size:',  len(val_index_list))
        print ('Train_set size:',  len(train_index_list))
        print ('Test_set size:',  len(test_index_list))
        
        since = time.time()
        for it in range(limit): # after get train set learn by cnn times by times
            
            epoch_loss, epoch_acc = train(train_index_list, model_ft,optimizer_ft) #train process
            if it%5==0:       # show results every 5 time
                print('Epoch',it)
                print("Train Loss of each epoch: {} Acc: {}".format(epoch_loss, epoch_acc))

        time_elapsed = time.time() - since   #count train time
        print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60)) # _ m _s


        val_epoch_loss, val_epoch_acc = val(val_index_list,  model_ft)  # val process after cnn epoch
        val_acc_history.append(val_epoch_acc)
        val_loss_history.append(val_epoch_loss)
        queried += k  # how many samples taken from val to train
        test_epoch_loss, test_epoch_acc,class_correct,class_total = test(test_index_list, model_ft) # test process after val
        test_acc_history.append(test_epoch_acc)
        test_loss_history.append(test_epoch_loss)
        if test_epoch_acc > best_test_acc: # save best acc model and trainlist after each AL round
            best_test_acc = test_epoch_acc
            best_model_wts = copy.deepcopy(model_ft.state_dict())
            best_train_index_list = train_index_list
            best_class_correct = class_correct
            best_class_total = class_total

    ###################################################algo ends     

    print("-"*10)

    print('Tal_acc',val_acc_history)
    print('Val_loss',val_loss_history)
    print('Test_acc',test_acc_history)
    print('Test_loss',test_loss_history)

    best_train_dataframe = val_dataframe.iloc[best_train_index_list,:] # get best train dataframe
    best_counts = val_dataframe['image_id'].value_counts()
    number = best_train_dataframe['image_id'].drop_duplicates() # how many imgs used originally

    csv_saved_name = 'AL_' + str(k) + '_' + 'accuracy' + '_' + str(best_test_acc)+ '_' + 'images'+ '_' +str(len(number))+'select'+ '_' +str(select)+'.csv'
    
    best_train_dataframe.to_csv(csv_saved_name,index=False,header=True) # save best train set dataframe
    print("best_train_dataframe.csv saved")

    print(len(number),'images were used')
    for i in range(4):
        print('Best Accuracy of %5s : %2d %%' %(classes[i],100*best_class_correct[i]/best_class_total[i]))
    for i in range(len(best_counts.values)):
        count = best_counts.values[i]
        if count > 1:
            print("image_id {} were used {} times".format(best_counts._stat_axis[i], count))


    model_parameter_saved_name = 'AL_' + str(k) + '_' + 'accuracy' + '_' + str(best_test_acc) + 'images'+ '_' +str(len(number))+'select'+ '_' +str(select)+'_parameter.pkl'
    torch.save(best_model_wts, model_parameter_saved_name) # save  best acc model
    print("model_parameter saved")

    txt_name =  'AL_' + str(k) + '_' + 'accuracy' + '_' + str(best_test_acc) + 'images'+ '_' +str(len(number))+'select'+ '_' +str(select)+'.txt'
    f = open(txt_name,'w+') # write some important info in txt

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

    ###################################################set pic and saved automatically

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


###################################################set hyper parameter
k = 20 # how many imgs takes in each AL process
max_queried_AL = 750  #Al taken imgs in 800 val
limit_cnn = 45 # CNN train epoch 
since = time.time()

select_list = [2,3,4,1] # selection methods written in list of course can be written in [1,2]

val_acc_history_all = []
test_acc_history_all = []
val_loss_history_all = []
test_loss_history_all = []

for i in range(len(select_list)): #save history of each selection methods
    active_iteration, val_acc_history, test_acc_history, val_loss_history, test_loss_history = algorithm(test_index_list, k,select = select_list[i],max_queried = max_queried_AL,limit = limit_cnn)
    val_acc_history_all.append(val_acc_history)
    test_acc_history_all.append(test_acc_history)
    val_loss_history_all.append(val_loss_history)
    test_loss_history_all.append(test_loss_history)

time_elapsed = time.time() - since
print("all compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
# all 4 selectio methods finished in about 210 min with GTX1650

###################################################set pic and saved automatically

fig = plt.figure(5)
plt.title("Validation Accuracy vs. Test Accuracy")
plt.xlabel("AL Rounds")
plt.ylabel("Accuracy")
for i in range(len(select_list)):
    plt.plot(range(1,active_iteration+1),val_acc_history_all[i],label="Val_Select "+str(select_list[i]))
    plt.plot(range(1,active_iteration+1),test_acc_history_all[i],label="Test_Select "+str(select_list[i]))
pic_acc_name = 'AL_' + str(k) +'_select_'+str(select_list)+'_acc.png' #remember to change name otherwise it will reload to original pic
plt.ylim((0,1.))
plt.xticks(np.arange(1, active_iteration+1, 1.0))
plt.legend()
plt.savefig(pic_acc_name,bbox_inches='tight')

fig = plt.figure(6)
plt.title("Validation loss vs. Test loss")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")

for i in range(len(select_list)):
    plt.plot(range(1,active_iteration+1),val_loss_history_all[i],label="Val_Select "+str(select_list[i]))
    plt.plot(range(1,active_iteration+1),test_loss_history_all[i],label="Test_Select "+str(select_list[i]))


pic_acc_name = 'AL_' + str(k) +'_select_'+str(select_list)+'_loss.png'
plt.ylim((0,1.5))
plt.xticks(np.arange(1, active_iteration, 1.0))
plt.legend()
plt.savefig(pic_acc_name,bbox_inches='tight')