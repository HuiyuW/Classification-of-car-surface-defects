import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import copy
import time
import os
import operator
import functools
import torch

# torchvision for pre-trained models


class CIFAR10_IMG(Dataset):

    def __init__(self, root, num=1, transform = None, target_transform=None):
        super(CIFAR10_IMG, self).__init__()
        self.num = num
        self.transform = transform
        self.target_transform = target_transform

        if self.num==1 :
            file_annotation = root + '/sample_set_new/train.txt'
            # img_folder = root + '/train_cifar10/'
        elif self.num==2:
            file_annotation = root + '/sample_set_new/val.txt'
        
            # img_folder = root + '/test_cifar10/'
        else:
            file_annotation = root + '/sample_set_new/test.txt'
        # fp = open(file_annotation,'r')
        list_load = np.loadtxt(file_annotation)


        self.filenames = []
        self.labels = []
        self.img_folder = "../Annotated_images_224/"

        label_frame = pd.read_csv("label0-896_clean.csv")
        label_frame_copy = label_frame.copy()


        for i in range(len(list_load)):
            # self.filenames.append(list_load[i])
            A_index = label_frame_copy['annotation_index'][list_load[i]]
            self.filenames.append(A_index)
            label_list_type = label_frame_copy[(label_frame_copy['annotation_index']==A_index)]['human_label'].values
            label_list_type = label_list_type - 1
            self.labels.append(label_list_type)

    def __getitem__(self, index): #搭配dataloader自动取出使用
        # img_name = self.img_folder + self.filenames[index]
        img_index = self.filenames[index]
        
        img_index=img_index.astype(int)

        with open('annotated_functional_test3_fixed.json','r',encoding='utf-8') as f:   #json file should be same path
         objectDict = json.load(f)   #load json
        image_id = objectDict['annotations'][img_index]['image_id']
        img_name = self.img_folder + str(image_id) + '_' + str(img_index) + '.jpg'
        # img_name = self.img_folder + self.filenames[index]
        label = self.labels[index]

        img = plt.imread(img_name)
        if self.transform is not None:
            img = self.transform(img)
        label = label.squeeze()

        return img, label

    def __len__(self):
        return len(self.filenames)






root_path=os.getcwd()
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CIFAR10_IMG(root_path,num=1,transform=transform)
val_dataset = CIFAR10_IMG(root_path,num=2,transform=transform)
test_dataset = CIFAR10_IMG(root_path,num=3,transform=transform)


batch_size=16
image_datasets={'train':train_dataset,'val':val_dataset,'test':test_dataset}
dataloaders_dict = {x:DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val','test']}

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)



def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))




class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(8 * 8 * 8, 4)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def test(model, device, test_loader):
    
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += F.nll_loss(outputs, target, reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))         #target.shape=torch.Size([32])
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
    
    
def validate(model, device, val_loader):
    correct =0

    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            val_loss += F.nll_loss(outputs, target, reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    val_loss /= len(test_loader.dataset)
    print('Accuracy of the network on the val images: %d %%' % (
    100 * correct / total))         #target.shape=torch.Size([32])
    print('\nval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    val_loss, correct, len(val_loader.dataset),
    100. * correct / len(val_loader.dataset)))

def train(model, device, train_loader, val_loader, optimizer, epoch, log_interval=10):
    # model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # in case you wanted a semi-full example
        outputs = model(data)              
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # _, predicted = torch.max(outputs.data, 1)
        # total += target.size(0)
        # correct += (predicted == target).sum().item()
        
        if batch_idx % log_interval == 9: # print every 10 mini-batches
             print('[%d, %5d] loss: %.3f' %
             (epoch + 1, batch_idx + 1, running_loss / 10))
             validate(model, device, val_loader)
    #          print('Accuracy of the network on the 10000 test images: %d %%' % (
    # 10 * correct / total))
            #  loss_process.append(running_loss/10)
             running_loss = 0.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_name = "resnet"
num_classes = 4
num_epochs = 10
feature_extract = True      

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False       

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)      
        set_parameter_requires_grad(model_ft, feature_extract)      

        num_ftrs = model_ft.fc.in_features              
        model_ft.fc = nn.Linear(num_ftrs, num_classes)  

        input_size = 224                                

    return model_ft, input_size

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
# print(model_ft)


model_ft = model_ft.to(device)

print("Params to learn:")
if feature_extract:
    params_to_update = []                           
    for name,param in model_ft.named_parameters():  
        if param.requires_grad == True:              
            params_to_update.append(param)           
            print("\t",name)
else:                                                
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()                                  

def train_model(model, dataloaders, criterion, optimizer, num_epochs=5, round=1):
    since = time.time()
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())     
    best_acc = 0.
    best_acc1 = 0.
    print("-"*10)
    print("Round {}".format(round))

    for epoch in range(num_epochs): 
        # print("-"*10)
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        # print("-"*10)

        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            if phase == "train":
                model.train() 
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:    
                inputs = inputs.to(device)              
                labels = labels.to(device)            

                with torch.autograd.set_grad_enabled(phase=="train"):  
                    outputs = model(inputs)            
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)       

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)                                
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()      


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print("{} Loss: {} Acc: {}".format(phase, epoch_loss, epoch_acc))
            # print("-"*10)
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())   
            
            if phase == "train" and epoch_acc > best_acc1:
                best_acc1 = epoch_acc           

            if phase == "val":
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == "train":
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)                                 

        print()

    time_elapsed = time.time() - since
    # print("-"*10)
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best train Acc: {}".format(best_acc1))
    print("Best val Acc: {}".format(best_acc))
    print("round {}".format(round))
    print("-"*10)

    model.load_state_dict(best_model_wts)                                         
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

def test_model(model, dataloaders, criterion, optimizer,round=1):
    since = time.time()
    running_loss = 0.
    running_corrects = 0.
    # print("-"*10)
    print("round {}".format(round))
    model.eval()
    for inputs, labels in dataloaders['test']:   
        inputs = inputs.to(device)               #inputs.shape = torch.Size([32, 3, 224, 224])
        labels = labels.to(device)
        outputs = model(inputs)              #outputs.shape = torch.Size([32, 10])
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)                                
        running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
    epoch_loss = running_loss / len(dataloaders['test'].dataset)
    epoch_acc = running_corrects / len(dataloaders['test'].dataset)
    # print("-"*10)
    print("test Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    print("-"*10)
    
    time_elapsed = time.time() - since
    print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))


                                       
    return model, epoch_acc,epoch_loss


trhist_long = []
ohist_long = []
testhist_long = []
testloss_long = []
train_loss_long = []
val_loss_long = []
round = 7
for i in range(round):
    model_ft, trhist, ohist,train_loss_history, val_loss_history = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,round=i)
    model_ft, epoch_acc,epoch_loss = test_model(model_ft, dataloaders_dict, criterion, optimizer_ft,round=i)
    trhist_long.append(trhist)
    ohist_long.append(ohist)
    testhist_long.append(epoch_acc)
    testloss_long.append(epoch_loss)
    train_loss_long.append(train_loss_history)
    val_loss_long.append(val_loss_history)


trhist_long = functools.reduce(operator.concat, trhist_long)
ohist_long = functools.reduce(operator.concat, ohist_long)
train_loss_long = functools.reduce(operator.concat, train_loss_long)
val_loss_long = functools.reduce(operator.concat, val_loss_long)
print(testhist_long)



model_saved_name = 'Epoch_' + str(num_epochs) + '_' + 'Round' + '_' + str(round) +'.pkl'
model_parameter_saved_name = 'Epoch_' + str(num_epochs) + '_' + 'Round' + '_' + str(round) + '_parameter.pkl'
torch.save(model_ft, model_saved_name)
torch.save(model_ft.state_dict(), model_parameter_saved_name)


fig = plt.figure(1)
plt.title("Validation Accuracy vs. Training Accuracy")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.plot(range(1,round*num_epochs+1),ohist_long,label="val")
plt.plot(range(1,round*num_epochs+1),trhist_long,label="train")
plt.ylim((0,1.))
plt.xticks(np.arange(1, round*num_epochs+1, 5))
plt.legend()
# plt.show()
pic_acc_name = 'Resnet_acc1.png'
plt.savefig(pic_acc_name,bbox_inches='tight')

fig = plt.figure(2)
plt.title("Validation loss vs. Train loss")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.plot(range(1,round*num_epochs+1),val_loss_long,label="val")
plt.plot(range(1,round*num_epochs+1),train_loss_long,label="train")
plt.ylim((0,1.5))
plt.xticks(np.arange(1, round*num_epochs+1, 5))
plt.legend()
# plt.show()
pic_acc_name = 'Resnet_loss1.png'
plt.savefig(pic_acc_name,bbox_inches='tight')


fig = plt.figure(3)
plt.title("test Accuracy")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.plot(range(1,round+1),testhist_long,label="test")
# plt.plot(range(1,round*num_epochs+1),train_loss_long,label="train")
plt.ylim((0,1.))
plt.xticks(np.arange(1, round+1, 1.0))
plt.legend()
# plt.show()
pic_acc_name = 'Resnet_testacc1.png'
plt.savefig(pic_acc_name,bbox_inches='tight')


fig = plt.figure(4)
plt.title("test Loss")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.plot(range(1,round+1),testloss_long,label="test")
# plt.plot(range(1,round*num_epochs+1),train_loss_long,label="train")
plt.ylim((0,1.5))
plt.xticks(np.arange(1, round+1, 1.0))
plt.legend()
# plt.show()
pic_acc_name = 'Resnet_testloss1.png'
plt.savefig(pic_acc_name,bbox_inches='tight')