import numpy as np
import torch
import torch.nn as nn
import time
import copy
from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from torch.utils.data import random_split
from src.model_parameter import initialize_model
from src.select_image import play_query
from src.WENN_dataset import WENN_dataset
from src.fetch_image import dataPreprocess
from src.creterion import ConfusionMatrix




def main(path,select,k):
################################################################################################ Basic settings
    classes = ('Dent','Other','Rim','Scratch')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes=4
    model_name = "resnet"
    feature_extract = True   
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    model_ft = model_ft.cuda()

    # print("Params to learn:")
    if feature_extract:
        params_to_update = []                            
        for name,param in model_ft.named_parameters():   
            if param.requires_grad == True:              
                params_to_update.append(param)           

    else:                                               
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                pass

    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9) 




    dp = dataPreprocess(path) #create folder with all 897 annotated images in size 224*224
    dp.fetchImages()
    label_frame = pd.read_csv('Labels\label0-896_clean.csv') # load csv clean as original dataframe
    label_frame_copy = label_frame.copy()
    label_frame_copy['human_label'] = label_frame_copy['human_label'].map(lambda x: x-1)
    dataframe = label_frame_copy
    torch.manual_seed(0)
    test_split = 0.3 #val:test = 0.7:0.3
    test_size = int(test_split * dataframe.shape[0])
    val_size = dataframe.shape[0] - test_size
    val_index_list, test_index_list = random_split(np.arange(dataframe.shape[0]),[val_size, test_size]) # val set and test set are 
    #represented by index list of dataframe

    transform = transforms.Compose([transforms.ToPILImage(), #transform will not change
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])

################################################################################################ Settings for AL rounds


    max_queried = int(0.7 * len(val_index_list)) #After rounds of AL will train set reach maximum 0.85 val set
    limit = 80 #Each round of AL will train 50 epochs.
    queried = k # Each round of AL will take k imgs from val to train set
    active_iteration = 0 # count rounds of AL
    val_acc_history = []
    val_loss_history = []
    test_acc_history = []
    test_loss_history = []
    train_index_list = []
    k_p = 15

################################################################################################  First round of AL random sample
    img_index_select = play_query(val_index_list,model_ft,dataframe,transform,k_p,select=4) # randomly choose k images from val to train
    img_index_select = img_index_select.tolist()
    img_index_select = map(int,img_index_select)
    img_index_select = list(img_index_select)

    for i in range(len(img_index_select)):
        train_index_list.append(val_index_list[img_index_select[i]]) # get imgs from val set to train set
    val_index_list = np.delete(val_index_list, img_index_select) # delete imgs from val
    best_test_acc = 0.
################################################################################################ Al rounds
    while queried < max_queried: # queried will increase until max_queried, AL will stop
        active_iteration += 1
        
        img_index_select = play_query(val_index_list,model_ft,dataframe,transform,k,select) #select k imgs from val to train in select=1 way
        img_index_select = img_index_select.tolist()
        img_index_select = map(int,img_index_select)
        img_index_select = list(img_index_select)

   
        for i in range(len(img_index_select)):
            train_index_list.append(val_index_list[img_index_select[i]])
        val_index_list = np.delete(val_index_list, img_index_select)

    
        print("-"*10)
        print ('val_set size:',  len(val_index_list))
        print ('train_set size:',  len(train_index_list))
        print ('test_set size:',  len(test_index_list))

################################################################################################ Train        
        since = time.time()
        for it in range(limit):
            running_loss = 0.
            running_corrects = 0.
            model_ft.train()
            train_dataset = WENN_dataset(train_index_list,dataframe,transform=transform) #set up WENN_dataset get img from Annotated images 224 folder
            train_dataloader = DataLoader(train_dataset, batch_size=int(len(train_dataset)/3), num_workers=0, drop_last=True)
            dict = train_dataset.count_classes() # count classes distribution in train set and reweighting it
            class_sample_counts = [dict[0], dict[1], dict[2], dict[3]]
            weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
            weights = weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=weights) #add weighting in criterion to solve inbalance of dataset


            for inputs, labels in train_dataloader:
                inputs = inputs.cuda()        
                labels = labels.cuda()
                with torch.autograd.set_grad_enabled(True):

                    outputs = model_ft(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1) #get predictions of model
                optimizer_ft.zero_grad()
                loss.backward()
                optimizer_ft.step()
                running_loss += loss.item() * inputs.size(0)                                 
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()     
            train_epoch_loss = running_loss / len(train_dataloader.dataset)
            train_epoch_acc = running_corrects / len(train_dataloader.dataset)
            train_epoch_acc = round(train_epoch_acc,4)
            if it%5==0:
                print('Epoch',it)
                print("train Loss of each epoch: {} Acc: {}".format(train_epoch_loss, train_epoch_acc))
        time_elapsed = time.time() - since
        print("Training compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))
################################################################################################ Val
        running_loss = 0.
        running_corrects = 0.
        model_ft.eval()
        val_datalset = WENN_dataset(val_index_list,dataframe,transform=transform) #set up val set dataset
        val_dataloader = DataLoader(val_datalset, batch_size=4, num_workers=0, drop_last=True)
        for inputs, labels in val_dataloader:
            inputs = inputs.cuda()               
            labels = labels.cuda()
            with torch.autograd.set_grad_enabled(False): # No learning when val
                outputs = model_ft(inputs)              
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)                                
            running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()
        val_epoch_loss = running_loss / len(val_dataloader.dataset)
        val_epoch_acc = running_corrects / len(val_dataloader.dataset)
        val_epoch_acc = round(val_epoch_acc,4)
        print("val_Loss: {} Acc: {}".format(val_epoch_loss, val_epoch_acc))
        val_acc_history.append(val_epoch_acc)
        val_loss_history.append(val_epoch_loss)
        queried += k
################################################################################################ Test
        class_correct = list(0. for i in range(4)) # in order to count accuracy of each classes
        class_total = list(0. for i in range(4))
        running_loss = 0.
        running_corrects = 0.
        confusion = ConfusionMatrix(num_classes=4) #set up Matrix to calculate TN TP FN FP 
        model_ft.eval()
        test_datalset = WENN_dataset(test_index_list,dataframe,transform=transform)
        test_dataloader = DataLoader(test_datalset, batch_size=4, num_workers=0, drop_last=True)
        for inputs, labels in test_dataloader:
            inputs = inputs.cuda()               
            labels = labels.cuda()
            with torch.autograd.set_grad_enabled(False): # No learning when train
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
                class_correct[label] += c[i].item() # count accuracy of each classes
                class_total[label] += 1
        acc,table = confusion.summary() # get Precision Recall Specificity and F1 from table

        test_epoch_loss = running_loss / len(test_dataloader.dataset)
        test_epoch_acc = running_corrects / len(test_dataloader.dataset)
        test_epoch_acc = round(test_epoch_acc,4)
        print("test_Loss: {} Acc: {}".format(test_epoch_loss, test_epoch_acc))
        print("the model accuracy is ", acc)
        print(table)
        for i in range(4):
            print('Accuracy of %5s : %2d %%' %(classes[i],100*class_correct[i]/class_total[i]))
        test_acc_history.append(test_epoch_acc)
        test_loss_history.append(test_epoch_loss)
        if test_epoch_acc > best_test_acc: # save best parameter if accuracy decreases when training
            best_test_acc = copy.deepcopy(test_epoch_acc)
            best_model_wts = copy.deepcopy(model_ft.state_dict())
            best_train_index_list = copy.deepcopy(train_index_list)
            best_class_correct = copy.deepcopy(class_correct)
            best_class_total = copy.deepcopy(class_total)
            best_confusion = copy.deepcopy(confusion)
    print("-"*10)
################################################################################################ Plot and save
    print('val_acc',val_acc_history)
    print('val_loss',val_loss_history)
    print('test_acc',test_acc_history)
    print('test_loss',test_loss_history)
    best_train_dataframe = dataframe.iloc[best_train_index_list,:] # get best accuracy corresponding train dataframe
    best_counts = len(best_train_index_list) #count the size of train set
    best_acc,best_table = best_confusion.summary() # get best accuracy corresponding criterion table and acc
    print("the model best accuracy is ", best_acc)
    print(best_table)
    table_txt = best_table.get_string() # save PrettyTable in txt form
    best_confusion.plot()
    csv_saved_name = './results/AL_' + str(k) + '_' +'limit'+str(limit)+ 'accuracy' + '_' + str(best_test_acc)+ '_' + 'images'+ '_' +str(best_counts)+'select'+ '_' +str(select)+'.csv'
    
    best_train_dataframe.to_csv(csv_saved_name,index=False,header=True) # save train dataframe in csv
    print("best_train_dataframe.csv saved")

    print(best_counts,'images were used')
    for i in range(4):
        print('Best Accuracy of %5s : %2d %%' %(classes[i],100*best_class_correct[i]/best_class_total[i])) # get best accuracy corresponding classes distribution

    model_parameter_saved_name = './results/AL_' + str(k) + '_' +'limit'+str(limit)+ 'accuracy' + '_' + str(best_test_acc) + 'images'+ '_' +str(best_counts)+'select'+ '_' +str(select)+'_parameter.pkl'
    torch.save(best_model_wts, model_parameter_saved_name) # save best accuracy corresponding model parameters
    print("model_parameter saved")

    txt_name =  './results/AL_' + str(k) + '_' +'limit'+str(limit)+ 'accuracy' + '_' + str(best_test_acc) + 'images'+ '_' +str(best_counts)+'select'+ '_' +str(select)+'.txt'
    f = open(txt_name,'w+') # save some info in txt form
    f.write('Val_acc_history'+' : '+str(val_acc_history)+'\n')
    f.write('Val_loss_history'+' : '+str(val_loss_history)+'\n')
    f.write('Test_acc_history'+' : '+str(test_acc_history)+'\n')
    f.write('Test_loss_history'+' : '+str(test_loss_history)+'\n')
    f.write(table_txt+'\n')
    f.write(str(best_counts)+' images were used'+'\n')
    for i in range(4):
        f.write('Best Accuracy of '+str(classes[i])+' : '+ str(100*best_class_correct[i]/best_class_total[i])+'\n')
    f.close()


    fig = plt.figure(num = select,figsize=(10,6)) #plot all sampling methods comparison
    plt.title("Validation Accuracy vs. Test Accuracy")
    plt.xlabel("AL Rounds")
    plt.ylabel("Accuracy")
    plt.plot(range(1,active_iteration+1),val_acc_history,label="Val_"+str(select))
    plt.plot(range(1,active_iteration+1),test_acc_history,label="Test_"+str(select))
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, active_iteration+1, 5))
    plt.legend()
    pic_acc_name = './results/AL_' + str(k) +'_'+'limit'+str(limit)+'_select_'+str(select)+'_accuracy' + '_' + str(best_test_acc) +'_acc.png'
    plt.savefig(pic_acc_name,bbox_inches='tight')

    fig = plt.figure(num = select+50,figsize=(10,6))
    plt.title("Validation loss vs. Test loss")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,active_iteration+1),val_loss_history,label="Val_"+str(select))
    plt.plot(range(1,active_iteration+1),test_loss_history,label="test_"+str(select))
    plt.ylim((0,1.5))
    plt.xticks(np.arange(1, active_iteration, 5))
    plt.legend()
    pic_acc_name = './results/AL_' + str(k)+'_'+'limit'+str(limit) +'_select_'+str(select) +'_accuracy' + '_' + str(best_test_acc) +'_loss.png'
    plt.savefig(pic_acc_name,bbox_inches='tight')

    return active_iteration, val_acc_history, test_acc_history, val_loss_history, test_loss_history











if __name__ == '__main__':

    select_list = [1,2,3,4] # sampling strategies list of AL 
    
    path = 'C:/Users/wanghuiyu/Desktop/AMI dataset/Data/' #load WENN Data folder path
    k = 20  #how many images are sampled at each round of AL
    val_acc_history_all = []
    test_acc_history_all = []
    val_loss_history_all = []
    test_loss_history_all = []    
    since = time.time()
    for i in range(len(select_list)): #Train sequentially according to sampling method list
        active_iteration, val_acc_history, test_acc_history, val_loss_history, test_loss_history = main(path,select_list[i],k)
        val_acc_history_all.append(val_acc_history)
        test_acc_history_all.append(test_acc_history)
        val_loss_history_all.append(val_loss_history)
        test_loss_history_all.append(test_loss_history)
    time_elapsed = time.time() - since
    print("all compete in {}m {}s".format(time_elapsed // 60, time_elapsed % 60))


    fig = plt.figure(num = 5,figsize = (10,6))
    plt.title("Validation Accuracy vs. Test Accuracy")
    plt.xlabel("AL Rounds")
    plt.ylabel("Accuracy")
    for i in range(len(select_list)):
        plt.plot(range(1,active_iteration+1),val_acc_history_all[i],label="Val_"+str(select_list[i]))
        plt.plot(range(1,active_iteration+1),test_acc_history_all[i],label="Test_"+str(select_list[i]))
        
        #'AL_' + str(k) +'_select_'

    pic_acc_name = './results/AL_' + str(k) +'_selectnum_'+str(len(select_list))+'_acc.png'
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, active_iteration+1, 5))
    plt.legend()
    plt.savefig(pic_acc_name,bbox_inches='tight')

    fig = plt.figure(num = 6,figsize = (10,6))
    plt.title("Validation loss vs. Test loss")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")

    for i in range(len(select_list)):
        plt.plot(range(1,active_iteration+1),val_loss_history_all[i],label="Val_"+str(select_list[i]))
        plt.plot(range(1,active_iteration+1),test_loss_history_all[i],label="Test_"+str(select_list[i]))


    pic_acc_name = './results/AL_' + str(k) +'_selectnum_'+str(len(select_list))+'_loss.png'
    plt.ylim((0,1.5))
    plt.xticks(np.arange(1, active_iteration, 5))
    plt.legend()
    plt.savefig(pic_acc_name,bbox_inches='tight')

    