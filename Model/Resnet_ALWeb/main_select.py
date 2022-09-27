import numpy as np
import torch
from torchvision import transforms
from src.model_parameter import initialize_model
from src.select_Webimage import select_Webimg


def main_select(path,num,select):
    torch.cuda.set_device(0) # Set gpu number here
    torch.cuda.manual_seed(0)
    np.random.seed(0) # Keep val_list controlable


    num_classes=4
    model_name = "resnet"
    feature_extract = True   
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

    model_ft = model_ft.cuda()
    model_PATH = './results/AL_15_accuracy_0.8174images_375select_3_parameter.pkl'
    model_ft.load_state_dict(torch.load(model_PATH)) # load pretrained model

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

    transform = transforms.Compose([transforms.ToPILImage(), #transform will not change
    transforms.RandomHorizontalFlip(p=0.5), # random flip for more generally model
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])


    img_path_list = select_Webimg(path,model_ft,transform, num,select) # use AL sampling methods select num most uncertrain imgs
    return img_path_list
    


if __name__ == '__main__':
    path = 'C:/Users/wanghuiyu/Desktop/AMI dataset/Data/Annotated_images_test/' # user upload img folder path
    # path_dataset = 'C:/Users/wanghuiyu/Desktop/AMI dataset/Data/' # WENN dataset Data path
    num = 5
    select = 1
    img_path_list = main_select(path,num,select)
    print(img_path_list)