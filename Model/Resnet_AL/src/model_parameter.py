import torch.nn as nn
from torchvision import models



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