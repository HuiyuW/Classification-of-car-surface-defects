# Applied Machine Intelligence Group 06

This repository contains codes, jupyter notebooks, report, Pecha Kuchas, video etc. for the final project  "**classification for distinguishing the cropped damage patches**".
> The source dataset are obtained from our industry partner WENN, which is not included in this repository.

![Video Preview](./Video/group6.gif)

Watch our full project video in YouTube:
https://www.youtube.com/channel/UCXvKvZtF3pL0RIcO6XKddOw/featured

## Team Members
- **Zucheng Han** (Responsible for: Data Preprocessing, Baseline Algorithm, Web, Kubernetes)
- **Huiyu Wang** (Responsible for: Resnet-18, Active Learning)
- **Xin Zhang** (Responsible for: Docker, Web)
- **Mingcong Li** (Responsible for: Web Fronted Design, Video)
- **Wantao Li** (Responsible for: InceptionV3, Report)
- **Fengrui Gan** (Responsible for: Resnet-50, Report)
- **Runze Li** (Responsible for: Report, Video)
- **Xueqing Nie** (Responsible for: MobilenetV2, Web Fronted Design)
- **Yingyi Zhang** (Responsible for: Report)
- **Bowen Yuan** (Responsible for: Report)

## Introduction

## Content
- [Preprocessing: fetch images according to the json. file](./dataPreprocess)
- [The annotation tool we use to manually annotate images](./Label_Tool)
- [Baseline classification algorithms](./Classification)
- [Final model includes active learning](./Model)
- [Pecha_Kucha PPT](./Pecha_Kucha%20PPT)
- [Kubernetes](./Kubernetes)
- [Web](./Web)
- [Video](./Video)
- [Report](./AMI_Group_06_Report.pdf)

## Classification Algorithm
- [ResNet18](./Model/Resnet18/main.py)
- [InceptionV3](./Classification/3.%20InceptionV3_0.8166.ipynb)
- [MobilenetV2](./Classification/mobilenetV2)
- [Resnet-50](./Classification/4.ResNet50_0.777.ipynb)
- [VGG](./Classification/2.%20VGG16_Fine%20Tuning_0.7722.ipynb)
- [CNN](./Classification/1.%20CNN_0.6389.ipynb)
- [Autoencoder](./Classification/7.Autoencoder%20and%20tsne%20help%20correct%20labels%20in%20early%20stage.ipynb)
- [Conventional machine learning algorithm](./Classification/8.Autoencoder%20improve%20the%20classification%20performance%20-%20KNN%2C%20Logistic%20Regression%2C%20SVM%20and%20Random%20Forest.ipynb)
- [Kmeans](./Classification/9.Kmeans.ipynb)

## Active Learning with ResNet18
- [Baseline Model](./Model/Resnet_AL/src/model_parameter.py)
- [Randomly selection of a small set of labeled data](./Model/Resnet_AL/src/select_image.py)
- [Classifier based on labeled data learning](./Model/Resnet_AL/main.py)
- [Most uncertain data Query](./Model/Resnet_AL/src/select_image.py)
- [Manual labeling](./Model/Resnet_AL/main.py)
- [Model refine](./Model/Resnet_AL/main.py)

## How to run our docker image Web locally and how to deploy our docker image Web in Kubernetes
You can check the [Web folder](./Web) to know how to run our docker image Web locally. And there are two ways to deploy our web image in Kubernetes. We highly recommend the manual way, which you can view the method in the [Web folder](./Web). Due to the unstable Gitlab runner tag ami, automatic deployment by ```.gitlab-ci.yml``` often doesn't work, but this document can help you understand how to manually deploy our web application in the Kubernetes.


