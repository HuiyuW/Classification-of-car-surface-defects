from dataclasses import dataclass
from inspect import Parameter
import os
from select import select
import shutil
from sys import flags
#from tkinter import Button
from django.conf import settings
import json
import imghdr
from pickle import TRUE
from django.shortcuts import render, redirect, HttpResponseRedirect
from django.http import HttpResponse, JsonResponse

# from regex import F
from .models import Post, UploadPhotos, ProcessPhotos, LabelPhotos2
from django.core.files.storage import FileSystemStorage

# from tests import imageResize

from PIL import Image
import numpy as np

# prepare for inceptionV3
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
import cv2


# prepare for Resnet18
# just use it to predict
from .Resnet_ALWeb.main_train import resnet18Predict, main_select, main_train, newresnet18Predict


IMAGE_SIZE = (236, 255)
IMAGE_SIZE2 = (224, 224)
class_names = ["Dent", "Other", "Rim", "Scratch"]


def preprocess(imagepath, image_size):
    image = cv2.imread(imagepath)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    # image = image / 255.0
    return image


def pre_labels_output(pred_report):
    predictions = []
    predictions.append(pred_report)
    predictions = np.array(predictions)
    predictions = predictions.sum(axis=0)
    pred_labelsid = predictions.argmax(axis=1)
    pred_labels = []
    for id in pred_labelsid:
        pred_labels.append(class_names[id])
    return pred_labels


model = load_model("imgProcess/models/mymodel.h5")
image_list_url = []
preimage_list_url = []
photos = []
prephotos = []
predict_labels = []
param_ac = []

# try to make empty folder for active learning
def mkdir(path):

    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


        


def indexPage(request):
    # upphotos = UploadPhotos.objects.all()
    global image_list_url
    global preimage_list_url
    global photos
    global predict_labels
    global param_ac
    # global model_sel
    pagestatus = 1
    model_sel = 1
    if request.method == "POST":
        image_list_url = []
        infos = []
        photos = []
        images = request.FILES.getlist("images")
        fs = FileSystemStorage()
        for image in images:
            name = fs.save(image.name, image)
            url = fs.url(name)
            image_list_url.append(url)
            photo = UploadPhotos.objects.create(description=url, image=image,)
            photos.append(photo)

        context = {
            "image_list_url": image_list_url,
            "preimage_list_url": preimage_list_url,
            "photos": photos,
            "maxnums": len(photos),
            "pagestatus": pagestatus,
        }
        return render(request, "home.html", context)

    if request.GET.get("detection") and len(photos)>0: 
        pagestatus = 2
        active = False
        preimage_list_url = []
        test_imgs = []
        predict_labels = []
        model_sel = request.GET.get("model_sel")
        num = int(request.GET.get("label_ac"))
        select = int(request.GET.get("method_ac"))
        #print("...................")
        #print(model_sel)
        #print(num)
        #print(select)
        param_ac = [num, select]


        if model_sel == "4":
            active = True
            print(active)

            ## create new empty folder
            activ_learning_folder = "media/activ_learning/"
            mkdir(activ_learning_folder) 

            activ_learning_dict = dict()

            ac_dict = {}
            for i, url in enumerate(image_list_url):
                imagepath = "." + url
                img = preprocess(imagepath, IMAGE_SIZE2)
                test_imgs.append(img)
                namafilebaru = imagepath[:-5] + imagepath[-5:]
                #cv2.imwrite(namafilebaru, img)
                # filebaru = img.save(namafilebaru)
                preimage_url = url[:-5] + "dete_" + url[-5:]
                preimage_list_url.append(preimage_url)

                predict_label = resnet18Predict(img)
                predict_labels.append(predict_label)
                # prepare the folder and save images in activ learning folder
                active_learning_path = "./media/activ_learning/"+ namafilebaru[8:]
                cv2.imwrite(active_learning_path, img)

                #save the label the path in the dict
                ac_dict[active_learning_path] = [predict_label, i]
            #print("-----predict labels old model-----------")
            #print(predict_labels)
            
            for i in np.arange(len(photos)):
                photos[i].description = predict_labels[i]
                photos[i].save(update_fields=["description"])
                print(photos[i].description)

            path = './media/activ_learning/' # user upload img folder path
            
            img_path_list = main_select(path,num,select)# as long as you get img_path_list then you can train the model
            #print("1231")
            #print(img_path_list)
            ac_predict_labels = []
            ac_photos = []

            for path in img_path_list:
                if path in ac_dict:
                    ac_predict_labels.append(ac_dict[path][0])
                    ac_photos.append(photos[ac_dict[path][1]])

            #param_ac.append(ac_predict_labels)
            param_ac.append(img_path_list)
            data = zip(img_path_list, ac_photos, ac_predict_labels)
            context = {
                "image_list_url": image_list_url,
                "preimage_list_url": preimage_list_url,
                "photos": photos,
                "alldata": data,
                "pagestatus": pagestatus,
                "active": active,
                # "prephotos": prephotos,
            }
            return render(request, "home.html", context)

        elif model_sel == "1":
            print(active)
            for url in image_list_url:
                imagepath = "." + url
                img = preprocess(imagepath, IMAGE_SIZE)
                test_imgs.append(img)
                namafilebaru = imagepath[:-5] + "dete_" + imagepath[-5:]
                cv2.imwrite(namafilebaru, img)
                # filebaru = img.save(namafilebaru)
                preimage_url = url[:-5] + "dete_" + url[-5:]
                preimage_list_url.append(preimage_url)
                # predict_labels.append("other")

            test_imgs = np.array(test_imgs, dtype="float32") / 255
            test_features = InceptionV3(weights="imagenet", include_top=False).predict(
                test_imgs
            )
            
            predict_labels = pre_labels_output(model.predict(test_features))

            for i in np.arange(len(photos)):
                photos[i].description = predict_labels[i]
                photos[i].save(update_fields=["description"])

                print(photos[i].description)

            data = zip(preimage_list_url, photos, predict_labels)
            context = {
                "image_list_url": image_list_url,
                "preimage_list_url": preimage_list_url,
                "photos": photos,
                "alldata": data,
                "pagestatus": pagestatus,
                "active": active,
                # "prephotos": prephotos,
            }
            return render(request, "home.html", context)

        elif model_sel == "2":
            pass

        elif model_sel == "3":
            #print(active)
            for url in image_list_url:
                imagepath = "." + url
                img = preprocess(imagepath, (224, 224))
                test_imgs.append(img)
                namafilebaru = imagepath[:-5] + "dete_" + imagepath[-5:]
                cv2.imwrite(namafilebaru, img)
                preimage_url = url[:-5] + "dete_" + url[-5:]
                preimage_list_url.append(preimage_url)

                predict_label = resnet18Predict(img)
                predict_labels.append(predict_label)

            #print(predict_labels)

            for i in np.arange(len(photos)):
                photos[i].description = predict_labels[i]
                photos[i].save(update_fields=["description"])

                print(photos[i].description)

            data = zip(preimage_list_url, photos, predict_labels)
            context = {
                "image_list_url": image_list_url,
                "preimage_list_url": preimage_list_url,
                "photos": photos,
                "alldata": data,
                "pagestatus": pagestatus,
                "active": active,
                # "prephotos": prephotos,
            }
            return render(request, "home.html", context)


    data = zip(preimage_list_url, photos, predict_labels)
    for i in np.arange(len(photos)):
        print(photos[i].description)

    context = {
        "image_list_url": image_list_url,
        "preimage_list_url": preimage_list_url,
        "photos": photos,
        "alldata": data,
        "pagestatus": pagestatus,
        "model_sel": model_sel,
    }

    return render(request, "home.html", context)


def addtomodel(request):

    if request.method == "POST":
        #print("hier image_id")
        img_id = request.POST["img_id"]
        #print(img_id)
        img_label = request.POST["img_label"]
        #print(img_label)
        upimage = UploadPhotos.objects.get(id=img_id)
        upimage.description = img_label
        upimage.save(update_fields=["description"])
        print("change to " + " " + upimage.description)
        img = upimage.image

        photo = ProcessPhotos.objects.create(description=img_label, image=img,)
        success = "Image added successfully"
        # return render(request, "home.html")
        HttpResponseRedirect(request.path_info)
    print("hier")
    return render(request, "home.html")


# activ learning refine


def image_list(request):
    success = False
    processdata = ProcessPhotos.objects.all()
    pagestatus = 1
    cur_len = 0
    # prepare result matrix
    p0 = 0
    p1 = 0
    p2 = 0
    p3 = 0
    r0 = 0
    r1 = 0
    r2 = 0
    r3 = 0
    s0 = 0
    s1 = 0
    s2 = 0
    s3 = 0
    f0 = 0
    f1 = 0
    f2 = 0
    f3 = 0
    sum_acc = 0
    sum_recall = 0
    sum_f1 = 0
    if request.method == "POST":
        # refine the model with the image in ProcessPhotos
        '''
        label_list = []
        for photo in processdata:
            label_list.append(photo.description)
        '''
        processdata = ProcessPhotos.objects.all()
        #print("----------------param_ac-------------------")
        #print(param_ac)
        table = []
        if param_ac:
            #print("----------------change labels-------------------")
            labels = [data.description for data in processdata]
            #print(labels)
            label_list = []   
            #print("------labels--------")     
            #print(labels)
            #print(len(labels))
            #print(len(param_ac[2]))
            max_len = len(labels)
            cur_len = len(param_ac[2])

            if labels:
                for label in labels[max_len-cur_len:max_len]:
                    if label == "Dent":
                        label_list.append(0)
                    elif label == "Other":
                        label_list.append(1)
                    elif label == "Rim":
                        label_list.append(2)
                    elif label == "Scratch":
                        label_list.append(3)

                #print("--------------list--------------")
                #print(label_list)
                #print(len(label_list))
                success = True
                path_dataset = './imgProcess/Resnet_ALWeb/Annotated_images/' # WENN dataset Data path
                num = param_ac[0]
                select = param_ac[1]
                img_path_list = param_ac[2]
                #print(img_path_list)
                
                table, acc_list = main_train(path_dataset,num,select,label_list,img_path_list)
                # generate the values of table
                p0 = table[0][1]
                p1 = table[1][1]
                p2 = table[2][1]
                p3 = table[3][1]
                r0 = table[0][2]
                r1 = table[1][2]
                r2 = table[2][2]
                r3 = table[3][2]
                s0 = table[0][3]
                s1 = table[1][3]
                s2 = table[2][3]
                s3 = table[3][3]
                f0 = table[0][4]
                f1 = table[1][4]
                f2 = table[2][4]
                f3 = table[3][4]
                #print("-------table---------")
                #print(table)
                #print("acc_list")
                #print(acc_list)
                # get sum_acc , recall , F1 for return
                sum_acc = round(float(acc_list[4]),3)
                sum_recall = round((table[0][2] + table[1][2] + table[2][2] + table[3][2])/4,3)
                sum_f1 = round((table[0][4] + table[1][4] + table[2][4] + table[3][4])/4,3)
                predict_labels = []
                for i, url in enumerate(image_list_url):

                    imagepath = "." + url
                    img = preprocess(imagepath, (224, 224))
                    predict_label = newresnet18Predict(img)
                    predict_labels.append(predict_label)

                #print("----------predict labels new model------------------")
                #print(predict_labels)
                    
                for i in np.arange(len(photos)):
                    photos[i].description = predict_labels[i]
                    photos[i].save(update_fields=["description"])
                    print(photos[i].description)
                success = True

                # after refine it will remove all pictures from table, it exists some problem 
                processdata = ProcessPhotos.objects.all()
                for data in processdata:
                    data.delete()
                

                pagestatus = 2
        else:
            pass

        return render(
            #request, "image_list.html", {"data": processdata, "success": success}
            request, "image_list.html", {"success": success, "pagestatus": pagestatus, 
            "p0": p0, "r0": r0, "s0": s0, "f0":f0,
            "p1": p1, "r1": r1, "s1": s1, "f1":f1,
            "p2": p2, "r2": r2, "s2": s2, "f2":f2,
            "p3": p3, "r3": r3, "s3": s3, "f3":f3,
            "sum_acc":sum_acc,"sum_recall":sum_recall, "sum_f1":sum_f1,"cur_len":cur_len}
        )
    return render(request, "image_list.html", {"data": processdata, "success": success, "pagestatus": pagestatus})


def delete_photoac(request, id):
    if request.method == "POST":
        photo = ProcessPhotos.objects.get(id=id)
        photo.delete()
    return redirect("image_list")


# download report


def final_report(request):
    
    processdata = UploadPhotos.objects.all()
    savepath = settings.STATICFILES_DIRS[0] + "/jsonfile/final_report.json"
    jsontext = {"annotations": []}
    for data in processdata:
        jsontext["annotations"].append(
            {"file_name": os.path.basename(data.image.name), "label": data.description}
        )
    jsondata = json.dumps(jsontext, indent=4, separators=(",", ": "))
    f = open(savepath, "w")
    f.write(jsondata)
    f.close()

    if request.GET.get("clear_report"):
        processdata = UploadPhotos.objects.all()
        for data in processdata:
            data.delete()
        
        processdata = UploadPhotos.objects.all()
        savepath = settings.STATICFILES_DIRS[0] + "/jsonfile/final_report.json"
        jsontext = {"annotations": []}
        for data in processdata:
            jsontext["annotations"].append(
                {
                    "file_name": os.path.basename(data.image.name),
                    "label": data.description,
                }
            )
        jsondata = json.dumps(jsontext, indent=4, separators=(",", ": "))
        f = open(savepath, "w")
        f.write(jsondata)
        f.close()
        return render(request, "final_report.html")

    return render(request, "final_report.html", {"data": processdata,})


def delete_photo(request, id):
    if request.method == "POST":
        photo = UploadPhotos.objects.get(id=id)
        photo.delete()
    return redirect("final_report")


def change_label(request, id):
    if request.method == "POST":
        photo = UploadPhotos.objects.get(id=id)
        img_changelabel = request.POST.get("img_changelabel")
        photo.description = img_changelabel
        photo.save(update_fields=["description"])
        print("hier_change_label")
    return redirect("final_report")


# website for manuel label
def image_label(request):
    labeldata = LabelPhotos2.objects.all()
    context = {
        "labeldata": labeldata,
    }
    savepath = settings.STATICFILES_DIRS[0] + "/jsonfile/image_label.json"
    jsontext = {"labels": []}
    for data in labeldata:
        if data.labelstatus:
            jsontext["labels"].append(
                {
                    "file_name": os.path.basename(data.image.name),
                    "label": data.description,
                }
            )
    jsondata = json.dumps(jsontext, indent=4, separators=(",", ": "))
    f = open(savepath, "w")
    f.write(jsondata)
    f.close()
    if request.method == "POST":
        images = request.FILES.getlist("labelimages")
        for image in images:
            labelphoto = LabelPhotos2.objects.create(
                description=" ", image=image, labelstatus=False,
            )

        labeldata = LabelPhotos2.objects.all()
        context = {
            "labeldata": labeldata,
        }
        return render(request, "image_label.html", context)

    if request.GET.get("clear_unlabeled"):
        labeldata = LabelPhotos2.objects.all()
        for data in labeldata:
            if not data.labelstatus:
                data.delete()
        # labeldata = LabelPhotos2.objects.all()
        labeldata = LabelPhotos2.objects.all()
        context = {
            "labeldata": labeldata,
        }
        return render(request, "image_label.html", context)

    if request.GET.get("clear_labeled"):
        labeldata = LabelPhotos2.objects.all()
        for data in labeldata:
            if data.labelstatus:
                data.delete()
        labeldata = LabelPhotos2.objects.all()
        savepath = settings.STATICFILES_DIRS[0] + "/jsonfile/image_label.json"
        jsontext = {"labels": []}
        for data in labeldata:
            if data.labelstatus:
                jsontext["labels"].append(
                    {
                        "file_name": os.path.basename(data.image.name),
                        "label": data.description,
                    }
                )
        jsondata = json.dumps(jsontext, indent=4, separators=(",", ": "))
        f = open(savepath, "w")
        f.write(jsondata)
        f.close()
        context = {
            "labeldata": labeldata,
        }
        return render(request, "image_label.html", context)

    return render(request, "image_label.html", context)


def add_label(request, id):
    if request.method == "POST":
        photo = LabelPhotos2.objects.get(id=id)
        img_addlabel = request.POST.get("img_addlabel")
        photo.description = img_addlabel
        photo.save(update_fields=["description"])
        photo.labelstatus = True
        photo.save(update_fields=["labelstatus"])
        print("hier_add_label")
    return redirect("image_label")


def delete_unlabelphoto(request, id):
    if request.method == "POST":
        photo = LabelPhotos2.objects.get(id=id)
        photo.delete()
        print("hier_delete")
    return redirect("image_label")


def remove_labelphoto(request, id):
    if request.method == "POST":
        photo = LabelPhotos2.objects.get(id=id)
        photo.labelstatus = False
        photo.save(update_fields=["labelstatus"])
    return redirect("image_label")


def startPage(request):
    return render(request, "start.html")


def labeling_web(request):
    return render(request, "labeling_web.html")


def activ_learning(request):
    labeldata = LabelPhotos2.objects.all()
    context = {
        "labeldata": labeldata,
    }
    savepath = settings.STATICFILES_DIRS[0] + "/jsonfile/image_label.json"
    jsontext = {"labels": []}
    for data in labeldata:
        if data.labelstatus:
            jsontext["labels"].append(
                {
                    "file_name": os.path.basename(data.image.name),
                    "label": data.description,
                }
            )
    jsondata = json.dumps(jsontext, indent=4, separators=(",", ": "))
    f = open(savepath, "w")
    f.write(jsondata)
    f.close()
    if request.method == "POST":
        images = request.FILES.getlist("labelimages")
        for image in images:
            labelphoto = LabelPhotos2.objects.create(
                description=" ", image=image, labelstatus=False,
            )

        labeldata = LabelPhotos2.objects.all()
        context = {
            "labeldata": labeldata,
        }
        return render(request, "activ_learning.html", context)
    return render(request, "activ_learning.html", context)


def model_introduce(request):
    return render(request, "model_introduce.html")


def support(request):
    return render(request, "support.html")
