import json
from tokenize import String
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import os

# def mkdir(path):
 
# 	folder = os.path.exists(path)
 
# 	if not folder:                   
# 		os.makedirs(path)           
# 		print ("---  new folder...  ---")
# 		print ("---  OK  ---")
 
# 	else:
# 		print ("---  There is this folder!  ---")

# new_folder = '../Annotated_images_224/'
# mkdir(new_folder)     #create a new folder for images with bbox
# 
class dataPreprocess():
    def __init__(self, jsonName: String, width=120, height=80):
        # set initial value
        self._jsonPath = jsonName
        self._resImagewidth = width
        self._resImageheight = height
        # load json file
        jsonFile = open(self._jsonPath)
        # read json in dict
        dataDict = json.load(jsonFile)
        self._dataDictimages = dataDict["images"]
        self._dataDictannoted = dataDict["annotations"]
        # get length images and annoted
        self._lenImages = len(self._dataDictimages)
        self._lenAnnoted = len(self._dataDictannoted)


        # create new folder Annotated_images
        newFolderpath = "../Annotated_images_224"
        isExists=os.path.exists(newFolderpath)
        if not isExists:
            os.makedirs(newFolderpath) 


    
    def imageQuery(self, imageid):

        for indexImages in range(0, self._lenImages):
            if imageid == self._dataDictimages[indexImages]["id"]:
                return indexImages
            elif indexImages == self._lenImages:
                print("Don't find this annnoted image")
    
    def fetchImages(self) -> None:
        for indexAnnoted in range(0, self._lenAnnoted):
            curImageid = self._dataDictannoted[indexAnnoted]["image_id"]
            curPointpos = self._dataDictannoted[indexAnnoted]["segmentation"]
            curindexImages = self.imageQuery(curImageid)
            # curFilepath is only file path in folder Images
            curImagepath = self._dataDictimages[curindexImages]["file_name"]
            # get crop images
            curfullImagepath = "../Images/" + curImagepath
            img = Image.open(curfullImagepath)
            # get position right up corner and left down corner
            curRegionpos = np.array([curPointpos[0][0], curPointpos[0][1], curPointpos[0][4], curPointpos[0][5]])
            curRegion = img.crop(curRegionpos)
            # reshape image size
            curRegion = curRegion.resize((self._resImagewidth, self._resImageheight))
            # convert to grey image
            # curRegion = curRegion.convert("L")
            # annoted image name is image_id
            curfullsaveImagepath = "../Annotated_images_224/" + str(curImageid) + "_" + str(indexAnnoted) + ".jpg"
            curRegion.save(curfullsaveImagepath)
            # save image_id in json some problem give up
            # curImagedict = {"image_id": curImageid}
            # json.dump(curImagedict, annotjsonFile)
    
    def getXvaulue(self):
        Xvalue = []
        # query all images in file annotated images
        annotImagepath = "../Annotated_images"
        for imageName in os.listdir(annotImagepath):
            #print(imageName)
            curFullpath = annotImagepath + "/" + imageName
            curImg = Image.open(curFullpath)
            arImg = np.asarray(curImg).flatten()/255
            Xvalue.append(arImg)

        X = np.array(Xvalue)
        return X


    


        

def main():
    dp = dataPreprocess('annotated_functional_test3_fixed.json', 224, 224)
    dp.fetchImages()
    # get X value
    X = dp.getXvaulue()
    print(X)


if __name__ == "__main__":
    main()