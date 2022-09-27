import json
from tokenize import String
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import numpy as np
import os


# do the data preprocess work, help us to fetch images according to the json file
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
        newFolderpath = "../Annotated_images"
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
        #self._lenAnnoted = 540
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
            curRegion = curRegion.convert("L")
            # annoted image name is image_id
            curfullsaveImagepath = "../Annotated_images/" + str(indexAnnoted) + "_" + str(curImageid) + ".jpg"
            curRegion.save(curfullsaveImagepath)

    def fetchgetXvalue(self):
        Xvalue = []
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
            curRegion = curRegion.convert("L")
            # directly convert
            #print(curImageid)
            arImg = np.asarray(curRegion).flatten()/255
            Xvalue.append(arImg)

        X = np.array(Xvalue)
        return X



    # try to use zero padding        
    def fetchImagesnew(self) -> None:
        #self._lenAnnoted = 540
        for indexAnnoted in range(0, self._lenAnnoted):
            curImageid = self._dataDictannoted[indexAnnoted]["image_id"]
            curPointpos = self._dataDictannoted[indexAnnoted]["segmentation"]
            curRegionhw = self._dataDictannoted[indexAnnoted]["bbox"]
            curindexImages = self.imageQuery(curImageid)
            # curFilepath is only file path in folder Images
            curImagepath = self._dataDictimages[curindexImages]["file_name"]
            # get crop images
            curfullImagepath = "../Images/" + curImagepath
            img = Image.open(curfullImagepath)
            # get position right up corner and left down corner
            curRegionpos = np.array([curPointpos[0][0], curPointpos[0][1], curPointpos[0][4], curPointpos[0][5]])
            curRegion = img.crop(curRegionpos)
            # get the height and width of image
            curHeight = int(curRegionhw[3])
            curWidth = int(curRegionhw[2])
            top = int(0.5*(self._resImageheight - curHeight))
            left = int(0.5*(self._resImagewidth - curWidth))
            # zero padding the image size
            curRegion = ImageOps.expand(curRegion, border=(left,top), fill=0)
            # reshape image size to aviod bugs
            curRegion = curRegion.resize((self._resImagewidth, self._resImageheight))
            # convert to grey image
            curRegion = curRegion.convert("L")
            # annoted image name is image_id
            curfullsaveImagepath = "../Annotated_images/" + str(curImageid) + "_" + str(indexAnnoted) + ".jpg"
            curRegion.save(curfullsaveImagepath)


    def getXvalue(self):
        Xvalue = []
        # query all images in file annotated images
        annotImagepath = "../Annotated_images"
        for imageName in os.listdir(annotImagepath):
            #print(imageName)
            curFullpath = annotImagepath + "/" + imageName
            #print(curFullpath)
            curImg = Image.open(curFullpath)
            arImg = np.asarray(curImg).flatten()/255
            Xvalue.append(arImg)

        X = np.array(Xvalue)
        return X


    


        

def main():
    dp = dataPreprocess('annotated_functional_test3_fixed.json', 255, 236)
    dp.fetchImages()
    # get X value from the fetching images path
    X = dp.getXvaulue()
    # directly get X value without fetching images
    X = dp.fetchgetXvalue()
    #print(X)


if __name__ == "__main__":
    main()