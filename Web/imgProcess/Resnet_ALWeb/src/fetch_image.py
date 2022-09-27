import json
from PIL import Image
import numpy as np
import os


class dataPreprocess():# Class with function to create folder and save cropped imgs in folder
    def __init__(self, path, width=224, height=224):
        # set initial value
        self.path = path
        self._jsonPath = path + '/Annotations/'+'annotated_functional_test3_fixed.json'
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
        newFolderpath = "./Annotated_images_224"
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
            curfullImagepath = self.path + "/Images/" + curImagepath #这里要改成path+Images
            img = Image.open(curfullImagepath)
            # get position right up corner and left down corner
            curRegionpos = np.array([curPointpos[0][0], curPointpos[0][1], curPointpos[0][4], curPointpos[0][5]])
            curRegion = img.crop(curRegionpos)
            # reshape image size
            curRegion = curRegion.resize((self._resImagewidth, self._resImageheight))
            # convert to grey image
            # curRegion = curRegion.convert("L")
            # annoted image name is image_id
            curfullsaveImagepath = "./Annotated_images_224/" + str(curImageid) + "_" + str(indexAnnoted) + ".jpg"
            curRegion.save(curfullsaveImagepath)
            # save image_id in json some problem give up
            # curImagedict = {"image_id": curImageid}
            # json.dump(curImagedict, annotjsonFile)

# path = '../../Data'            
# dp = dataPreprocess(path)
# dp.fetchImages()
