import json
import pandas as pd
import os
from drawbbox import draw_bbox
from save_bbox_images import save_bbox


def mkdir(path):

    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


new_folder = "saved_images_with_bbox"

mkdir(new_folder)  # create a new folder for images with bbox

label_dict = {
    "1": "Dent",
    "2": "Other",
    "3": "Rim",
    "4": "Scratch",
}  # labels and names dict

########################################################################################################

with open(
    "Label_Tool//annotated_functional_test3_fixed.json", "r", encoding="utf-8"
) as f:  # json file should be same path
    objectDict = json.load(f)  # load json
    len_annotation = len(objectDict["annotations"])  # annotations 897
    human_label = []
    image_id_list = []
    annotation_index = []
    label_name = []
    print("hier")

    #  for idx in range(len_annotation):1
    # you can change labeling range here!
    for idx in range(550, 555):
        print(idx)
        if idx > 720:
            break
        image_id = objectDict["annotations"][idx]["image_id"]  # get image id
        print("current image id is " + str(image_id))
        print(
            "label 1 -> Dent \nlabel 2 -> Other \nlabel 3 -> Rim \nlabel 4 -> Scratch."
        )
        draw_bbox(idx)  # show image with bbox but no save
        save_bbox(
            idx, new_folder
        )  # save image in turn but no show delete this line if needed
        print("hier__")
        label = int(
            input("What do u think expert? write label number here ：")
        )  # type in label number
        label_name.append(label_dict[str(label)])
        human_label.append(label)
        image_id_list.append(image_id)
        annotation_index.append(idx)
        objectDict["annotations"][idx]["damage"] = label  # this was for saving in json

#  new_json = json.dumps(objectDict)
#  f2 = open('annotated_functional_test3_fixed copy.json', 'w')
#  f2.write(new_json)
#  f2.close()
#################################################################################################################
# save labels in csv
data = {
    "annotation_index": annotation_index,
    "image_id": image_id_list,
    "human_lebel": human_label,
    "label_name": label_name,
}
data_df = pd.DataFrame(data)
data_df.to_csv("label.csv", index=False, header=True)
print("csv_file saved")
