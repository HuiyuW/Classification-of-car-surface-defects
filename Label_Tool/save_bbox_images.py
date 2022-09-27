import json
import cv2


def save_bbox(index, filename):
    train1_json_path = "Label_Tool//annotated_functional_test3_fixed.json"

    with open(train1_json_path, "rb") as f:
        params = json.load(f)
    image_id = params["annotations"][index]["image_id"]
    bbox = params["annotations"][index]["segmentation"]
    x_min = bbox[0][0]
    y_min = bbox[0][1]
    x_max = bbox[0][4]
    y_max = bbox[0][5]
    image_len = len(params["images"])

    idx = []
    for indexImages in range(image_len):
        if image_id == params["images"][indexImages]["id"]:
            idx = indexImages
        elif indexImages == image_len:
            print("Don't find this annnoted image")

    curImagepath = params["images"][idx]["file_name"]
    curfullImagepath = "../Images/" + curImagepath
    folder_path = "./" + filename + "/"
    img = cv2.imread(curfullImagepath)
    cv2.rectangle(
        img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 4
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(3)
    cv2.putText(img, text, (int(x_min), int(y_min - 10)), font, 2, (0, 0, 255), 1)
    cv2.imwrite(folder_path + str(image_id) + ".jpg", img)

