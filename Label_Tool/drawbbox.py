import json
from PIL import Image
import matplotlib.pyplot as plt
import PIL.ImageDraw as ImageDraw
from PIL import ImageColor


def draw_bbox(index):
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
    img = Image.open(curfullImagepath)

    draw = ImageDraw.Draw(img)
    left = x_min
    top = y_min
    right = x_max
    bottom = y_max
    line_thickness = 8
    color = ImageColor.getrgb("Green")
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=line_thickness,
        fill=color,
    )

    plt.imshow(img)
    plt.show()

