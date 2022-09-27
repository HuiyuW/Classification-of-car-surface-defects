from PIL import Image
import numpy as np


def imageResize(path):
    img = Image.open(path)
    return img.resize((400, 200))
