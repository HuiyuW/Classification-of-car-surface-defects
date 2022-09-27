import numpy as np

def image_to_vector(image, n):
    image_array=np.array(image)

    returnVect = np.zeros((1, n*n))
    for i in range(n):
        lineStr = image_array[i]
        for j in range(n):
            returnVect[0, n*i+j] = int(lineStr[j])

    return returnVect