from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataPreprocess import dataPreprocess
import pandas as pd

def main():
    # default value 255, 236 
    dp = dataPreprocess('annotated_functional_test3_fixed.json', 255, 236)
    #dp.fetchImages()
    # get X value
    X = dp.fetchgetXvalue()

    print("the size of input images are"+str(X.shape))

    dataLabel = pd.read_csv("label0-896.csv")

    y = dataLabel.loc[:, "human_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3, weights="uniform"))

    knn.fit(X_train, y_train)

    acc_knn = knn.score(X_test, y_test)

    print("the current accuracy of knn is "+ str(acc_knn))



if __name__ == "__main__":
    main()