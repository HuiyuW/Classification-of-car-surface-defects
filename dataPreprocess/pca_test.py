from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from dataPreprocess import dataPreprocess

import pandas as pd


def main():
    dp = dataPreprocess('annotated_functional_test3_fixed.json', 255, 236)
    #dp.fetchImages()
    # get X value
    X = dp.fetchgetXvalue()

    print("the size of input images are"+str(X.shape))

    dataLabel = pd.read_csv("label0-896.csv")

    y = dataLabel.loc[:, "human_label"]
    pca = PCA(n_components=3)
    X_r = pca.fit_transform(X)

    evr1 = pca.explained_variance_ratio_
    print(evr1)
    print(X_r.shape)

    plt.figure("test1")
    plt.axis('on')
    for i in range(0, X.shape[0]):
        plt.scatter(X_r[i, 0], X_r[i, 1], c=np.array(plt.cm.tab10(dataLabel.loc[i, "human_label"])).reshape(1,-1), label=dataLabel.loc[i, "label_name"],  s=20, alpha=0.9, marker='o')


    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()