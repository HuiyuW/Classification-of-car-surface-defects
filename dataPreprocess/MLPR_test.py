import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import sys
import os
from PIL import Image

from dataPreprocess import dataPreprocess
"""
tem = []
# query all images in file annotated images
annotImagepath = "../Annotated_images"
for imageName in os.listdir(annotImagepath):
    #print(imageName)
    curFullpath = annotImagepath + "/" + imageName
    curImg = Image.open(curFullpath)
    arImg = np.asarray(curImg).flatten()/255
    tem.append(arImg)

X = np.array(tem)
"""
dp = dataPreprocess('annotated_functional_test3_fixed.json', 28, 28)
#dp.fetchImages()
# get X value
X = dp.fetchgetXvalue()

X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)


# Shape of input and latent variable

n_input = 28*28

# Encoder structure
n_encoder1 = 500
n_encoder2 = 300

n_latent = 2

# Decoder structure
n_decoder2 = 300
n_decoder1 = 500

reg = MLPRegressor(hidden_layer_sizes = (n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1), 
                   activation = 'tanh', 
                   solver = 'adam', 
                   learning_rate_init = 0.0001, 
                   max_iter = 20, 
                   tol = 0.0000001, 
                   verbose = True)

reg.fit(X_train, X_train)

score = reg.score(X_train, X_train)
print("the current accuracy of Multilayer Perceptron is "+str(score))

idx = np.random.randint(X_test.shape[0])
x_reconst = reg.predict(X_test[idx].reshape(-1,784))



def encoder(data):
    data = np.asmatrix(data)
    
    encoder1 = data*reg.coefs_[0] + reg.intercepts_[0]
    encoder1 = (np.exp(encoder1) - np.exp(-encoder1))/(np.exp(encoder1) + np.exp(-encoder1))
    
    encoder2 = encoder1*reg.coefs_[1] + reg.intercepts_[1]
    encoder2 = (np.exp(encoder2) - np.exp(-encoder2))/(np.exp(encoder2) + np.exp(-encoder2))
    
    latent = encoder2*reg.coefs_[2] + reg.intercepts_[2]
    latent = (np.exp(latent) - np.exp(-latent))/(np.exp(latent) + np.exp(-latent))
    
    return np.asarray(latent)

test_latent = encoder(X_train)
#print(test_latent.shape)
#print(test_latent)

plt.figure("test")
plt.axis('on')
plt.scatter(test_latent[:, 0], test_latent[:, 1], alpha=.8)
plt.show()


