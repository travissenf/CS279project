from scipy import ndimage
import imageio
import numpy as np
import matplotlib.pyplot as plt
import sys

def transform(X):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            res  = (0.2126*X[i, j, 0] + 0.7152*X[i, j, 1] + 0.0722*X[i,j,1])
            if res <= 20:
                X[i,j] = 255
            else: 
                X[i, j] = ((0.2126*X[i, j, 0] + 0.7152*X[i, j, 1] + 0.0722*X[i,j,2]) > 120) * 255
    return X
img = plt.imread('dataset2-master\dataset2-master\images\TEST\EOSINOPHIL\_0_187.jpeg')
plt.figure('orig')
plt.imshow(img)
img = transform(img)
print('Image values: ')

plt.rcParams['image.cmap']='jet'
plt.figure('new')
plt.imshow(img)
plt.colorbar()
plt.show()
