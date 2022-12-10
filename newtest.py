from scipy import ndimage
import imageio
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2


def newTransform(X):
    return X - cv2.GaussianBlur(X, (23, 23), 8) + 127


img = plt.imread(
    'dataset2-master\dataset2-master\images\TEST\EOSINOPHIL\_0_187.jpeg')

plt.imshow(img)
plt.show()
# img = newTransform(img)
print('Image values: ')

image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite('sv.jpg', img)
