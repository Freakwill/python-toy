#!/usr/bin/env python

"""complete the missing pixels of an image
by NMF
"""

from PIL import Image
from utils import *
import numpy as np
from sklearn.model_selection import *


im = Image.open('../hanzi.jpeg')
im = im.crop((7,7,im.size[0]-7, im.size[1]-7))


m, n = 14, 20
width, height = im.size[0] // n, im.size[1] // m

def get_data(image, w, h, channel=0):
    data = np.asarray(image, dtype=np.float64)
    M, N = data.shape[:2]
    return np.array([data[i*h:(i+1)*h, j*w:(j+1)*w, channel].ravel() for i in range(m) for j in range(n)])

X = get_data(im, width, height)
X_train, X_test = train_test_split(X, test_size=0.05)


rate = 0.8  # proba. of missing

M = np.random.rand(*X_test.shape)>rate # random missing matrix
M = np.vstack((np.ones_like(X_train, dtype=np.bool_), M))

# mask = Image.open('/Users/william/Pictures/heart.png')
# mask = mask.resize(im.size)
# M = np.asarray(mask, dtype=np.int_) == 0
# M = np.dstack((M,M,M))
# rate = np.mean(M)

n_components = 5
nmf = IncompletedNMF(n_components=n_components)

channel = 2
mask = M
nmf.fit(X, mask=mask)
Y = nmf.complete(X, mask=mask)

def to_image(x):
    i = Image.fromarray(x.reshape((height,width)).astype('uint8'), mode='L')
    return i


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,3)

ax[0].imshow(to_image(X[-3]))
ax[1].imshow(to_image((X*M)[-3]))
ax[2].imshow(to_image(Y[-3]))

err = np.mean(np.abs(Y[-3] - X[-3])) / rate

titles = ('Original Image', f'Missing pixels ({rate:.1%})', f'Completion Image (MAE: {err:.2f})')
for a, t in zip(ax, titles):
    a.set_title(t)
    a.set_axis_off()

plt.show()
