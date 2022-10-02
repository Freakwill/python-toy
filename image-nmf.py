#!/usr/bin/env python

"""complete the missing pixels of an image
by NMF
"""

from PIL import Image
from wnmf import *
import numpy as np

im = Image.open('/Users/william/Pictures/lenna.jpg')
X = np.asarray(im, dtype=np.float_)

rate = 0.97  # proba. of missing
M = np.random.rand(*X.shape)>rate # random missing matrix

# mask = Image.open('/Users/william/Pictures/heart.png')
# mask = mask.resize(im.size)
# M = np.asarray(mask, dtype=np.int_) == 0
# M = np.dstack((M,M,M))
# rate = np.mean(M)

n_components = 40
nmf = IncompletedNMF(n_components=n_components, init_method='mean', max_iter=10, mu_alpha=0.01)
nmf_ = MyNMF(n_components=n_components)
Y = []
Y1 = []
ss = []
for channel in range(3):
    mask = M[:,:,channel]

    nmf.fit(X[:,:,channel], mask=mask)
    Y.append(nmf.complete(X[:,:,channel], mask=mask, to_int='p'))
    Y1.append(nmf.init_complete(X[:,:,channel], mask=mask, method='mean'))
    nmf.init_completion_ = None
    ss.append(nmf.significance_)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
for s in ss:
    ax.plot(np.cumsum(s))
plt.show()

Y = np.dstack(Y)
Y1 = np.dstack(Y1)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(im)
X_ = X * M
im_ = Image.fromarray(X_.astype('uint8'))
ax[0,1].imshow(im_)
ax[1,0].imshow(Image.fromarray(Y.astype('uint8')))
ax[1,1].imshow(Image.fromarray(Y1.astype('uint8')))

err = np.mean(np.abs(Y - X)) / rate
err1 = np.mean(np.abs(Y1 - X)) / rate


titles = ('Original Image', f'Missing pixels ({rate:.1%})',
    f'Completion by NMF (MAE: {err:.2f})', f'by mean (MAE: {err1:.2f})')
for a, t in zip(ax.ravel(), titles):
    a.set_title(t)
    a.set_axis_off()

plt.show()
