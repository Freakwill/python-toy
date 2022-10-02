#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""VQ algorithm for image compression

Reference: https://www.sciencedirect.com/topics/engineering/vector-quantization
"""

import numpy as np
from sklearn.mixture import *
from sklearn.cluster import *
from sklearn.base import TransformerMixin

from PIL import Image

class ImageEncoder(TransformerMixin):

    def make_codebook(self):  
        if hasattr(self, 'cluster_centers_'):
            self.codebook_ = self.cluster_centers_
        elif hasattr(self, 'means_'):
            self.codebook_ = self.means_

    def _decode(self, codes):
        """Decode `codes` with the codebook
        
        Args:
            codes (array-like): array of codes
        
        Returns:
            array: decoding result
        """
        return np.array([self.codebook_[k] for k in codes])

    def decode(self, codes, shape):
        Y = self._decode(codes)
        if self.block_size == (1, 1):
            Y = Y.reshape((im.size[1], im.size[0], self.n_channels))
            return Image.fromarray(Y.astype('uint8'))
        else:
            return data2image(Y, block_size=self.block_size, shape=shape)

    def query(self, k):
        w, h = self.block_size
        return self.codebook_[k].reshape((h, w, self.n_channels))

    def encode(self, im, shape=None):
        X, shape = image2data(im, block_size=self.block_size, shape=shape)
        return self.predict(X), shape

    def reconstruct(self, im=None):
        if im is None:
            code, shape = self.labels_, self.shape
        else:
            code, shape = self.encode(im)
        return self.decode(code, shape)

    def stat(self, im, shape=None):
        Y, shape = self.encode(im, shape=shape)
        import collections
        f = collections.Counter(Y)
        N = len(Y)
        p = {k: (v / N) for k, v in f.items()}
        return f, p

# stat the pixels
# a, _ = image2data(im)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# fig.suptitle('RBG直方图及核密度估计')
# N = a.shape[0]
# for _, i in enumerate('RBG'):
#     ax = fig.add_subplot('13%d'%_)
#     ai = a[:, _]
#     ax.hist(ai, bins=50, density=True)
#     p = gaussian_kde(ai)
#     x = np.linspace(0, 255, 500)
#     ax.plot(x, p(x))
#     ax.set_title(i)
# plt.show()
# plt.savefig()

class VQTransformer(ImageEncoder, KMeans):
    """VQ algorithm for image compression

    Example:
        model = KMeans(n_clusters=2)
        im = Image.open('cloth.jpg')
        t = VQTransformer(model=model)
        t.fit(im)
        im = t.transform(im)
        im.save('cloth.kmeans2.jpg')
    """

    def __init__(self, block_size=(1, 1), n_channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.n_channels = n_channels

    def fit(self, im, shape=None, *args, **kwargs):
        # fit the image and biuld the codebook
        X, shape = image2data(im, block_size=self.block_size, shape=shape)
        self.shape = shape
        super().fit(X, *args, **kwargs)
        self.make_codebook()
        return self

    def make_codebook(self):  
        self.codebook_ = self.cluster_centers_


from semi_kmeans import *

class SemiVQTransformer(ImageEncoder, SemiKMeans):
    """Semi VQ algorithm for image compression
    """

    def __init__(self, block_size=(1, 1), n_channels=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.n_channels = n_channels

    def fit(self, im, mask, shape=None, *args, **kwargs):
        X, shape = image2data(im, block_size=self.block_size, shape=shape)
        self.shape = shape
        _mask = mask.ravel()
        y0 = _mask[_mask != -1]
        X0 = X[_mask!=-1]
        X1 = X[_mask==-1]
        super().fit(X0, y0, X1, *args, **kwargs)
        self.codebook_ = self.cluster_centers_
        n0 = y0.shape[0]
        self.labels_[_mask==-1], self.labels_[_mask!=-1] = self.labels_[n0:], y0
        return self


def image2data(im, block_size=(1, 1), shape=None):
    """Transform an image to 2d array
    where block_size = (w, h)
    
    Arguments:
        im {Image} -- an image
    
    Keyword Arguments:
        block_size {tuple} -- a rectangle domain of an image (default: (1, 1))
                        block_size[0] * block_size[1] is the number of features
        shape {tuple} -- size of block matrix (default: {None})
                        calculated based on the size of the image by default
                        shape[0]*shape[1] is the number of samples
    
    Returns:
        tuple of a block matrix and its shape=(r, c)
    """
    w, h = block_size
    if shape is None:
        W, H = im.size
        c, cc = divmod(W, w)
        r, rr = divmod(H, h)
        im = im.resize((w*c, h*r))
        shape = r, c
    else:
        r, c = shape
        im = im.resize((0, 0, w*c, h*r))
    I = np.asarray(im, dtype=np.float64)
    # blocks <- concatenate the columns of the block matrix
    blocks = [I[i*h:(i+1)*h, j*w:(j+1)*w].ravel() for i in range(r) for j in range(c)]
    return np.array(blocks), shape

# def image2data(im, block_size=(1,1), shape=None):
#     blocks, shape = image2blockmatrix(im, block_size=(1,1), shape=None)
#     data = np.array([blocks[i,j].ravel() for j in range(c) for i in range(r)])
#     return data, shape

def data2image(X, block_size, shape):
    # inverse of image2data
    r, c = shape
    w, h = block_size
    X = X.reshape((r, c, w*h*3))
    X = np.block([[[X[i, j].reshape((h, w, 3))] for j in range(c)] for i in range(r)])
    return Image.fromarray(X.astype('uint8', copy=False))
    

# def vq(im, block_size=(2,2), model=None):
#     # vector quantization with a clustering model
#     if model is None:
#         model = KMeans(n_clusters=2)

#     X, shape = image2data(im, block_size=block_size)

#     model.fit(X)
#     code = model.predict(X)  # encoding
#     if hasattr(model, 'cluster_centers_'):
#         codebook = model.cluster_centers_
#     elif hasattr(model, 'means_'):
#         codebook = model.means_
#     Y = np.array([model.codebook[k] for k in code]) 
    
#     return data2image(Y, block_size=block_size, shape=shape)


if __name__ == '__main__':

    im = Image.open('lenna.jpeg')

    block_size = 8, 8
    n_clusters = 5
    t = VQTransformer(block_size=block_size, n_clusters=n_clusters)
    t.fit(im)
    im0 = t.reconstruct()

    def _mask(im, block_size=(1,1), shape=None):
        """Transform an image to 2d array
        where block_size = (w, h)
        
        Arguments:
            im {Image} -- an image
        
        Keyword Arguments:
            block_size {tuple} -- a rectangle domain of an image (default: (1, 1))
            shape {tuple} -- size of small domain (default: {None})
                            calculated based on the size of the image by default
        
        Returns:
            tuple of a block matrix and its shape=(r, c)
        """
        w, h = block_size
        if shape is None:
            W, H = im.size
            c, cc = divmod(W, w)
            r, rr = divmod(H, h)
            shape = r, c
            im = im.resize((w*c, h*r))
        else:
            r, c = shape
            im = im.resize((0, 0, w*c, h*r))
        I = np.asarray(im, dtype=np.float64)
        shape = (r, c)
        # blocks <- concatenate the columns of the block matrix
        def _t(a):
            def _f(x):
                if np.all(x == (255,0,0)): return 0
                elif np.all(x == (0,255,0)): return 1
                elif np.all(x == (0,0,255)): return 2
                else: return -1
            return np.max(np.apply_along_axis(_f, 2, a))
        blocks = [[_t(I[i*h:(i+1)*h, j*w:(j+1)*w]) for j in range(c)] for i in range(r)]
        return np.array(blocks)

    # Semi VQ
    t = SemiVQTransformer(block_size=block_size, n_clusters=n_clusters)
    masked = Image.open('lenna-masked.jpg')
    mask = _mask(masked, block_size=block_size)
    t.fit(im, mask=mask)
    im1 = t.reconstruct()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2)

    ax[0,0].imshow(im)
    ax[0,0].set_title('Original Image')
    ax[0,1].imshow(masked)
    ax[0,1].set_title('Masked Image')
    ax[1,0].imshow(im0)
    ax[1,0].set_title('VQ')
    ax[1,1].imshow(im1)
    ax[1,1].set_title('Semisupervised VQ')

    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')

    plt.show()
