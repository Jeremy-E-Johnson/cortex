import cPickle
from glob import glob
import gzip
import multiprocessing as mp
import numpy as np
from os import path
import PIL
import random
import sys
from sys import stdout
import theano
from theano import tensor as T
import traceback

from mnist import Chains
from utils.vis_utils import tile_raster_images


def reshape_image(img, shape, crop_image=True):
    if crop_image:
        bbox = img.getbbox()
        img = img.crop(bbox)

    img.thumbnail(shape, PIL.Image.BILINEAR)
    new_img = PIL.Image.new('L', shape)
    offset_x = max((shape[0] - img.size[0]) / 2, 0)
    offset_y = max((shape[1] - img.size[1]) / 2, 0)
    offset_tuple = (offset_x, offset_y)
    new_img.paste(img, offset_tuple)
    return new_img

class Horses(Chains):
    def __init__(self, batch_size=10, source=None,
                 inf=False, chain_length=97, chain_build_batch=97, window=7,
                 stop=None, out_path=None, dims=None, chain_stride=None, shuffle=True,
                 crop_image=True, n_chains=1):
        # load MNIST
        assert source is not None

        self.dims = dims

        data = []
        for f in glob(path.join(path.abspath(source), '*.png')):
            img = PIL.Image.open(f)
            if self.dims is None:
                self.dims = img.size
            img = reshape_image(img, self.dims, crop_image=crop_image)
            data.append(np.array(img))
        self.dims = self.dims[1], self.dims[0]

        X = np.array(data).astype('float32')
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X = (X - X.min()) / float(X.max() - X.min())

        self.f_energy = None
        self.chain_length = chain_length
        self.window = window
        self.chains_build_batch = chain_build_batch
        self.bs = batch_size

        self.out_path = out_path
        if chain_stride is None:
            self.chain_stride = self.chain_length
        else:
            self.chain_stride = chain_stride

        self.shuffle = shuffle

        if stop is not None:
            X = X[:stop]

        self.n, self.dim = X.shape
        self.chain_length = min(self.chain_length, self.n)
        self.chains = [[] for _ in xrange(n_chains)]
        self.chain_pos = 0
        self.pos = 0
        self.chain_idx = range(0, self.chain_length - window, self.chain_stride)
        self.spos = 0

        self.X = X

        print 'Shuffling horses (neh!)'
        self.randomize()

    def next_simple(self, batch_size=10):
        cpos = self.spos
        if cpos + batch_size > self.n:
            self.spos = 0
            cpos = self.spos
            if self.shuffle:
                self.randomize()

        x = self.X[cpos:cpos+batch_size]
        self.spos += batch_size

        return x

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]


class SimpleHorses(object):
    def __init__(self, batch_size=10, source=None, inf=False, stop=None,
                 dims=None):
        # load MNIST
        assert source is not None

        self.dims = dims

        data = []
        for f in glob(path.join(path.abspath(source), '*.png')):
            img = PIL.Image.open(f)
            if self.dims is None:
                self.dims = img.size
            img = reshape_image(img, self.dims)
            data.append(np.array(img))
        self.dims = self.dims[1], self.dims[0]

        X = np.array(data)
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        X = (X - X.min()) / (X.max() - X.min())

        self.n = X.shape[0]
        if stop is not None:
            X = X[:stop]
            self.n = stop
        self.dim = X.shape[1]

        self.pos = 0
        self.bs = batch_size
        self.inf = inf
        self.X = X
        self.next = self._next

        # randomize
        print 'Shuffling horses'
        self.randomize()

    def __iter__(self):
        return self

    def randomize(self):
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]

    def next(self):
        raise NotImplementedError()

    def _next(self, batch_size=None):
        if batch_size is None:
            batch_size = self.bs

        cpos = self.pos
        if cpos + batch_size > self.n:
            # reset
            self.pos = 0
            cpos = self.pos
            if self.shuffle:
                self.randomize()
            if not self.inf:
                raise StopIteration

        x = self.X[cpos:cpos+batch_size]

        self.pos += batch_size

        return x, None

    def save_images(self, x, imgfile, transpose=False, x_limit=None):
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], 1, x.shape[1]))

        if x_limit is not None and x.shape[0] > x_limit:
            x = np.concatenate([x, np.zeros((x_limit - x.shape[0] % x_limit,
                                             x.shape[1],
                                             x.shape[2])).astype('float32')],
                axis=0)
            x = x.reshape((x_limit, x.shape[0] * x.shape[1] // x_limit, x.shape[2]))

        tshape = x.shape[0], x.shape[1]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        image = self.show(x.T, tshape, transpose=transpose)
        #print 'Saving to ', imgfile
        image.save(imgfile)

    def show(self, image, tshape, transpose=False):
        fshape = self.dims
        if transpose:
            X = image
        else:
            X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))
