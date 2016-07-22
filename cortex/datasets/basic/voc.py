"""
Data itteratoor for VOC classification data.
"""

from .. import BasicDataset
from os import path
import logging
from ...utils.tools import resolve_path
from PIL import Image
import PIL
import random
import numpy as np


class VOC(BasicDataset):
    """Dataset iterator for VOC classification data. (Designed for use with Pyramid RNNs)

    Attributes:

    """

    def __init__(self, images_loaded=10, chunk_size=5, out_path=None, chunks=1000,
                 start_image=0, mode='train', source=None, name='voc', **kwargs):
        """

        Args:
            images_loaded (int): How many images to load
            chunk_size (int): Dimension of chunks to be made.
            mode (str): Type of data to load, train, valid, test.
            source (str): Path to directory containing VOCdevkit
            name: Name of iterator
            **kwargs:
        """

        self.mode_resolve = {'train': 'train', 'valid': 'trainval', 'test': 'val'}
        self.mode = self.mode_resolve[mode]

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.logger.info('Loading %s from %s as %s' % (name, source, self.mode))

        if source is None:
            raise ValueError('No source file provided.')
        source = resolve_path(source)

        self.start_image = start_image
        self.chunks = chunks
        self.images_loaded = images_loaded
        if chunk_size % 2:
            self.chunk_size = chunk_size
        else:
            self.logger.info('Using %d + 1 to get an odd chunk size.' % chunk_size)
            self.chunk_size = chunk_size + 1

        X, Y = self.get_data(source, self.mode)
        data = {name: X, 'label': Y}
        distributions = {name: 'multinomial', 'label': 'multinomial'}

        super(VOC, self).__init__(data, distributions=distributions,
                                  name=name, **kwargs)

        self.out_path = out_path

        if self.shuffle:
            self.randomize()

    @staticmethod
    def factory(split=None, idx=None, batch_sizes=None, **kwargs):
        if split is None:
            raise NotImplementedError('Idx are not supported for this dataset yet.')
        if batch_sizes is None:
            raise ValueError('Need batch sizes')

        chunks = kwargs['chunks']
        chunk_ammounts = []
        for val in split:
            chunk_ammounts.append(int(chunks * val))

        train = VOC(images_loaded=10, start_image=0, chunk_size=kwargs['chunk_size'],
                    chunks=chunk_ammounts[0], mode='train', source=kwargs['source'],
                    batch_size=batch_sizes[0])
        valid = VOC(images_loaded=5, start_image=10, chunk_size=kwargs['chunk_size'],
                    chunks=chunk_ammounts[1], mode='valid', source=kwargs['source'],
                    batch_size=batch_sizes[1])
        test = VOC(images_loaded=5, start_image=15, chunk_size=kwargs['chunk_size'],
                   chunks=chunk_ammounts[2], mode='test', source=kwargs['source'],
                   batch_size=batch_sizes[2])

        accum = 0
        idx = []
        for val in chunk_ammounts:
            idx.append(range(accum, accum + val))
            accum += val

        return train, valid, test, idx

    def get_data(self, source, mode):
        """Gets data given source, chunks it, and returns chunks with center labels.

        Args:
            source (str): File path to directory containing VOCdevkit.
            mode (str): Mode of data, eg. train, valid, test.

        Returns:

        """
        rand = random.Random()
        buff_dist = (self.chunk_size + 1)/2

        def get_unique(pixels):
            """Helper function for get_data, returns the number of unique classifiers in an image.

            Args:
                pixels (list of lists): Pixel classifier values.

            Returns (int): Number of unique classifiers in image.

            """
            unique = []
            for l in pixels:
                for j in l:
                    if j not in unique:
                        unique.append(j)
            return len(unique)

        def image_to_pixels(im):
            """

            Args:
                im (Image): Image object form PIL

            Returns (list of lists): Pixels

            """
            pixels = list(im.getdata())
            width, height, = im.size
            return [pixels[i * width:(i + 1) * width] for i in xrange(height)]

        def project_to_binary(pixels):
            """Helper function for get_data, returns binary version of input pixels.

            Args:
                pixels (list of lists of ints): pixels of an image.

            Returns: Pixels projected to binary.

            """
            retval = []
            for ln in pixels:
                retval.append([int(bool(val)) for val in ln])
            return retval

        def get_random_chunk(pixels_data, pixels_label):
            """Helper function for get_data, gets random chunk from data, and returns label for center.

            Args:self, VOC
                pixels_data (list of lists): Image pixels of data.
                pixels_label (list of lists): Image pixels of label.

            Returns: data_chunk (list of lists subsection of pixels_data), label_val (value of label at center of chunk)

            """
            y = rand.randint(buff_dist, len(pixels_data) - buff_dist)
            x = rand.randint(buff_dist, len(pixels_data[0]) - buff_dist)
            data_chunk = []
            label_val = pixels_label[y][x]
            for index in range(y - buff_dist + 1, y + buff_dist):
                data_chunk.append(pixels_data[index][x - buff_dist + 1:x + buff_dist])
            assert len(data_chunk) == self.chunk_size and len(data_chunk[0]) == self.chunk_size
            return data_chunk, label_val

        names = []
        with open(source + '/basic/VOCdevkit/VOC2010/ImageSets/Segmentation/' + mode + '.txt') as f:
            for line in f:
                names.append(line[:-1])

        self.data_images = []
        self.label_images = []
        images_loaded = 0
        for name in names:
            if images_loaded < (self.images_loaded + self.start_image) and images_loaded >= self.start_image:
                label_im = Image.open(source + '/basic/VOCdevkit/VOC2010/SegmentationObject/' + name + '.png')
                label_pixels = image_to_pixels(label_im)
                if get_unique(label_pixels) == 3:
                    self.label_images.append(project_to_binary(label_pixels))

                    data_im = Image.open(source + '/basic/VOCdevkit/VOC2010/JPEGImages/' + name + '.jpg').convert('L')
                    self.data_images.append(image_to_pixels(data_im))

                    images_loaded += 1
            elif images_loaded < self.start_image:
                images_loaded += 1
            else:
                break

        X = []
        Y = []
        for i in xrange(0, self.chunks):
            k = rand.randint(0, len(self.data_images) - 1)
            x, y = get_random_chunk(self.data_images[k], self.label_images[k])
            X.append(np.array(x, dtype='float32')/255.0)  # Normalize
            Y.append(np.array(y, dtype='float32'))

        assert len(X) == self.chunks and len(Y) == self.chunks

        return np.array(X), np.array(Y)

    def next(self):
        rval = super(VOC, self).next()

        rval['label'] = np.array([b[1] for b in rval['label']])

        return rval
