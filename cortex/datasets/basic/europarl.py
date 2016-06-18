"""
Europarl dataset for machine translation.

Currently only supports fr-en datasets.
"""

from .. import BasicDataset, make_one_hot
import string
import numpy as np
from collections import defaultdict
from functools import partial
import logging
from guppy import hpy


class Europarl(BasicDataset):
    """
    Europarl dataset itterator.
    """
    def __init__(self, source=None, mode='train', english_to_french=True,
                 name='europarl', out_path=None, max_words=5000,
                 max_sentence=30, max_length=7000, **kwargs):

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.logger.info('Loading %s from %s' % (name, source))

        if source is None:
            raise ValueError('No source file provided.')
        print 'Loading {name} ({mode}) from {source}'.format(
            name=name, mode=mode, source=source)

        self.masken = None
        self.maskfr = None
        self.masky = None
        self.maskx = None
        self.max_sentence = max_sentence
        self.max_length = max_length
        self.max_words = max_words
        self.n_observations = 0
        self.english_to_french = english_to_french
        X, Y = self.get_data(source)
        data = {name: X, 'label': Y}
        distributions = {name: 'multinomial', 'label': 'multinomial'}

        super(Europarl, self).__init__(data, distributions=distributions,
                                       name=name, mode=mode, **kwargs)

        self.out_path = out_path

        if self.shuffle:
            self.randomize()

    def slice_data(self, idx, data=None):  # Function for restricting dataset in instance.
        if data is None: data = self.data
        for k, v in data.iteritems():
            self.data[k] = v[idx]
        self.n_observations = len(idx)
        self.X = data[self.name]
        if self.labels in data.keys():
            self.Y = data[self.labels]
        self.n = self.X.shape[0]

    def get_data(self, source):
        fr = open(source + 'europarl-v7.fr-en.fr') #### NOT SURE IF SOURCING IS CORRECT
        en = open(source + 'europarl-v7.fr-en.en')

        X = []
        Y = []
        fMax = 0
        eMax = 0
        self.itt_pos1 = 4
        self.itt_pos2 = 4
        self.frStringToToken = defaultdict(partial(self.count1, self.max_words, 3))
        self.enStringToToken = defaultdict(partial(self.count2, self.max_words, 3))
        special_tokens = {'<PAD>': 0, '<BEG>': 1, '<END>': 2, '<UNK>': 3}

        i = 0
        for eSentence, fSentence in zip(en.__iter__(), fr.__iter__()):  # Itterate through file lines
            if len(self.string_process(eSentence)) <= self.max_sentence\
                    and len(self.string_process(fSentence)) <= self.max_sentence:
                X.append([self.enStringToToken[eWord] for eWord in self.string_process(eSentence)])  # Convert to numerical
                if len(X[-1]) > eMax:  # Keep track of largest sentence in language.
                    eMax = len(X[-1])
                Y.append([self.frStringToToken[fWord] for fWord in self.string_process(fSentence)])
                if len(Y[-1]) > fMax:
                    fMax = len(Y[-1])
                i += 1
                if i >= self.max_length:
                    break

        fr.close()
        en.close()
        del fr
        del en

        print 'Data loaded, preprocessing...'
        print 'Padding data.'
        self.n_observations = len(X)  # Update sample size

        X = map(partial(self.pad_array, length=(eMax + 2)), X)
        Y = map(partial(self.pad_array, length=(fMax + 2)), Y)

        print 'Creating masks.'
        self.masken = map(self.create_mask, X)
        self.maskfr = map(self.create_mask, Y)

        self.masken = np.array(self.masken, dtype='float32')
        self.maskfr = np.array(self.maskfr, dtype='float32')

        print 'Converting to one-hot.'
        # The following couple lines are really slow to run.
        X = make_one_hot(np.array(X).reshape((eMax + 2) * self.n_observations))\
            .reshape((self.n_observations, eMax + 2, max(self.enStringToToken.values()) + 1))  # Convert to one hot, (array -> vector -> one-hot -> array)
        Y = make_one_hot(np.array(Y).reshape((fMax + 2) * self.n_observations))\
            .reshape((self.n_observations, fMax + 2, max(self.frStringToToken.values()) + 1))

        print 'Converting to float32.'
        #  Conversion after one-hot as float32 arrays slow down one-hot conversion.
        X = X.astype('float32')
        Y = Y.astype('float32')

        self.frStringToToken.update(special_tokens)
        self.enStringToToken.update(special_tokens)

        print 'Data prepared.'
        if self.english_to_french:
            self.maskx = self.masken
            self.masky = self.maskfr
            return X, Y
        else:
            self.maskx = self.maskfr
            self.masky = self.masken
            return Y, X

    @staticmethod
    def factory(C=None, split=None, idx=None, batch_sizes=None, **kwargs):
        if C is None:
            C = Europarl
        europarl = C(batch_size=10, **kwargs)
        if hasattr(europarl, 'logger'):
            logger = europarl.logger
            europarl.logger = None
        else:
            logger = logging.getLogger('.'.join([europarl.__module__, europarl.__class__.__name__]))

        if idx is None:
            logger.info('Splitting dataset into ratios %r' % split)
            if round(np.sum(split), 5) != 1. or len(split) != 3:
                raise ValueError(split)

            if europarl.balance:
                raise NotImplementedError()
            else:
                split_idx = []
                accum = 0
                for s in split:  # Create indicies from percentage values
                    s_i = int(s * europarl.n_observations + accum)
                    split_idx.append(s_i)
                    accum += s_i
                idx = range(europarl.n_observations)

                train_idx = idx[:split_idx[0]]
                valid_idx = idx[split_idx[0]:split_idx[1]]
                test_idx = idx[split_idx[1]:]
            idx = [train_idx, valid_idx, test_idx]
        else:
            logger.info('Splitting dataset into ratios  %.2f / %.2f /%.2f '
                        'using given indices'
                        % tuple(len(idx[i]) / float(europarl.n_observations)
                                for i in range(3)))

        assert len(batch_sizes) == len(idx)  # Shouldn't have different number of batch sizes than datasets

        datasets = []
        modes = ['train', 'valid', 'test']
        data = europarl.data
        europarl.data = dict()
        for i, bs, mode in zip(idx, batch_sizes, modes):  # Create correctly restricted copies of dataset
            if bs is None:
                dataset = None
            else:
                dataset = europarl.copy()
                dataset.slice_data(i, data=data)
                dataset.batch_size = bs
                dataset.logger = logger
                dataset.mode = mode
            datasets.append(dataset)

        return datasets + [idx]

    table = string.maketrans('', '')  # Variable for string_process()

    def string_process(self, s):  # Helper method for get_data()
        return s.lower().translate(self.table, string.punctuation).split()

    @staticmethod
    def pad_array(arr, length):  # Helper method for get_data()
        return [1] + arr + [2] + ([0]*(length - len(arr) - 2))

    @staticmethod  # Helper method for creating mask array from a list.
    def create_mask(lst):
        return np.array([int(bool(x)) for x in lst])

    def count1(self, max_words, unknown_val):  # Crappy replacement for count as itterators can not be copied. :(
        if self.itt_pos1 <= max_words:
            self.itt_pos1 += 1
            return self.itt_pos1 - 1
        else:
            return unknown_val

    def count2(self, max_words, unknown_val):
        if self.itt_pos2 <= max_words:
            self.itt_pos2 += 1
            return self.itt_pos2 - 1
        else:
            return unknown_val

def count(start, max_words, unknown_val):  # Helper method for defaultdict in get_data()
    s = start
    while True:
        if s <= max_words:
            yield s
            s += 1
        else:
            yield unknown_val


def main():
    #data = Europarl(source='/export/mialab/users/jjohnson/data/basic/', batch_size=10)
    train, valid, test, idx = Europarl.factory(source='/export/mialab/users/jjohnson/data/basic/',
                                               batch_sizes=[100, 100, 100], split=[0.7, 0.2, 0.1])
    h = hpy()
    print h.heap()
    print train.data['europarl'].shape
    print valid.data['europarl'].shape
    print test.data['europarl'].shape

main()
