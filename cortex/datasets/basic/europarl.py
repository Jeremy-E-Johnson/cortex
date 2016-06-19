'''
Europarl dataset for machine translation.

Currently only supports fr-en datasets.
'''

from collections import defaultdict
from functools import partial
from guppy import hpy
import logging
import numpy as np
from os import path
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    Timer
)
import string

from .. import BasicDataset, make_one_hot
from ...utils import floatX, intX


np.set_printoptions(threshold=np.nan)
logger = logging.getLogger(__name__)


class Europarl(BasicDataset):
    '''Europarl dataset itterator.

    Attributes:
        max_sentence (int): Maximum sentence length.
        max_length (int): Maximum number of sentences.
        max_words (int): Maximum size of vocabulary.
        english_to_french (bool): If true English is under name key, and French under label key, else reversed.
        debug (bool): If true restricts max_length to 1000.

    '''
    _PAD = 0
    _BEG = 1
    _END = 2
    _UNK = 3
    table = string.maketrans('', '')

    def __init__(self, source=None, english_to_french=True,
                 name='europarl', out_path=None, max_words=5000,
                 max_sentence=30, max_length=7000, debug=False, **kwargs):
        """
        Args:
            source (str): Path to where the europarl data is stored.
            english_to_french (bool): True for English input French labels, False for reverse.
            name (str): Name of dataset.
            out_path (str): Path to save outs.
            max_words (int): Maximum vocab size, extra words are marked unknown.
            max_sentence (int): Maximum sentence length, longer sentences are ignored.
            max_length (int): Maximum number of sentences.
            debug (bool): If True restricts max_length to 1000.
            **kwargs:
        """

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.logger.info('Loading %s from %s' % (name, source))

        if source is None:
            raise ValueError('No source file provided.')

        self.max_sentence = max_sentence
        self.max_length = max_length
        self.max_words = max_words
        self.english_to_french = english_to_french

        if debug:
            self.max_length = 1000

        X, Y, Mx, My = self.get_data(source)
        data = {name: X,
                'label': Y,
                'mask_in': Mx,
                'mask_out': My}
        distributions = {name: 'multinomial',
                         'label': 'multinomial',
                         'mask_in': None,
                         'mask_out': None}

        super(Europarl, self).__init__(data, distributions=distributions,
                                       name=name, one_hot=False, **kwargs)

        self.out_path = out_path

        if self.shuffle:
            self.randomize()

    def slice_data(self, idx, data=None):
        '''Function for restricting dataset in instance.

        Args:
            idx (list): Indices of data to be kept.
            data (dict): Data to be sliced and kept.

        '''
        if data is None: data = self.data
        for k, v in data.iteritems():
            self.data[k] = v[idx]

    def get_data(self, source):
        special_tokens = {
            '<PAD>': self._PAD, '<BEG>': self._BEG,
            '<END>': self._END, '<UNK>': self._UNK}

        def preprocess(s):
            '''Preprocesses string.

            Args:
                s (str): string to be preprocessed.

            Returns:
                str: preprocessed string.

            '''
            return s.lower().translate(self.table, string.punctuation).split()

        def make_dictionary(sentences, n_lines, max_words=None):
            '''Forms a dictionary from words in sentences.

            If there are more words than max_words, use the top frequent ones.

            Args:
                sentences (file Handle)
                n_lines (int): number of lines in file.
                max_words (Optional[int]): maximum number of words. Default
                    is self.max_words.

            Returns:
                dict: word string to token dictionary.
                int: maximum length sentence.

            '''
            self.logger.info('Forming dictionary')
            if max_words is None: max_words = self.max_words

            count_dict = defaultdict(int)

            widgets = ['Counting words', ' (', Timer(), ') [', Percentage(), ']']
            pbar = ProgressBar(widgets=widgets, maxval=n_lines).start()

            max_len = 0
            for i, sentence in zip(range(0, n_lines), sentences):
                ps = preprocess(sentence)
                l = len(ps)
                if l <= self.max_sentence:
                    for word in ps:
                        count_dict[word] += 1
                    max_len = max(l, max_len)
                pbar.update(i)

            count_keys_sorted = sorted(
                count_dict, key=count_dict.get, reverse=True)
            vals_sorted = sorted(count_dict.values(), reverse=True)
            keys = count_keys_sorted[:max_words]
            omit_freq = sum(vals_sorted[max_words:]) / float(sum(vals_sorted))
            self.logger.info('Setting %d words as <UNK> with total frequency '
                             '%.3g.'
                             % (max(0, len(count_keys_sorted) - max_words),
                                omit_freq))
            values = range(4, len(keys) + 4)

            d = dict()
            d.update(**special_tokens)
            d.update(**dict(zip(keys, values)))
            return d, max_len

        def tokenize(sentence, d, pad_length):
            '''Tokenize sentence using dictionary.

            If sentence is longer than max_sentence, returns [].

            Args:
                sentence (str): sentence to be tokenized.
                d (dict): token dictionary.
                pad_length (int): total length up to pad.

            Returns:
                list: tokenized sentence as list.

            '''
            ps = preprocess(sentence)
            if len(ps) > self.max_sentence:
                return []
            s = [self._BEG] + [d.get(w, self._UNK) for w in ps] + [self._END]
            s += [self._PAD] * max(0, pad_length + 2 - len(s))
            return s

        def read_and_tokenize(file_path, max_length):
            '''Read and tokenize a file of sentences.

            Args:
                file_path (str): path to file.
                max_length (int): maximum number of lines to read.

            Returns:
                list: list of tokenized sentences.
                dict: token disctionary.
                dict: reverse dictionary.

            '''
            self.logger.info('Reading sentences from %s' % file_path)
            with open(file_path) as f:
                n_lines = min(sum(1 for line in f), max_length)
                f.seek(0)
                d, max_len = make_dictionary(f, n_lines)
                r_d = dict((v, k) for k, v in d.iteritems())
                tokenized_sentences = []

                f.seek(0)
                self.logger.info('Tokenizing sentences from %s' % file_path)
                widgets = ['Tokenizing sentences' ,
                           ' (', Timer(), ') [', Percentage(), ']']
                pbar = ProgressBar(widgets=widgets, maxval=n_lines).start()
                for i, sentence in zip(range(0, n_lines), f):
                    ts = tokenize(sentence, d, max_len)
                    assert len(ts) <= self.max_sentence + 2, (ts, len(ts))
                    tokenized_sentences.append(ts)
                    pbar.update(i)
            return tokenized_sentences, d, r_d

        def match_and_trim(sentences_a, sentences_b):
            '''Matches 2 lists of sentences and removes incomplete pairs.

            If one of the pairs is `[]`, remove pair.

            Args:
                sentences_a (list).
                sentences_b (list).

            Returns:
                list: new sentences_a
                list: new sentences_b

            '''
            self.logger.info('Matching datasets and trimming')
            if len(sentences_a) != len(sentences_b):
                raise TypeError('Sentence lists are different lengths.')

            sentences_a_tr = []
            sentences_b_tr = []
            widgets = ['Matching sentences',
                       ' (', Timer(), ') [', Percentage(), ']']
            trimmed = 0
            pbar = ProgressBar(widgets=widgets, maxval=len(sentences_a)).start()
            for i, (s_a, s_b) in enumerate(zip(sentences_a, sentences_b)):
                if len(s_a) > 0 and len(s_b) > 0:
                    sentences_a_tr.append(s_a)
                    sentences_b_tr.append(s_b)
                else:
                    trimmed += 1
                pbar.update(i)
            self.logger.debug('Trimmed %d sentences' % trimmed)

            return sentences_a_tr, sentences_b_tr

        fr_sentences, self.fr_dict, self.fr_dict_r = read_and_tokenize(
            path.join(path.join(source, 'europarl-v7.fr-en.fr')), self.max_length)

        en_sentences, self.en_dict, self.en_dict_r = read_and_tokenize(
            path.join(path.join(source, 'europarl-v7.fr-en.en')), self.max_length)

        fr_sentences, en_sentences = match_and_trim(fr_sentences, en_sentences)

        if self.english_to_french:
            X = np.array(en_sentences).astype(intX)
            Y = np.array(fr_sentences).astype(intX)
        else:
            X = np.array(fr_sentences).astype(intX)
            Y = np.array(en_sentences).astype(intX)

        self.nX_tokens = len(np.unique(X).tolist())
        self.nY_tokens = len(np.unique(Y).tolist())

        self.logger.info('Creating masks')
        Mx = (X != 0).astype(intX)
        My = (Y != 0).astype(intX)

        return X, Y, Mx, My

    @staticmethod
    def factory(C=None, split=None, idx=None, batch_sizes=None, **kwargs):
        '''

        Args:
            C: Data iterator to use, defaults to Europarl.
            split: List of percentage values for train, valid, and test datasets respectively.
            idx: List of indices for train, valid and test datasets respectively.
            batch_sizes: List of batch sizes for train, valid, and test datasets respectively.
            **kwargs: Other arguments to be passed to the data iterator.

        Returns: Train, valid, test,(datasets) indices(list of indices for data of each).

        '''

        if C is None:
            C = Europarl
        europarl = C(batch_size=10, **kwargs)
        if hasattr(europarl, 'logger'):
            logger = europarl.logger
            europarl.logger = None

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
                    s_i = int(s * europarl.n + accum)
                    split_idx.append(s_i)
                    accum += s_i
                idx = range(europarl.n)

                train_idx = idx[:split_idx[0]]
                valid_idx = idx[split_idx[0]:split_idx[1]]
                test_idx = idx[split_idx[1]:]
            idx = [train_idx, valid_idx, test_idx]
        else:
            logger.info('Splitting dataset into ratios  %.2f / %.2f /%.2f '
                        'using given indices'
                        % tuple(len(idx[i]) / float(europarl.n)
                                for i in range(3)))

        # Shouldn't have different number of batch sizes than datasets
        assert len(batch_sizes) == len(idx)

        datasets = []
        modes = ['train', 'valid', 'test']
        data = europarl.data
        europarl.data = dict()
        # Create correctly restricted copies of dataset
        for i, bs, mode in zip(idx, batch_sizes, modes):
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

    def next(self, batch_size=None):
        rval = super(Europarl, self).next(batch_size=batch_size)
        rval[self.name] = make_one_hot(rval[self.name],
                                       n_classes=self.nX_tokens)
        rval['label'] = make_one_hot(rval['label'],
                                     n_classes=self.nY_tokens)
        return rval

    def save_images(self, out_file=None):
        '''Shows tokenized in terms of original words.

        Uses reverse dictionary.

        '''
        raise NotImplementedError()
