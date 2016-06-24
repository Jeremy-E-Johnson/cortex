"""
Module for 2 dimensional pyramid RNN layers.
"""

from .rnn import RNN
import collections
import theano.tensor as T
import numpy as np
from ..utils import tools


class Pyramid_RNN(RNN):

    def __init__(self, dim_in, dim_hs, dim_out=None, output_net=None,
                 input_net=None, name='pyramid', **kwargs):

        if dim_out is None:
            self.dim_out = 1
        super(Pyramid_RNN, self).__init__(dim_in=dim_in, dim_hs=dim_hs, name=name,
                                          output_net=output_net, input_net=input_net, **kwargs)

    @staticmethod
    def factory(dim_in=None, dim_out=None, dim_hs=None, **kwargs):
        '''Factory for creating MLPs for Pyramid_RNN and returning .

        Convenience to quickly create MLPs from dictionaries, linking all
        relevant dimensions and distributions.

        Args:
            dim_in (int): input dimension.
            dim_hs (list): dimensions of recurrent units.
            dim_out (Optional[int]): output dimension. If not provided, assumed
                to be dim_in.

        Returns:
            RNN

        '''
        assert len(dim_hs) > 0
        if dim_out is None:
            dim_out = 1
        mlps, kwargs = RNN.mlp_factory(dim_in, dim_out, dim_hs, **kwargs)
        kwargs.update(**mlps)

        return Pyramid_RNN(dim_in, dim_hs, dim_out=dim_out, **kwargs)

    def set_params(self):
        '''Initialize RNN parameters.

        '''
        self.params = collections.OrderedDict()
        for i, dim_h in enumerate(self.dim_hs):
            Ur = tools.norm_weight(3 * dim_h, dim_h)
            self.params['Ur%d' % i] = Ur

        self.set_net_params()

    def _step(self, m, y, h_, Ur):
        '''Step function for RNN call.

        Args:
            m (T.tensor): masks.
            y (T.tensor): inputs.
            h_ (T.tensor): recurrent state.
            Ur (theano.shared): recurrent connection.

        Returns:
            T.tensor: next recurrent state.

        '''
        H_t    = T.concatenate((h_, T.roll(h_, 1, 2), T.roll(h_, -1, 2)), 2)
        preact = y + T.dot(H_t, Ur)
        h      = T.tanh(preact)
        h      = m * h + (1 - m) * h_
        return h
