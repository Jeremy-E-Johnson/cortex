"""
Module for 2 dimensional pyramid RNN layers.
"""

from .rnn import RNN
import collections
import theano.tensor as T
import theano
import numpy as np
from ..utils import tools
import collections as coll
from ..utils import floatX


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

    def step_call(self, x, m, h0s, *params):
        '''Step version of __call__ for scan

        Args:
            x (T.tensor): input.
            m (T.tensor): mask.
            h0s (list): list of recurrent initial states. Calculated in this function now, ie NOT IMPLEMENTED
            *params: list of theano.shared.

        Returns:
            OrderedDict: dictionary of results. and now calculated h0s

        '''
        n_steps = (x.shape[0] + 1)/2
        n_samples = x.shape[1]
        input = x

        updates = theano.OrderedUpdates()

        h0s = []
        hs = []
        directional_values = []
        for k in range(0, 4):  # Iterate through directions.
            x = np.swapaxes(np.rot90(np.swapaxes(input, 1, 2), k), 1, 2)[0:(self.dim_in + 1)/2].astype('float32')
            h0s.append([T.alloc(0., x.shape[1], self.dim_in, dim_h).astype(floatX) for dim_h in self.dim_hs])
            for i, h0 in enumerate(h0s[k]):
                seqs         = [m[:, :, None]] + self.call_seqs(x, None, i, *params)
                outputs_info = [h0]
                non_seqs     = [self.get_recurrent_args(*params)[i]]
                h, updates_ = theano.scan(
                    self._step,
                    sequences=seqs,
                    outputs_info=outputs_info,
                    non_sequences=non_seqs,
                    name=self.name + '_recurrent_steps_%d' % i,
                    n_steps=n_steps)
                hs.append(h)
                x = h
                updates += updates_
            directional_values.append(h[(self.dim_in + 1)/2])  # Remember directional outputs.

        o_params    = self.get_output_args(*params)
        out_net_out = self.output_net.step_call(sum(directional_values), *o_params)  # Sum different directions.
        preact      = out_net_out['z']
        p           = out_net_out['p']

        return coll.OrderedDict(hs=hs, p=p, z=preact), updates, h0s

    def __call__(self, x, m=None, h0s=None, condition_on=None):
        '''Call function.

        For learning RNNs.

        Args:
            x (T.tensor): input sequence. window x batch x dim (a x b x a) where a is chunk size, b is batch size.
            m (T.tensor): mask. window x batch. For masking in recurrent steps. NOT IMPLEMENTED
            h0s (Optional[list]): initial h0s. NOT IMPLEMENTED
            condition_on (Optional[T.tensor]): conditional for recurrent step.

        Returns:
            OrderedDict: dictionary of results: hiddens, probabilities, and
                preacts.
            theano.OrderedUpdates.

        '''
        constants = []
        input_rotations = []

        ''' Calculating h0s in step_call so that rotations of data happen once and don't need to be saved.
        if h0s is None and self.init_net is not None:
            h0s = self.init_net.initialize(x[0])
            constants += h0s
        elif h0s is None:
            h0s = [T.alloc(0., x.shape[1], dim_h).astype(floatX) for dim_h in self.dim_hs]
        '''

        if m is None:
            m = T.ones((x.shape[0], x.shape[1])).astype(floatX)

        params = self.get_sample_params()

        results, updates, h0s = self.step_call(x, m, h0s, *params)
        results['h0s'] = h0s
        return results, updates, constants

    def call_seqs(self, x, condition_on, level, *params):
        '''Prepares the input for `__call__`.

        Args:
            x (T.tensor): input
            condtion_on (T.tensor or None): tensor to condition recurrence on.
            level (int): reccurent level.
            *params: list of theano.shared.

        Returns:
            list: list of scan inputs.

        '''
        print x.shape
        if level == 0:
            i_params = self.get_input_args(*params)
            a = self.input_net.step_preact(x, *i_params)
        else:
            i_params = self.get_inter_args(level - 1, *params)
            a = self.inter_nets[level - 1].step_preact(x, *i_params)
        print a.shape
        print self.input_net.dim_in
        print self.input_net.dim_out

        if condition_on is not None:
            a += condition_on

        return [a]
