'''
Module for testing 2D pyramid RNN.
'''

from cortex.models.pyramid_rnn import Pyramid_RNN
import numpy as np
import theano
import theano.tensor as T

theano.config.optimizer = 'None'

def test_build(dim_in=13, dim_h=17):
    pyramid = Pyramid_RNN.factory(dim_in=dim_in, dim_hs=[dim_h],
                                  dim_out=1)
    pyramid.set_tparams()

    return pyramid


def test_step(pyramid=None, dim_in=13, dim_h=17):
    if pyramid is None:
        pyramid = test_build(dim_in=dim_in, dim_h=dim_h)

    m = theano.tensor.tensor3()
    y = theano.tensor.tensor3()
    h_ = theano.tensor.tensor3()
    Ur = theano.tensor.matrix()

    activation = pyramid._step(m, y, h_, Ur)
    f = theano.function([m, y, h_, Ur], activation)


    t = f(np.ones((10, dim_in, dim_h), dtype='float32'), np.ones((10, dim_in, dim_h), dtype='float32'),
          np.ones((10, dim_in, dim_h), dtype='float32'), pyramid.params['Ur0'])

    preact = np.ones((10, dim_in, dim_h), dtype='float32') + \
             np.dot(np.ones((10, dim_in, 3*dim_h), dtype='float32'), pyramid.params['Ur0'])
    n = np.tanh(preact)

    np.testing.assert_almost_equal(t, n)


def test_call(pyramid=None, dim_in=13, dim_h=17):
    if pyramid is None:
        pyramid = test_build(dim_in=dim_in, dim_h=dim_h)

    rng = np.random.RandomState()

    x = rng.randn(13, 10, 13)

    y = pyramid(x)

    f = theano.function([], y[0]['p'])

    print f()
