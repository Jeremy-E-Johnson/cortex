'''
Module for testing 2D pyramid RNN.
'''

from cortex.models.pyramid_rnn import Pyramid_RNN
import numpy as np
import theano
import theano.tensor as T

theano.config.optimizer = 'None'


def test_build(dim_in=1, dim_h=17, width=13):
    pyramid = Pyramid_RNN.factory(dim_in=dim_in, dim_hs=[dim_h],
                                  width=width, dim_out=1)
    pyramid.set_tparams()

    return pyramid


def test_step(pyramid=None, dim_in=1, dim_h=17, width=13):
    if pyramid is None:
        pyramid = test_build(dim_in=dim_in, dim_h=dim_h, width=width)

    m = theano.tensor.tensor3()
    y = theano.tensor.tensor3()
    h_ = theano.tensor.tensor3()
    Ur = theano.tensor.matrix()

    activation = pyramid._step(m, y, h_, Ur)
    f = theano.function([m, y, h_, Ur], activation)

    t = f(np.ones((10, width, dim_h), dtype='float32'), np.ones((10, width, dim_h), dtype='float32'),
          np.ones((10, width, dim_h), dtype='float32'), pyramid.params['Ur0'])

    preact = np.ones((10, width, dim_h), dtype='float32') + \
             np.dot(np.ones((10, width, 3*dim_h), dtype='float32'), pyramid.params['Ur0'])

    n = np.tanh(preact)

    np.testing.assert_almost_equal(t, n)


def test_call(pyramid=None, dim_in=1, dim_h=17, width=13):
    if pyramid is None:
        pyramid = test_build(dim_in=dim_in, dim_h=dim_h, width=width)

    rng = np.random.RandomState()

    x = rng.randn(13, 10, 13)

    y = pyramid(x)

    f = theano.function([], y[0]['z'])

    # Now calculate what f should be using Numpy.

    outs =[]
    for k in range(0, 4):  # Iterate through directions
        x = np.rot90(x.swapaxes(1, 2)).swapaxes(1, 2)  # Rotate input

        dir_input = x[:(width + 1)/2, :, :, None]
        dir_input = pyramid.input_net.params['b0'] + np.dot(dir_input, pyramid.input_net.params['W0'])

        h = np.zeros((x.shape[1], width, dim_h))
        Ur = pyramid.params['Ur0']

        for layer in dir_input:  # Iterate through height of pyramid
            h_t = np.concatenate((h, np.roll(h, 1, 2), np.roll(h, -1, 2)), 2)
            preact = layer + np.dot(h_t, Ur)
            h = np.tanh(preact)

        outs.append(h[:, (width + 1)/2, :])  # Remember output for direction

    output = pyramid.output_net.params['b0'] + np.dot(sum(outs), pyramid.output_net.params['W0'])  # Sum over direction\
    # and apply output network.

    # Test for equality.

    np.testing.assert_almost_equal(output, f())  # Check if they match.
