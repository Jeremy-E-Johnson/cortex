"""
Demo for Pyramid RNN on VOC classification dataset.

Try with 'cortex-run pyramid_voc.py'
"""

"""
Demo for next word guessing using an RNN.

Try with cortex-run 'rnn_europarl.py <optional .yaml>'
"""

from collections import OrderedDict
import theano
import numpy as np
from cortex.models.pyramid_rnn import Pyramid_RNN
from cortex.utils import intX, floatX
from cortex.datasets import resolve as resolve_dataset
import theano.tensor as T


# Default arguments
_learning_args = dict(
    learning_rate=0.01,
    learning_rate_scheduler=None,
    optimizer='rmsprop',
    optimizer_args=dict(),
    epochs=100,
    valid_key='-sum log p(x | y)',
    valid_sign='+',
    excludes=[]
)

_dataset_args = dict(
    train_batch_size=10,
    valid_batch_size=10,
    #test_batch_size=10,
    debug=False,
    dataset='voc',
    chunks=1000,
    distribution='multinomial',
    chunk_size=15,
    source='$data'
)

_model_args = dict(
    dim_h=None,
    l2_decay=None,
    mask_in='mask_in'
)

pyramid_args = dict(
    dim_hs=[17],
    input_layer='voc',
    output='label',
)

extra_arg_keys = ['pyramid_args']

theano.config.on_unused_input = 'ignore'
theano.config.optimizer = 'None'
#theano.config.exception_verbosity = 'high'
#theano.config.compute_test_value = 'warn'


def _build(module):
    models = OrderedDict()
    dataset = module.dataset
    pyramid_args = module.pyramid_args
    width = dataset.chunk_size
    dim_in = 1
    dim_out = 1
    distribution = dataset.distributions[pyramid_args['output']]

    model = Pyramid_RNN.factory(dim_in=dim_in, dim_out=dim_out, distribution=distribution,
                                width=width, **pyramid_args)

    models['pyramid_rnn'] = model
    return models


def _cost(module):
    models = module.models

    X = module.inputs[module.dataset.name]#.swapaxes(0, 1)
    Y = module.inputs['label']
    used_inputs = [module.dataset.name, 'label']

    model = models['pyramid_rnn']
    main(model)

    outputs, preact, updates = model(X)

    results = OrderedDict()
    p = outputs['p']
    base_cost = model.neg_log_prob(Y, p).sum(0).mean()
    cost = base_cost

    constants = []

    l2_decay = module.l2_decay
    if l2_decay is not False and l2_decay > 0.:
        module.logger.info('Adding %.5f L2 weight decay' % l2_decay)
        l2_rval = model.l2_decay(l2_decay)
        l2_cost = l2_rval.pop('cost')
        cost += l2_cost
        results['l2_cost'] = l2_cost

    # results['error'] = (Y * (1 - p)).sum(axis=1).mean()
    results['-sum log p(x | y)'] = base_cost
    results['cost'] = cost

    return used_inputs, results, updates, constants, outputs


def main(model):
    x = T.alloc(1, 8, 10, 17)

    params = model.get_sample_params()

    #print params

    a = model.call_seqs(x, None, 0, *params)[0]

    print a.eval().shape, '****************************************'
