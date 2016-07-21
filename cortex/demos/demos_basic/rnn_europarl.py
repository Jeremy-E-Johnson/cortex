"""
Demo for next word guessing using an RNN.

Try with cortex-run 'rnn_europarl.py <optional .yaml>'
"""

from collections import OrderedDict
import theano
import numpy as np
from cortex.models.rnn import SimpleRNN
from cortex.utils import intX, floatX
from cortex.datasets import resolve as resolve_dataset


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
    debug=False,
    dataset='europarl',
    distribution='multinomial',
    source='$data/basic/europarl'
)

_model_args = dict(
    dim_h=None,
    l2_decay=None,
    mask_in='mask_in'
)

simple_rnn_args = dict(
    dim_h=1000,
    input_layer='europarl',
    output='label',
    dropout=None
)

extra_arg_keys = ['simple_rnn_args']

#theano.config.compute_test_value = 'warn'

#theano.config.exception_verbosity = 'high'

#theano.config.optimizer = 'None'


def _build(module):
    models = OrderedDict()
    dataset = module.dataset
    simple_rnn_args = module.simple_rnn_args
    dim_in = dataset.dimsall[dataset.name][2]
    dim_out = dataset.dimsall[dataset.name][2]
    distribution = dataset.distributions[simple_rnn_args['output']]

    model = SimpleRNN.factory(dim_in=dim_in, dim_out=dim_out, distribution=distribution, **simple_rnn_args)

    models['rnn'] = model
    return models


def _cost(module):
    models = module.models

    mask_in = module.inputs['mask_in'].transpose(1, 0)
    X = module.inputs[module.dataset.name].transpose(1, 0, 2)
    used_inputs = [module.dataset.name, 'mask_in']

    model = models['rnn']
    outputs, preact, updates = model(X, m=mask_in)

    results = OrderedDict()
    p = outputs['p']
    base_cost = model.neg_log_prob(X[1:], p[:-1]).sum(0).mean()
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


def _viz(module, outputs, results, n_samples=10, n_steps=10):
    out_path = module.out_path
    out_path = None #### For testing purposes
    n_tokens = int(module.dataset.dimsall[module.dataset.name][2])

    pvals = np.zeros((n_samples, n_tokens)) + 1./float(n_tokens)
    x0 = module.models['rnn'].trng.multinomial(pvals=pvals, dtype=floatX)

    outputs, updates = module.models['rnn'].sample(x0=x0, n_steps=n_steps)

    updates = theano.OrderedUpdates(updates)

    f_vis = theano.function([], outputs['x'], updates=updates)

    def f_analysis():
        out = f_vis()
        return module.dataset.save_images(out, out_path=out_path)

    return f_analysis
