"""
Demo for next word guessing using an RNN.

Try with cortex-run 'rnn_europarl.py <optional .yaml>'
"""

from collections import OrderedDict
import theano
from cortex.models.rnn import SimpleRNN
from cortex.datasets import resolve as resolve_dataset


# Default arguments
_learning_args = dict(
    learning_rate=0.01,
    learning_rate_scheduler=None,
    optimizer='sgd',
    optimizer_args=dict(),
    epochs=100,
    valid_key='error',
    valid_sign='+',
    excludes=[]
)

_dataset_args = dict(
    train_batch_size=100,
    valid_batch_size=100,
    dataset='europarl',
    distribution='multinomial',
    source='$data/basic/'
)

_model_args = dict(
    dim_h=None,
    l2_decay=None,
)

simple_rnn_args = dict(
    dim_h=100,
    input_layer='europarl',
    output='label',
    dropout=None
)

extra_arg_keys = ['simple_rnn_args']


def _build(module):
    models = OrderedDict()
    dataset = module.dataset
    simple_rnn_args = module.simple_rnn_args
    dim_in = dataset.dims[dataset.name]
    dim_out = dataset.dims['label']
    distribution = dataset.distributions[simple_rnn_args['output']]

    model = SimpleRNN.factory(dim_in=dim_in, dim_out=dim_out, distribution=distribution, **simple_rnn_args)

    models['rnn'] = model
    return models


def _cost(module):
    models = module.models

    X = module.inputs[module.dataset.name]
    used_inputs = [module.dataset.name]

    model = models['rnn']
    outputs = model(X)

    results = OrderedDict()
    p = outputs[0]['p']
    base_cost = model.neg_log_prob(X[1:], p[:-1]).sum(axis=0)
    print base_cost, '####'
    print model.neg_log_prob(X[1:], p[:-1]), '#####'
    cost = base_cost

    updates = theano.OrderedUpdates()
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

    return used_inputs, results, updates, constants, outputs[0]
