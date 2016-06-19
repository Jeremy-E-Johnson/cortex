"""
Tests europarl.py, try with nosetests test_europarl.py

Checks that the shapes split properly and that the masks line up.
"""

from cortex.datasets.basic.europarl import Europarl


def test_europarl(split=[0.7, 0.2, 0.1]):
    train, valid, test, idx = Europarl.factory(split=split, batch_sizes=[10, 10, 10],
                                               debug=True, source='/export/mialab/users/jjohnson/data/basic')

    for i, dataset in enumerate([train, valid, test]):

        for key in ['europarl', 'mask_in', 'label', 'mask_out']:  # Test shapes.
            assert dataset.data[key].shape == (int(idx[2][-1] * split[i]) + 1, 32)

        for k in idx[i]:  # Test masks.
            relative_k = k - idx[i][0]
            for j in range(0, len(dataset.data['europarl'][relative_k])):
                assert bool(dataset.data['europarl'][relative_k][j]) == bool(dataset.data['mask_in'][relative_k][j])
            for j in range(0, len(dataset.data['label'][relative_k])):
                assert bool(dataset.data['label'][relative_k][j]) == bool(dataset.data['mask_out'][relative_k][j])
