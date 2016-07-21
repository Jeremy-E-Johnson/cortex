"""
Tests voc.py.
"""

from ..voc import VOC


def test_voc():
    test = VOC(source='$data', batch_size=10, chunk_size=15)
