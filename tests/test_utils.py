from .context import utils
from utils import loadpickle

import os, pickle

def test_loadpickle():
    expected = {'hello': 'world'}
    filename = '/tmp/test_loadpickle'
    with open(filename, 'wb') as f: pickle.dump(expected, f)

    actual = loadpickle(filename)
    os.remove(filename)
    assert actual == expected

def test_failed_loadpickle():
    noise = os.urandom(32)
    filename = '/tmp/test_failed_loadpickle'
    with open(filename, 'wb') as f: f.write(noise)

    actual = loadpickle(filename)
    os.remove(filename)
    assert actual is None
