from .context import models

import subprocess, pickle, os.path

import numpy as np

def test_recons():
    command = 'python princeton.py -c 784 100 -e 0.05 0.1 0.2 --epochs 10 --pickle'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    result = pickle.loads(process.stdout)

    assert len(result.keys()) == 6

    expected = np.array([22, 53, 94])
    actual = np.array([result[(784, 0.05)], result[(784, 0.1)], result[(784, 0.2)]])
    assert np.allclose(actual, expected, atol=5)

    expected = np.array([17, 43, 88])
    actual = np.array([result[(100, 0.05)], result[(100, 0.1)], result[(100, 0.2)]])
    assert np.allclose(actual, expected, atol=5)

def test_retrain():
    command = 'python princeton.py --retrain -c 784 100 -e 0.05 0.1 0.2 --epochs 10 --pickle'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    result = pickle.loads(process.stdout)

    assert len(result.keys()) == 6

    expected = np.array([21, 62, 96])
    actual = np.array([result[(784, 0.05)], result[(784, 0.1)], result[(784, 0.2)]])
    assert np.allclose(actual, expected, atol=5)

    expected = np.array([16, 49, 92])
    actual = np.array([result[(100, 0.05)], result[(100, 0.1)], result[(100, 0.2)]])
    assert np.allclose(actual, expected, atol=5)

def test_save():
    command = 'python princeton.py -c 784 -e 0.05 --epochs 0 --pickle --save --plot'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    assert os.path.exists('princeton.recons.pkl')
    assert os.path.exists('princeton.recons.png')
