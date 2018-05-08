from .context import models

import subprocess

import json

import numpy as np

def test_recons():
    command = './princeton.py -c 784 100 -e 0.05 0.1 0.2 --epochs 10'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    princeton = json.loads(process.stdout)

    assert len(princeton.keys()) == 2

    score_byeta = dict(princeton['784'])
    assert len(score_byeta.keys()) == 3
    actual = np.array(list(score_byeta.values()))
    expected = np.array([22, 53, 94])
    assert np.allclose(actual, expected, atol=5)

    score_byeta = dict(princeton['100'])
    assert len(score_byeta.keys()) == 3
    actual = np.array(list(score_byeta.values()))
    expected = np.array([17, 43, 88])
    assert np.allclose(actual, expected, atol=5)

def test_retrain():
    command = './princeton.py --retrain -c 784 100 -e 0.05 0.1 0.2 --epochs 10'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    princeton = json.loads(process.stdout)

    assert len(princeton.keys()) == 2

    score_byeta = dict(princeton['784'])
    assert len(score_byeta.keys()) == 3
    actual = np.array(list(score_byeta.values()))
    expected = np.array([21, 62, 96])
    assert np.allclose(actual, expected, atol=5)

    score_byeta = dict(princeton['100'])
    assert len(score_byeta.keys()) == 3
    actual = np.array(list(score_byeta.values()))
    expected = np.array([16, 49, 92])
    assert np.allclose(actual, expected, atol=5)
