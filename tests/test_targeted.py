from .context import models

import subprocess

import json

import numpy as np

def test_targeted():
    command = 'python targeted.py -c 784 40 -e 0.05 0.07 --epochs 10'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    targeted = json.loads(process.stdout)

    assert len(targeted.keys()) == 2

    matrix_byeta = dict(targeted['784'])
    assert len(matrix_byeta.keys()) == 2

    matrix = np.zeros(shape=(10, 10))
    for ((i, j), score) in matrix_byeta[0.05]: matrix[i, j] = score
    assert np.allclose(np.diagonal(matrix), np.ones(10), atol=0.1)

    matrix = np.zeros(shape=(10, 10))
    for ((i, j), score) in matrix_byeta[0.07]: matrix[i, j] = score
    assert np.allclose(np.diagonal(matrix), np.ones(10), atol=0.1)

    matrix_byeta = dict(targeted['40'])
    assert len(matrix_byeta.keys()) == 2

    matrix = np.zeros(shape=(10, 10))
    for ((i, j), score) in matrix_byeta[0.05]: matrix[i, j] = score
    assert np.allclose(np.diagonal(matrix), np.ones(10), atol=0.1)

    matrix = np.zeros(shape=(10, 10))
    for ((i, j), score) in matrix_byeta[0.07]: matrix[i, j] = score
    assert np.allclose(np.diagonal(matrix), np.ones(10), atol=0.1)
