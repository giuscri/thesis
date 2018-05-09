from .context import models

import subprocess, pickle

import numpy as np

def test_recons_withtargets():
    command = 'python princeton_withtargets.py -c 784 40 -e 0.05 --source 0 4 --destination 4 --epochs 10 --pickle'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    result = pickle.loads(process.stdout)

    expected = {(784, 0.05, 0, 4): 0.002040, (784, 0.05, 4, 4): 0.971486, (40, 0.05, 0, 4): 0.002040, (40, 0.05, 4, 4): 0.971486}

    assert result.keys() == expected.keys()
    for k in result: assert np.allclose(result[k], expected[k], atol=0.02)
