from .context import models

import subprocess, pickle

import numpy as np

def test_recons_withtargets():
    command = 'python princeton_withtargets.py -c 784 40 -e 0.05 0.07 --epochs 10 --pickle'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    result = pickle.loads(process.stdout)

    assert len(result.keys()) == 400

    for n_components, eta, source, destination in filter(lambda k: k[-2] == k[-1], result.keys()):
        assert source == destination
        assert np.allclose(result[(n_components, eta, source, destination)], [1.], atol=0.1)
