from .context import attacks
from attacks import fastgradientsign, adversarial_score
from models import fc100_100_10, train
from datasets import mnist

import numpy as np

MNIST = mnist()

def test_fgs_score():
    X_train, y_train, X_test, y_test = MNIST

    network = fc100_100_10()
    train(network, X_train, y_train, store=False, epochs=10)
    attack = lambda network, X: fastgradientsign(network, X, eta=0.25)

    assert adversarial_score(network, X_test, y_test, attack) > 0.90

def test_fgs_clipping():
    X_train, y_train, X_test, _ = MNIST

    network = fc100_100_10()
    train(network, X_train, y_train, store=False, epochs=10)
    attack = lambda network, X: fastgradientsign(network, X, eta=0.25)

    examples = attack(network, X_test)
    assert np.amin(examples) >= 0.
    assert np.amax(examples) <= 1.
