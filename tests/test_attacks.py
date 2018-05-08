from .context import attacks
from attacks import fastgradientsign, adversarial_score
from models import fc100_100_10, train, evaluate
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

def test_fgs_targetclass():
    X_train, y_train, X_test, y_test = MNIST

    network = fc100_100_10()
    train(network, X_train, y_train, store=False, epochs=10) # you need _decent_ accuracy before the attack

    y_target = np.full(shape=(len(X_test),), fill_value=7)
    examples = fastgradientsign(network, X_test, y_target=y_target, eta=0.25)

    _, accuracy = evaluate(network, examples, y_target)
    assert 0.65 < accuracy > 0.70
