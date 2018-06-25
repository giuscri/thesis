from .context import tools
from tools.attacks import fast_gradient_sign, fgs_adversarial_score
from tools.models import fc_100_100_10, train, accuracy
from tools.datasets import mnist

import numpy as np

MNIST = mnist()

def test_fgs_adversarial_score():
    X_train, y_train, X_test, y_test = MNIST

    network = fc_100_100_10()
    train(network, X_train, y_train, epochs=10)

    assert fgs_adversarial_score(network, X_test, y_test, eta=0.25) > 0.90

def test_fgs_examples_are_clipped():
    X_train, y_train, X_test, _ = MNIST

    network = fc_100_100_10()
    train(network, X_train, y_train, epochs=10)

    examples = fast_gradient_sign(network, X_test, eta=0.25)
    assert np.amin(examples) >= 0.
    assert np.amax(examples) <= 1.

def test_targeted_fgs_adversarial_score():
    X_train, y_train, X_test, y_test = MNIST

    network = fc_100_100_10()
    train(network, X_train, y_train, epochs=10) # you need _decent_ accuracy before the attack

    y_target = np.full(shape=(len(X_test),), fill_value=7)
    examples = fast_gradient_sign(network, X_test, y_target=y_target, eta=0.25)

    assert 0.60 < accuracy(network, examples, y_target) < 0.80
