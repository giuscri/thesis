from .context import tools
from tools.attacks import fast_gradient_sign, fgs_adversarial_score
from tools.models import fc_100_100_10, train, accuracy

import numpy as np
from math import isclose

def test_fgs_adversarial_score(mnist):
    X_train, y_train, X_test, y_test = mnist

    network = fc_100_100_10()
    train(network, X_train, y_train, epochs=2)

    assert isclose(fgs_adversarial_score(network, X_test, y_test, eta=0.25), 0.99, abs_tol=0.01)

def test_fgs_examples_are_clipped(mnist):
    X_train, y_train, X_test, _ = mnist

    network = fc_100_100_10()

    examples = fast_gradient_sign(network, X_test, eta=0.25)
    assert np.amin(examples) >= 0.
    assert np.amax(examples) <= 1.

def test_targeted_fgs_adversarial_score(mnist):
    X_train, y_train, X_test, y_test = mnist

    network = fc_100_100_10()
    train(network, X_train, y_train, epochs=2)

    y_target = np.full(shape=(len(X_test),), fill_value=7)
    examples = fast_gradient_sign(network, X_test, y_target=y_target, eta=0.25)

    assert isclose(accuracy(network, examples, y_target), 0.98, abs_tol=0.01)
