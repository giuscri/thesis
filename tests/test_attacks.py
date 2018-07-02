from .context import attacks
from attacks import fast_gradient_sign, fgs_adversarial_score, __fast_gradient_sign_tf_symbols
from .context import models
from models import fc_100_100_10, train, accuracy

import numpy as np
from math import isclose
from pickle import dumps

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

def test_fgs_generated_symbols_are_cached(mnist):
    X_train, y_train, _, _ = mnist
    serializedX_train = dumps(X_train)
    serializedy_train = dumps(y_train)

    network = fc_100_100_10()

    symbols = __fast_gradient_sign_tf_symbols(network, serializedX_train, serializedy_train)
    assert symbols is __fast_gradient_sign_tf_symbols(network, serializedX_train, serializedy_train)
