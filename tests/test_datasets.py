from .context import datasets
from datasets import mnist
import pytest


def test_parsed_mnist_has_expected_shape():
    X_train, y_train, X_test, y_test = mnist()

    assert len(X_train) == 60000
    assert len(y_train) == 60000

    assert X_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)

    assert len(X_test) == 10000
    assert len(y_test) == 10000

    assert X_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)


def test_parsed_mnist_is_not_writeable():
    X_train, y_train, X_test, y_test = mnist()
    with pytest.raises(ValueError):
        X_train[:100] = X_test[:100]
