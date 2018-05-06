from .context import datasets
from datasets import mnist

def test_mnist():
    X_train, y_train, X_test, y_test = mnist()

    assert len(X_train) == 60000
    assert len(y_train) == 60000

    assert X_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000,)

    assert len(X_test) == 10000
    assert len(y_test) == 10000

    assert X_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000,)
