import pytest
import tensorflow as tf
import numpy as np
import keras.backend as K
import os


@pytest.fixture(autouse=True)
def freeze_random_state():
    K.clear_session()
    tf.set_random_seed(1234)
    np.random.seed(1234)


@pytest.fixture
def environ():
    assert "PREFIX" not in os.environ or os.environ["PREFIX"] == "/tmp"
    os.environ["PREFIX"] = "/tmp"
    yield os.environ
    if "PREFIX" in os.environ:
        del os.environ["PREFIX"]


@pytest.fixture
def mnist():
    from .context import datasets

    return datasets.mnist()
