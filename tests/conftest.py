import pytest
import tensorflow as tf
import numpy as np
import keras.backend as K
import tools

@pytest.fixture(autouse=True)
def freeze_random_state():
    K.clear_session()
    tf.set_random_seed(1234)
    np.random.seed(1234)
