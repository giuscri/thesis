import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

from keras.utils import to_categorical
import keras.backend as K

import numpy as np
from functools import lru_cache
from pickle import loads, dumps

from models import filter_correctly_classified_examples


@lru_cache()
def __fast_gradient_sign_tf_symbols(model, serializedX, serializedy_target):
    X = loads(serializedX)
    y_target = loads(serializedy_target)

    cleverhans_model = KerasModelWrapper(model)
    attack = FastGradientMethod(cleverhans_model)

    X_sym = tf.placeholder(tf.float32, shape=model.input.shape)
    eta_sym = tf.placeholder(tf.float32)
    if y_target is not None:
        one_hot_y_target_sym = tf.placeholder(tf.float32, shape=model.output.shape)
    else:
        one_hot_y_target_sym = None

    kwargs = {"eps": eta_sym, "ord": np.inf, "clip_min": 0., "clip_max": 1.}
    if y_target is not None:
        kwargs["y_target"] = one_hot_y_target_sym

    example_sym = attack.generate(X_sym, **kwargs)
    return X_sym, one_hot_y_target_sym, example_sym, eta_sym


def adversarial_example(model, X, y_target=None, eta=0.15):
    assert y_target is None or len(y_target) == len(X)
    with_target = y_target is not None

    if with_target:
        num_classes = model.output.shape.as_list()[-1]
        one_hot_y_target = to_categorical(y_target, num_classes=num_classes)

    serializedX = dumps(X)
    serializedy_target = dumps(y_target)
    symbols = __fast_gradient_sign_tf_symbols(model, serializedX, serializedy_target)
    X_sym, one_hot_y_target_sym, example_sym, eta_sym = symbols
    session = K.get_session()
    feed_dict = {X_sym: X, eta_sym: eta}
    if with_target:
        feed_dict[one_hot_y_target_sym] = one_hot_y_target

    return session.run(example_sym, feed_dict=feed_dict)


def adversarial_score(model, X_test, y_test, eta=None, y_target=None):
    X, y = filter_correctly_classified_examples(model, X_test, y_test)
    adversarialX = adversarial_example(model, X, y_target, eta)
    fooling_examples, _ = filter_correctly_classified_examples(model, adversarialX, y)
    score = 1 - len(fooling_examples) / len(X)
    return score
