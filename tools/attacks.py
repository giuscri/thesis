import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

from keras.utils import to_categorical
import keras.backend as K

import numpy as np

from .models import filter_correctly_classified_examples

FAST_GRADIENT_SIGN_CACHE = {} # cache of tf symbolic variables indexed by tuple (with_target, model, input_shape)

def fast_gradient_sign(model, X, y_target=None, eta=0.15):
    assert y_target is None or len(y_target) == len(X)
    with_target = y_target is not None

    if with_target:
        num_classes = model.output.shape.as_list()[-1]
        one_hot_y_target = to_categorical(y_target, num_classes=num_classes)

    input_shape = X.shape[1:]
    cached = FAST_GRADIENT_SIGN_CACHE.get((with_target, model, input_shape))
    if cached:
        X_sym, one_hot_y_target_sym, example_sym, eta_sym = cached
    else:
        cleverhans_model = KerasModelWrapper(model)
        attack = FastGradientMethod(cleverhans_model)

        X_sym = tf.placeholder(tf.float32, shape=model.input.shape)
        eta_sym = tf.placeholder(tf.float32)
        one_hot_y_target_sym = None
        if with_target: one_hot_y_target_sym = tf.placeholder(tf.float32, shape=model.output.shape)

        kwargs = {'eps': eta_sym, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
        if with_target: kwargs['y_target'] = one_hot_y_target_sym
        example_sym = attack.generate(X_sym, **kwargs)

        FAST_GRADIENT_SIGN_CACHE[(with_target, model, input_shape)] = (X_sym, one_hot_y_target_sym, example_sym, eta_sym)

    session = K.get_session()
    feed_dict = {X_sym: X, eta_sym: eta}
    if with_target: feed_dict[one_hot_y_target_sym] = one_hot_y_target

    return session.run(example_sym, feed_dict=feed_dict)

def fgs_adversarial_score(model, X_test, y_test, eta=None, y_target=None):
    X, y = filter_correctly_classified_examples(model, X_test, y_test)
    adversarialX = fast_gradient_sign(model, X, y_target, eta)
    score = 1 - len(filter_correctly_classified_examples(model, adversarialX, y)[0]) / len(X)
    return score
