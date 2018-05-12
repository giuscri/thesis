import tensorflow as tf

import cleverhans.attacks
import cleverhans.utils_keras

import keras

import numpy as np

from models import correctly_classified

FGSCACHE = {} # cache of tf symbolic variables indexed by (withtarget, network, X.shape[1:], eta)

def fastgradientsign(network, X, y_target=None, eta=0.15):
    """Generate adversarial examples from X to y using FGS."""
    assert y_target is None or len(y_target) == len(X)
    withtarget = y_target is not None

    if withtarget:
        num_classes = network.output.shape.as_list()[-1]
        onehot_y_target = keras.utils.to_categorical(y_target, num_classes=num_classes)

    elementshape = X.shape[1:]
    cached = FGSCACHE.get((withtarget, network, elementshape, eta))
    if cached:
        Xsym, onehot_y_targetsym, examplesym = cached
    else:
        cleverhans_network = cleverhans.utils_keras.KerasModelWrapper(network)
        attack = cleverhans.attacks.FastGradientMethod(cleverhans_network)

        Xsym = tf.placeholder(tf.float32, shape=network.input.shape)
        onehot_y_targetsym = None
        if withtarget: onehot_y_targetsym = tf.placeholder(tf.float32, shape=network.output.shape)

        kwargs = { 'eps': eta, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
        if withtarget: kwargs['y_target'] = onehot_y_targetsym
        examplesym = attack.generate(Xsym, **kwargs)

        FGSCACHE[(withtarget, network, elementshape, eta)] = (Xsym, onehot_y_targetsym, examplesym)

    session = keras.backend.get_session()
    feed_dict = { Xsym: X }
    if withtarget: feed_dict[onehot_y_targetsym] = onehot_y_target

    return session.run(examplesym, feed_dict=feed_dict)

def adversarial_score(network, X_test, y_test, attack):
    """Compute adversarial score with `attack(network, X)`."""
    X, y = correctly_classified(network, X_test, y_test)
    adversarialX = attack(network, X)
    score = 1 - len(correctly_classified(network, adversarialX, y)[0]) / len(X)
    return score
