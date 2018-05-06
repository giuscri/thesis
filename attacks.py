import tensorflow as tf

import cleverhans.attacks
import cleverhans.utils_keras

import keras

import numpy as np

from models import correctly_classified

def fastgradientsign(network, X, eta=0.15):
    """Generate adversarial examples from X using FGS."""
    cleverhans_network = cleverhans.utils_keras.KerasModelWrapper(network)
    attack = cleverhans.attacks.FastGradientMethod(cleverhans_network)
    examples = attack.generate(network.input, eps=eta, ord=np.inf)

    session = keras.backend.get_session()

    var_list = map(lambda b: b.decode(), session.run(tf.report_uninitialized_variables()))
    uninitialized_variables = []
    for variable in tf.global_variables():
        if variable.name.split(':')[0] in var_list:
            uninitialized_variables.append(variable)
    # currently the only way of collecting uninitialized
    # variables in tf is to scan them all.

    session.run(tf.variables_initializer(uninitialized_variables))
    return session.run(examples, feed_dict={ network.input: X })

def adversarial_score(network, X_test, y_test, attack):
    """Compute adversarial score with `attack(network, X)`."""
    X, y = correctly_classified(network, X_test, y_test)
    adversarialX = attack(network, X)
    score = 1 - len(correctly_classified(network, adversarialX, y)[0]) / len(X)
    return score
