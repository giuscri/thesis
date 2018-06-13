import os
from binascii import hexlify

import keras

import numpy as np
from .filters import PCAFilterLayer

def save(network, filename):
    directory = '/'.join(filename.split('/')[:-1])
    os.makedirs(directory, exist_ok=True)
    network.save(filename)
    return network

def load(filename):
    return keras.models.load_model(filename, custom_objects={
        'PCAFilterLayer': PCAFilterLayer,
    })

def fc100_100_10():
    """Create or load FC100-100-10 network."""
    if os.path.exists('model/latest.h5'): return load()

    network = keras.models.Sequential([
        keras.layers.Flatten(batch_input_shape=(None, 28, 28)),
        keras.layers.Dense(100),
        keras.layers.Activation('sigmoid'),
        keras.layers.Dense(100),
        keras.layers.Activation('sigmoid'),
        keras.layers.Dense(10),
        keras.layers.Activation('softmax'),
    ])

    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True) # optimizer used by 1704.02654.pdf

    network.name = 'fc100-100-10'
    network.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return network

PCAFILTEREDFCCACHE = {}

def pcafiltered_fc(network, X_train, n_components=None):
    """Prepend filter layer to `network`."""
    if (network, n_components) in PCAFILTEREDFCCACHE:
        return PCAFILTEREDFCCACHE[(network, n_components)]
    batch_input_shape = network.input.shape.as_list()
    cloned_layers = []
    cloned_layers.append(PCAFilterLayer(X_train, n_components, batch_input_shape=batch_input_shape))
    for layer in network.layers:
        cloned_layer = type(layer).from_config(layer.get_config())
        cloned_layers.append(cloned_layer)

    filtered_network = keras.models.Sequential(cloned_layers)
    filtered_network.compile(
        optimizer=network.optimizer,
        loss=network.loss,
        metrics=network.metrics
    )
    for layer, cloned_layer in zip(network.layers, filtered_network.layers[1:]):
        cloned_layer.set_weights(layer.get_weights())
    PCAFILTEREDFCCACHE[(network, n_components)] = filtered_network
    return filtered_network

def train(network, X_train, y_train, epochs=500, earlystopping=True, tensorboard=False, store=False, verbose=False):
    """Train and save model"""
    kerasverbose = 1 if verbose else 0
    num_classes = len(np.unique(y_train))
    onehot_y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)

    callbacks = []

    if earlystopping:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10))

    if tensorboard:
        randomsuffix = hexlify(os.urandom(32)[:10]).decode()
        log_dir = f'tensorboardlogs/{network.name}/{randomsuffix}'
        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0))

    network.fit(X_train, onehot_y_train, epochs=epochs, batch_size=500, verbose=kerasverbose, callbacks=callbacks, validation_split=0.2)
    if store: save(network)
    return network

def evaluate(network, X_test, y_test, verbose=False):
    """Evaluate model."""
    num_classes = network.output.shape.as_list()[-1]
    onehot_y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    return network.evaluate(X_test, onehot_y_test, verbose=1 if verbose else 0)

def correctly_classified(network, X, y):
    """Filter correctly classified examples."""
    mask = np.argmax(network.predict(X), axis=1) == y
    return X[mask], y[mask]
