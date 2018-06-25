from os import makedirs, urandom
from os.path import exists
from binascii import hexlify

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import to_categorical

import numpy as np
from .layers import PCA

def save_to_file(model, filename):
    directory = '/'.join(filename.split('/')[:-1])
    makedirs(directory, exist_ok=True)
    model.save(filename)
    return model

def load_from_file(filename):
    return load_model(filename, custom_objects={
        'PCA': PCA,
    })

def fc_100_100_10():
    model = Sequential([
        Flatten(batch_input_shape=(None, 28, 28)),
        Dense(100),
        Activation('sigmoid'),
        Dense(100),
        Activation('sigmoid'),
        Dense(10),
        Activation('softmax'),
    ])

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True) # optimizer used by 1704.02654.pdf

    model.name = "fc-100-100-10"
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

PCA_FILTERED_MODEL_CACHE = {}

def pca_filtered_model(model, X_train, n_components=None):
    if (model, n_components) in PCA_FILTERED_MODEL_CACHE:
        return PCA_FILTERED_MODEL_CACHE[(model, n_components)]

    batch_input_shape = model.input.shape.as_list()
    layers = []
    layers.append(PCA(X_train, n_components, batch_input_shape=batch_input_shape))
    for wrapped_layer in model.layers:
        layer = type(wrapped_layer).from_config(wrapped_layer.get_config())
        layers.append(layer)

    filtered_model = Sequential(layers)
    filtered_model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics
    )

    for wrapped_layer, layer in zip(model.layers, filtered_model.layers[1:]):
        layer.set_weights(wrapped_layer.get_weights())

    PCA_FILTERED_MODEL_CACHE[(model, n_components)] = filtered_model
    filtered_model.name = "pca-filtered-model-{n_components}-components"
    return filtered_model

def train(model, X_train, y_train, epochs=500, early_stopping=True, tensorboard=False, verbose=False, prefix='.'):
    _verbose = 1 if verbose else 0
    num_classes = len(np.unique(y_train))
    one_hot_y_train = to_categorical(y_train, num_classes=num_classes)

    callbacks = []

    if early_stopping:
        callbacks.append(EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10))

    if tensorboard:
        random_string = hexlify(urandom(32)[:10]).decode()
        log_dir = f'{prefix}/tensorboardlogs/{model.name}/{random_string}'
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=0))

    model.fit(X_train, one_hot_y_train, epochs=epochs, batch_size=500, verbose=_verbose, callbacks=callbacks, validation_split=0.2)
    return model

def accuracy(model, X_test, y_test, verbose=False):
    _verbose = 1 if verbose else 0
    num_classes = model.output.shape.as_list()[-1]
    one_hot_y_test = to_categorical(y_test, num_classes=num_classes)

    return model.evaluate(X_test, one_hot_y_test, verbose=_verbose)[1]

def filter_correctly_classified_examples(network, X, y):
    mask = np.argmax(network.predict(X), axis=1) == y
    return X[mask], y[mask]
