from os import makedirs, urandom, environ
from os.path import exists, splitext, basename, dirname
from binascii import hexlify
from functools import partial

from keras.models import Sequential, load_model, clone_model
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.utils import to_categorical
import keras.backend as K

import numpy as np
from sklearn.decomposition import PCA, FastICA

from datasets import mnist
from utils import dump_pickle_to_file, load_pickle_from_file, random_string


class StopOnStableWeights(Callback):
    def __init__(self, delta=0.05, patience=10):
        self.collected_weights = []
        self.len_collected_weights = patience
        self.delta = delta

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        self.collected_weights.append(weights)

        if len(self.collected_weights) < self.len_collected_weights:
            return

        stacked_weights = np.stack(self.collected_weights)
        weights_per_epoch = len(self.collected_weights[0])
        weights_relative_std = []
        for i in range(weights_per_epoch):
            relative_std = np.std(stacked_weights[:, i]) / np.abs(np.mean(stacked_weights[:, i]))
            weights_relative_std.append(np.mean(relative_std))

        maximum_relative_std = max(weights_relative_std)
        if maximum_relative_std < self.delta: self.model.stop_training = True

        self.collected_weights = self.collected_weights[1:]
        assert len(self.collected_weights) == self.len_collected_weights - 1


def save_to_file(model, dirname):
    makedirs(dirname, exist_ok=True)
    model.save_weights(f"{dirname}/weights.h5")
    if "sklearn_transformer" in model.__dict__:
        sklearn_transformer = model.sklearn_transformer
        pickle_filename = f"{dirname}/{sklearn_transformer.__class__.__name__.lower()}.pkl"
        dump_pickle_to_file(sklearn_transformer, pickle_filename)
    return model


def load_from_file(dirname):
    model = fc_100_100_10()
    model.load_weights(f"{dirname}/weights.h5")
    X_train, _, _, _ = mnist()
    if exists(f"{dirname}/pca.pkl"):
        sklearn_transformer = load_pickle_from_file(f"{dirname}/pca.pkl")
        model = filtered_model(model, X_train, sklearn_transformer)
    elif exists(f"{dirname}/fast-ica.pkl"):
        sklearn_transformer = load_pickle_from_file(f"{dirname}/fast-ica.pkl")
        model = filtered_model(model, X_train, sklearn_transformer)

    return model


def fc_100_100_10():
    model = Sequential([
        Flatten(batch_input_shape=(None, 28, 28)),
        Dense(100),
        Activation("sigmoid"),
        Dense(100),
        Activation("sigmoid"),
        Dense(10),
        Activation("softmax"),
    ])

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)  # optimizer used by 1704.02654.pdf

    model.name = "fc-100-100-10"
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    model.preprocessing_fn = None

    return model


def filtered_model(model, X_train, sklearn_transformer=None):
    element_shape = X_train.shape[1:]
    pxs_per_element = np.prod(element_shape)

    def preprocessing_fn(X, sklearn_transformer):
        flatX = X.reshape(-1, pxs_per_element)
        filtered_flatX = sklearn_transformer.inverse_transform(sklearn_transformer.transform(flatX))
        return filtered_flatX.reshape(-1, *element_shape)

    filtered_model = clone_model(model)
    filtered_model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics)
    filtered_model.set_weights(model.get_weights())

    n_components = sklearn_transformer.n_components
    filtered_model.sklearn_transformer = sklearn_transformer
    filtered_model.preprocessing_fn = partial(preprocessing_fn, sklearn_transformer=sklearn_transformer)
    filtered_model.name = f"{sklearn_transformer.__class__.__name__.lower()}-filtered-model-{n_components}-components"
    return filtered_model


def pca_filtered_model(model, X_train, n_components=None, pca=None):
    element_shape = X_train.shape[1:]
    pxs_per_element = np.prod(element_shape)

    if pca is None:
        pca = PCA(n_components=n_components, svd_solver="full")
        flatX_train = X_train.reshape(-1, pxs_per_element)
        pca.fit(flatX_train)

    return filtered_model(model, X_train, sklearn_transformer=pca)


def fastica_filtered_model(model, X_train, n_components=None, fastica=None):
    element_shape = X_train.shape[1:]
    pxs_per_element = np.prod(element_shape)

    if fastica is None:
        fastica = FastICA(n_components=n_components)
        flatX_train = X_train.reshape(-1, pxs_per_element)
        fastica.fit(flatX_train)

    return filtered_model(model, X_train, sklearn_transformer=fastica)


def incrementalpca_filtered_model(model, X_train, n_components=None, incrementalpca=None):
    element_shape = X_train.shape[1:]
    pxs_per_element = np.prod(element_shape)

    if incrementalpca is None:
        incrementalpca = Incrementalpca(n_components=n_components)
        flatX_train = X_train.reshape(-1, pxs_per_element)
        incrementalpca.fit(flatX_train)

    return filtered_model(model, X_train, sklearn_transformer=incrementalpca)


def nmf_filtered_model(model, X_train, n_components=None, nmf=None):
    element_shape = X_train.shape[1:]
    pxs_per_element = np.prod(element_shape)

    if nmf is None:
        nmf = Nmf(n_components=n_components)
        flatX_train = X_train.reshape(-1, pxs_per_element)
        nmf.fit(flatX_train)

    return filtered_model(model, X_train, sklearn_transformer=nmf)


def truncatedsvd_filtered_model(model, X_train, n_components=None, truncatedsvd=None):
    element_shape = X_train.shape[1:]
    pxs_per_element = np.prod(element_shape)

    if truncatedsvd is None:
        truncatedsvd = Truncatedsvd(n_components=n_components)
        flatX_train = X_train.reshape(-1, pxs_per_element)
        truncatedsvd.fit(flatX_train)

    return filtered_model(model, X_train, sklearn_transformer=truncatedsvd)


def kernelpca_filtered_model(model, X_train, n_components=None, kernelpca=None):
    element_shape = X_train.shape[1:]
    pxs_per_element = np.prod(element_shape)

    if kernelpca is None:
        kernelpca = Kernelpca(n_components=n_components)
        flatX_train = X_train.reshape(-1, pxs_per_element)
        kernelpca.fit(flatX_train)

    return filtered_model(model, X_train, sklearn_transformer=kernelpca)


def _callbacks(reduce_lr_on_plateau=False, early_stopping=False,
               stop_on_stable_weights=False, reduce_lr_on_plateau_patience=30,
               early_stopping_patience=60, stop_on_stable_weights_patience=60):

        assert stop_on_stable_weights_patience // reduce_lr_on_plateau_patience > 1
        assert early_stopping_patience // reduce_lr_on_plateau_patience > 1
        # stop_on_stable_weights_patience and early_stopping_patience must be a multiple
        # of reduce_lr_on_plateau_patience

        r = []
        if reduce_lr_on_plateau:
            r.append(ReduceLROnPlateau(monitor="val_acc", patience=reduce_lr_on_plateau_patience))

        if early_stopping:
            r.append(EarlyStopping(monitor="val_acc", patience=early_stopping_patience))

        if stop_on_stable_weights:
            r.append(StopOnStableWeights(patience=stop_on_stable_weights_patience))

        return r


def train(model, X_train, y_train, epochs=500, verbose=True,
          early_stopping=False, reduce_lr_on_plateau=False,
          stop_on_stable_weights=False, early_stopping_patience=60,
          stop_on_stable_weights_patience=60, reduce_lr_on_plateau_patience=30):

    _verbose = 1 if verbose else 0
    num_classes = len(np.unique(y_train))
    one_hot_y_train = to_categorical(y_train, num_classes=num_classes)

    assert stop_on_stable_weights_patience // reduce_lr_on_plateau_patience > 1
    assert early_stopping_patience // reduce_lr_on_plateau_patience > 1
    # stop_on_stable_weights_patience and early_stopping_patience must be a multiple
    # of reduce_lr_on_plateau_patience

    callbacks = []
    if reduce_lr_on_plateau:
        callbacks.append(ReduceLROnPlateau(monitor="val_acc", patience=reduce_lr_on_plateau_patience))

    if early_stopping:
        callbacks.append(EarlyStopping(monitor="val_acc", patience=early_stopping_patience))

    if stop_on_stable_weights:
        callbacks.append(StopOnStableWeights(patience=stop_on_stable_weights_patience))

    if epochs == -1: # when `epochs` is -1 train _forever_
        epochs = 10**100

    return model.fit(X_train, one_hot_y_train, epochs=epochs, batch_size=500,
                     verbose=_verbose, callbacks=callbacks, validation_split=0.2)


def accuracy(model, X_test, y_test, verbose=True):
    _verbose = 1 if verbose else 0
    num_classes = model.output.shape.as_list()[-1]
    one_hot_y_test = to_categorical(y_test, num_classes=num_classes)

    if model.preprocessing_fn:
        X_test = model.preprocessing_fn(X_test)

    return model.evaluate(X_test, one_hot_y_test, verbose=_verbose)[1]


def predict(model, X):
    if model.preprocessing_fn:
        X = model.preprocessing_fn(X)

    return model.predict(X)


def filter_correctly_classified_examples(network, X, y):
    mask = np.argmax(predict(network, X), axis=1) == y
    return X[mask], y[mask]
