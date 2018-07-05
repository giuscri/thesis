from os import makedirs, urandom, environ
from os.path import exists, splitext, basename, dirname
from binascii import hexlify
from functools import partial
from pickle import load, dump

from keras.models import Sequential, load_model, clone_model
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import to_categorical

import numpy as np
from sklearn.decomposition import PCA

from datasets import mnist


def save_to_file(model, dirname):
    makedirs(dirname, exist_ok=True)
    model.save_weights(f"{dirname}/weights.h5")
    if "_pca" in model.__dict__:
        with open(f"{dirname}/pca.pkl", "wb") as f:
            dump(model._pca, f)
    return model


def load_from_file(dirname):
    model = fc_100_100_10()
    model.load_weights(f"{dirname}/weights.h5")
    if exists(f"{dirname}/pca.pkl"):
        with open(f"{dirname}/pca.pkl", "rb") as f:
            pca = load(f)
        X_train, _, _, _ = mnist()
        model = pca_filtered_model(model, X_train, pca=pca)
    return model


def fc_100_100_10():
    model = Sequential(
        [
            Flatten(batch_input_shape=(None, 28, 28)),
            Dense(100),
            Activation("sigmoid"),
            Dense(100),
            Activation("sigmoid"),
            Dense(10),
            Activation("softmax"),
        ]
    )

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)  # optimizer used by 1704.02654.pdf

    model.name = "fc-100-100-10"
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
    model.preprocessing_fn = None

    return model


def pca_filtered_model(model, X_train=None, n_components=None, pca=None):
    element_shape = X_train.shape[1:]
    pxs_per_element = np.prod(element_shape)

    def preprocessing_fn(X, pca):
        flatX = X.reshape(-1, pxs_per_element)
        filtered_flatX = pca.inverse_transform(pca.transform(flatX))
        return filtered_flatX.reshape(-1, *element_shape)

    filtered_model = clone_model(model)
    filtered_model.compile(
        optimizer=model.optimizer, loss=model.loss, metrics=model.metrics
    )
    filtered_model.set_weights(model.get_weights())

    if pca is None:
        pca = PCA(n_components=n_components, svd_solver="full")
        flatX_train = X_train.reshape(-1, pxs_per_element)
        pca.fit(flatX_train)

    n_components = pca.n_components
    filtered_model._pca = pca
    filtered_model.preprocessing_fn = partial(preprocessing_fn, pca=pca)
    filtered_model.name = f"pca-filtered-model-{n_components}-components"
    return filtered_model


def train(
    model,
    X_train,
    y_train,
    epochs=500,
    early_stopping=True,
    tensorboard=True,
    verbose=True,
    preprocess=False,
):
    _verbose = 1 if verbose else 0
    num_classes = len(np.unique(y_train))
    one_hot_y_train = to_categorical(y_train, num_classes=num_classes)

    callbacks = []

    if early_stopping:
        callbacks.append(EarlyStopping(monitor="val_acc", min_delta=0.001, patience=10))

    if tensorboard:
        random_string = hexlify(urandom(32)[:10]).decode()
        prefix = environ.get("PREFIX", ".")
        log_dir = f"{prefix}/model/tensorboardlogs/{model.name}/{random_string}"
        callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=0))

    if preprocess:
        X_train = model.preprocessing_fn(X_train)

    model.fit(
        X_train,
        one_hot_y_train,
        epochs=epochs,
        batch_size=500,
        verbose=_verbose,
        callbacks=callbacks,
        validation_split=0.2,
    )
    return model


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
