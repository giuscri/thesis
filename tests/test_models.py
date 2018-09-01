from .context import models
from models import (fc_100_100_10, pca_filtered_model, train, accuracy,
                    fastica_filtered_model, kernelpca_filtered_model,
                    incrementalpca_filtered_model, nmf_filtered_model,
                    truncatedsvd_filtered_model, save_to_file, load_from_file,
                    StopOnStableWeights)
from datasets import mnist

import os, shutil
from math import isclose
import pytest
import numpy as np

from sklearn.decomposition import (PCA, FastICA, IncrementalPCA,
                                   KernelPCA, TruncatedSVD, NMF)

def test_fc_100_100_10_structure(mnist):
    model = fc_100_100_10()
    X_train, y_train, X_test, y_test = mnist

    assert len(model.inputs) == 1
    assert model.inputs[0].shape.as_list() == [None, 28, 28]

    assert len(model.outputs) == 1
    assert model.outputs[0].shape.as_list() == [None, 10]

    assert len(model.layers) == 7


def test_fc_100_100_10_accuracy(mnist):
    model = fc_100_100_10()
    X_train, y_train, X_test, y_test = mnist

    train(model, X_train, y_train, epochs=2)
    assert isclose(accuracy(model, X_test, y_test), 0.544, abs_tol=0.01)


def test_pca_filtered_keeping_784_components_structure(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = pca_filtered_model(fc_100_100_10(), X_train)

    assert len(model.inputs) == 1
    assert model.input.shape.as_list() == [None, 28, 28]

    assert len(model.outputs) == 1
    assert model.output.shape.as_list() == [None, 10]

    assert len(model.layers) == 7


def test_pca_filtered_keeping_784_components_accuracy(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = pca_filtered_model(fc_100_100_10(), X_train)

    train(model, X_train, y_train, epochs=2)
    assert isclose(accuracy(model, X_test, y_test), 0.48, abs_tol=0.01)


def test_pca_filtered_keeping_10_components_structure(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = pca_filtered_model(fc_100_100_10(), X_train, 10)

    assert len(model.inputs) == 1
    assert model.input.shape.as_list() == [None, 28, 28]

    assert len(model.outputs) == 1
    assert model.output.shape.as_list() == [None, 10]

    assert len(model.layers) == 7

def test_pca_filtered_keeping_10_components_accuracy(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = pca_filtered_model(fc_100_100_10(), X_train, 10)

    train(model, X_train, y_train, epochs=2)
    assert isclose(accuracy(model, X_test, y_test), 0.44, abs_tol=0.01)

def test_pca_filtered_model_is_built_correctly(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = pca_filtered_model(fc_100_100_10(), X_train, 10)

    assert type(model.sklearn_transformer) is PCA
    assert model.name == "pca-filtered-model-10-components"

def test_fastica_filtered_model_is_built_correctly(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = fastica_filtered_model(fc_100_100_10(), X_train, 10)

    assert type(model.sklearn_transformer) is FastICA
    assert model.name == "fastica-filtered-model-10-components"

def test_nmf_filtered_model_is_built_correctly(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = nmf_filtered_model(fc_100_100_10(), X_train, 10)

    assert type(model.sklearn_transformer) is NMF
    assert model.name == "nmf-filtered-model-10-components"

def test_kernelpca_filtered_model_is_built_correctly(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = kernelpca_filtered_model(fc_100_100_10(), X_train[:1000], 10)
    # use a slice of X_train or you may run out of memory on Travis builds

    assert type(model.sklearn_transformer) is KernelPCA
    assert model.name == "kernelpca-filtered-model-10-components"

def test_truncatedsvd_filtered_model_is_built_correctly(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = truncatedsvd_filtered_model(fc_100_100_10(), X_train, 10)

    assert type(model.sklearn_transformer) is TruncatedSVD
    assert model.name == "truncatedsvd-filtered-model-10-components"

def test_incrementalpca_filtered_model_is_built_correctly(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = incrementalpca_filtered_model(fc_100_100_10(), X_train, 10)

    assert type(model.sklearn_transformer) is IncrementalPCA
    assert model.name == "incrementalpca-filtered-model-10-components"

def test_stop_on_stable_weights_callback():
    class Model:
        def __init__(self):
            self.stop_training = False
            self.__get_weights_call_counter = 0
            self.__weights = [
            np.array([[0.90316394, 0.66896059, 0.88231686], [0.96577754, 0.87451749, 0.87277546]]),
            np.array([0.08867801, 0.78951056, 0.76458674]),
            ]
            self.__noise = [w * 0.04 for w in self.__weights]

        def get_weights(self):
            if self.__get_weights_call_counter % 2 == 0:
                weights = [w + n for w, n in zip(self.__weights, self.__noise)]
            else:
                weights = [w - n for w, n in zip(self.__weights, self.__noise)]
            self.__get_weights_call_counter += 1
            return weights

    callback = StopOnStableWeights(patience=2, delta=0.05)
    callback.set_model(Model())
    callback.on_epoch_end(epoch=0)
    assert callback.model.stop_training is False
    callback.on_epoch_end(epoch=1)
    assert callback.model.stop_training is True

    callback = StopOnStableWeights(patience=2, delta=0.03)
    callback.set_model(Model())
    callback.on_epoch_end(epoch=0)
    assert callback.model.stop_training is False
    callback.on_epoch_end(epoch=1)
    assert callback.model.stop_training is False
