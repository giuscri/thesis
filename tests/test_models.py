from .context import models
from models import (fc_100_100_10, pca_filtered_model, train, accuracy,
                    fastica_filtered_model, kernelpca_filtered_model,
                    incrementalpca_filtered_model, nmf_filtered_model,
                    truncatedsvd_filtered_model)

import os, shutil
from math import isclose

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
