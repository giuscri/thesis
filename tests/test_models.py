from .context import models
from models import fc_100_100_10, pca_filtered_model, train, accuracy

import os, shutil
from math import isclose


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

    assert len(model.layers) == 8

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

    assert len(model.layers) == 8

def test_pca_filtered_keeping_10_components_accuracy(mnist):
    X_train, y_train, X_test, y_test = mnist
    model = pca_filtered_model(fc_100_100_10(), X_train, 10)

    train(model, X_train, y_train, epochs=2)
    assert isclose(accuracy(model, X_test, y_test), 0.44, abs_tol=0.01)

def test_pca_filtered_keeping_10_components_is_cached(mnist):
    X_train, y_train, X_test, y_test = mnist
    vanilla_model = fc_100_100_10()
    model = pca_filtered_model(vanilla_model, X_train, 10)
    assert model is pca_filtered_model(vanilla_model, X_train, 10)

def test_tensorboard_events_files_are_created(mnist, environ):
    model = fc_100_100_10()
    X_train, y_train, X_test, y_test = mnist

    train(model, X_train, y_train, epochs=2, tensorboard=True)
    assert isclose(accuracy(model, X_test, y_test), 0.54, abs_tol=0.01)

    dirname = '/tmp/tensorboardlogs/fc-100-100-10/'
    os.path.exists(dirname)
    shutil.rmtree(dirname)
