from .context import tools
from tools.models import fc_100_100_10, pca_filtered_model, train, accuracy
from tools.datasets import mnist

import os, shutil
from math import isclose

MNIST = mnist()

def test_fc_100_100_10_structure():
    model = fc_100_100_10()
    X_train, y_train, X_test, y_test = MNIST

    assert len(model.inputs) == 1
    assert model.inputs[0].shape.as_list() == [None, 28, 28]

    assert len(model.outputs) == 1
    assert model.outputs[0].shape.as_list() == [None, 10]

    assert len(model.layers) == 7

def test_fc_100_100_10_accuracy():
    model = fc_100_100_10()
    X_train, y_train, X_test, y_test = MNIST

    train(model, X_train, y_train, epochs=2)
    assert isclose(accuracy(model, X_test, y_test), 0.544, abs_tol=0.01)

def test_pca_filtered_keeping_784_components_structure():
    X_train, y_train, X_test, y_test = MNIST
    model = pca_filtered_model(fc_100_100_10(), X_train)

    assert len(model.inputs) == 1
    assert model.input.shape.as_list() == [None, 28, 28]

    assert len(model.outputs) == 1
    assert model.output.shape.as_list() == [None, 10]

    assert len(model.layers) == 8

def test_pca_filtered_keeping_784_components_accuracy():
    X_train, y_train, X_test, y_test = MNIST
    model = pca_filtered_model(fc_100_100_10(), X_train)

    train(model, X_train, y_train, epochs=2)
    assert isclose(accuracy(model, X_test, y_test), 0.48, abs_tol=0.01)

def test_pca_filtered_keeping_10_components_structure():
    X_train, y_train, X_test, y_test = MNIST
    model = pca_filtered_model(fc_100_100_10(), X_train, 10)

    assert len(model.inputs) == 1
    assert model.input.shape.as_list() == [None, 28, 28]

    assert len(model.outputs) == 1
    assert model.output.shape.as_list() == [None, 10]

    assert len(model.layers) == 8

def test_pca_filtered_keeping_10_components_accuracy():
    X_train, y_train, X_test, y_test = MNIST
    model = pca_filtered_model(fc_100_100_10(), X_train, 10)

    train(model, X_train, y_train, epochs=2)
    assert isclose(accuracy(model, X_test, y_test), 0.44, abs_tol=0.01)

def test_pca_filtered_keeping_10_components_is_cached():
    X_train, y_train, X_test, y_test = MNIST
    vanilla_model = fc_100_100_10()
    model = pca_filtered_model(vanilla_model, X_train, 10)
    assert model is pca_filtered_model(vanilla_model, X_train, 10)

def test_tensorboard_events_files_are_created():
    model = fc_100_100_10()
    X_train, y_train, X_test, y_test = MNIST

    train(model, X_train, y_train, epochs=2, tensorboard=True, prefix='/tmp')
    assert isclose(accuracy(model, X_test, y_test), 0.54, abs_tol=0.01)

    dirname = '/tmp/tensorboardlogs/fc-100-100-10/'
    os.path.exists(dirname)
    shutil.rmtree(dirname)
