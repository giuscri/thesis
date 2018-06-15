from .context import tools
from tools.models import fc100_100_10, pcafiltered_fc, train, evaluate
from tools.datasets import mnist
from tools.filters import PCAFilterLayer

import os, shutil

MNIST = mnist()

def test_fc100_100_10():
    network = fc100_100_10()
    X_train, y_train, X_test, y_test = MNIST

    assert len(network.inputs) == 1
    assert network.inputs[0].shape.as_list() == [None, 28, 28]

    assert len(network.outputs) == 1
    assert network.outputs[0].shape.as_list() == [None, 10]

    assert len(network.layers) == 7

    train(network, X_train, y_train, epochs=10, store=False)
    _, accuracy = evaluate(network, X_test, y_test)
    assert 0.85 < accuracy < 0.90

def test_filtered_fc_pca784():
    X_train, y_train, X_test, y_test = MNIST
    network = pcafiltered_fc(fc100_100_10(), X_train)

    assert len(network.inputs) == 1
    assert network.input.shape.as_list() == [None, 28, 28]

    assert len(network.outputs) == 1
    assert network.output.shape.as_list() == [None, 10]

    assert len(network.layers) == 8

    train(network, X_train, y_train, epochs=10, store=False)
    _, accuracy = evaluate(network, X_test, y_test)
    assert 0.85 < accuracy < 0.90

def test_filtered_fc_pca10():
    X_train, y_train, X_test, y_test = MNIST
    network = pcafiltered_fc(fc100_100_10(), X_train, 10)

    assert len(network.inputs) == 1
    assert network.input.shape.as_list() == [None, 28, 28]

    assert len(network.outputs) == 1
    assert network.output.shape.as_list() == [None, 10]

    assert len(network.layers) == 8

    train(network, X_train, y_train, epochs=10, store=False)
    _, accuracy = evaluate(network, X_test, y_test)
    assert 0.75 < accuracy < 0.80

def test_cached_fc_pca10():
    X_train, y_train, X_test, y_test = MNIST
    vanilla_network = fc100_100_10()
    network = pcafiltered_fc(vanilla_network, X_train, 10)
    assert network is pcafiltered_fc(vanilla_network, X_train, 10)

def test_tensorboard():
    network = fc100_100_10()
    X_train, y_train, X_test, y_test = MNIST

    train(network, X_train, y_train, epochs=10, store=False, tensorboard=True, prefix='/tmp')
    _, accuracy = evaluate(network, X_test, y_test)
    assert 0.85 < accuracy < 0.90

    dirname = '/tmp/tensorboardlogs/fc100-100-10/'
    os.path.exists(dirname)
    shutil.rmtree(dirname)
