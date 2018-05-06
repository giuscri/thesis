from .context import models
from models import fc100_100_10, filtered_fc, train, evaluate
from datasets import mnist
from filters import pcafilter

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
    filterfn = pcafilter(X_train)
    network = filtered_fc(fc100_100_10(), filterfn)

    assert len(network.inputs) == 1
    assert network.input.shape.as_list() == [None, 28, 28]

    assert len(network.outputs) == 1
    assert network.output.shape.as_list() == [None, 10]

    assert len(network.layers) == 9

    train(network, X_train, y_train, epochs=10, store=False)
    _, accuracy = evaluate(network, X_test, y_test)
    assert 0.85 < accuracy < 0.90

def test_filtered_fc_pca10():
    X_train, y_train, X_test, y_test = MNIST
    filterfn = pcafilter(X_train, n_components=10)
    network = filtered_fc(fc100_100_10(), filterfn)

    assert len(network.inputs) == 1
    assert network.input.shape.as_list() == [None, 28, 28]

    assert len(network.outputs) == 1
    assert network.output.shape.as_list() == [None, 10]

    assert len(network.layers) == 9

    train(network, X_train, y_train, epochs=10, store=False)
    _, accuracy = evaluate(network, X_test, y_test)
    assert 0.75 < accuracy < 0.80

