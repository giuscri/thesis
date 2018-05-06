from .context import attacks
from attacks import fastgradientsign, adversarial_score
from models import fc100_100_10, train
from datasets import mnist

MNIST = mnist()

def test_fgs():
    X_train, y_train, X_test, y_test = MNIST

    network = fc100_100_10()
    train(network, X_train, y_train, store=False, epochs=10)
    attack = lambda network, X: fastgradientsign(network, X, eta=0.25)

    assert adversarial_score(network, X_test, y_test, attack) > 0.90
