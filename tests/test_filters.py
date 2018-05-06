from .context import filters, datasets

from filters import pcafilter
from datasets import mnist

import tensorflow as tf
import numpy as np
import sklearn.decomposition

MNIST = mnist()

def test_pcafilter_784components():
    X_train, _, _, _ = MNIST
    flatX_train = X_train.reshape(-1, 784)

    sklearnfilter = sklearn.decomposition.PCA(svd_solver='full')
    sklearnfilter.fit(flatX_train)
    flatimage = flatX_train[:1]
    filteredimage = sklearnfilter.inverse_transform(sklearnfilter.transform(flatimage))
    expected = filteredimage.reshape(-1, 28, 28)

    actualfilter = pcafilter(X_train)
    tensor = actualfilter(flatimage)
    with tf.Session() as session: actual = session.run(tensor)

    assert np.allclose(actual, expected, atol=0.001)

def test_pcafilter_100components():
    X_train, _, _, _ = MNIST
    flatX_train = X_train.reshape(-1, 784)

    sklearnfilter = sklearn.decomposition.PCA(n_components=100, svd_solver='full')
    sklearnfilter.fit(flatX_train)
    flatimage = flatX_train[:1]
    filteredimage = sklearnfilter.inverse_transform(sklearnfilter.transform(flatimage))
    expected = filteredimage.reshape(-1, 28, 28)

    actualfilter = pcafilter(X_train, n_components=100)
    tensor = actualfilter(flatimage)
    with tf.Session() as session: actual = session.run(tensor)

    assert np.allclose(actual, expected, atol=0.001)
