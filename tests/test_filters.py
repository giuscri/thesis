from .context import tools
from tools.layers import PCA

import tensorflow as tf
import numpy as np
import sklearn.decomposition

def test_pca_implementation_is_equivalent_to_sklearn_when_keeping_784_components(mnist):
    X_train, _, _, _ = mnist
    flatX_train = X_train.reshape(-1, 784)

    sklearnfilter = sklearn.decomposition.PCA(svd_solver='full')
    sklearnfilter.fit(flatX_train)
    flatimage = flatX_train[:1]
    filteredimage = sklearnfilter.inverse_transform(sklearnfilter.transform(flatimage))
    expected = filteredimage.reshape(-1, 28, 28)

    actualfilter = PCA(X_train)
    flatimage_sym = tf.placeholder(tf.float32)
    with tf.Session() as session:
        actual = session.run(actualfilter(flatimage_sym), feed_dict={flatimage_sym: flatimage})

    assert np.allclose(actual, expected, atol=0.001)

def test_pca_implementation_is_equivalent_to_sklearn_when_keeping_100_components(mnist):
    X_train, _, _, _ = mnist
    flatX_train = X_train.reshape(-1, 784)

    sklearnfilter = sklearn.decomposition.PCA(n_components=100, svd_solver='full')
    sklearnfilter.fit(flatX_train)
    flatimage = flatX_train[:1]
    filteredimage = sklearnfilter.inverse_transform(sklearnfilter.transform(flatimage))
    expected = filteredimage.reshape(-1, 28, 28)

    actualfilter = PCA(X_train, 100)
    flatimage_sym = tf.placeholder(tf.float32)
    with tf.Session() as session:
        actual = session.run(actualfilter(flatimage_sym), feed_dict={flatimage_sym: flatimage})

    assert np.allclose(actual, expected, atol=0.001)
