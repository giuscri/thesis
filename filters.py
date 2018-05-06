import tensorflow as tf
import numpy as np

def pcafilter(X_train, n_components=None):
    assert X_train.dtype == np.float32
    elementshape = X_train.shape[1:]
    pxs_per_element = np.prod(elementshape)
    if not n_components: n_components = pxs_per_element

    flatX_train = np.reshape(X_train, newshape=(-1, pxs_per_element))
    mean = np.mean(flatX_train, axis=0)
    centeredflatX_train = flatX_train - mean
    covariance = np.matmul(np.transpose(centeredflatX_train), centeredflatX_train)
    _, _, vh = np.linalg.svd(covariance, full_matrices=False)
    value = np.matrix(vh).H[:, :n_components]
    tv = tf.constant(value=value, shape=[pxs_per_element, n_components], dtype=tf.float32)
    # kudos @neheller, https://github.com/neheller/TensorFlow-PCA/blob/master/pca.py

    def transform(X):
        flatX = tf.reshape(X, shape=(-1, pxs_per_element)) - mean
        T = tf.matmul(flatX, tv)
        flatR = tf.matmul(T, tv, transpose_b=True)
        R = tf.reshape(flatR + mean, shape=(-1, *elementshape))
        return R

    return transform
