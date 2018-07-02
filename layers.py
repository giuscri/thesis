import tensorflow as tf
import numpy as np
from keras.engine.topology import Layer

class PCA(Layer):
    def __init__(self, X_train=None, n_components=None, mean=None, tv=None, elementshape=None, **kwargs):
        if X_train is not None: # Layer must be _trained_ before!
            assert X_train.dtype == np.float32
            elementshape = X_train.shape[1:]
            pxs_per_element = np.prod(elementshape)
            if n_components is None: n_components = pxs_per_element

            flatX_train = np.reshape(X_train, newshape=(-1, pxs_per_element))
            mean = np.mean(flatX_train, axis=0)
            centeredflatX_train = flatX_train - mean
            covariance = np.matmul(np.transpose(centeredflatX_train), centeredflatX_train)
            _, _, vh = np.linalg.svd(covariance, full_matrices=False)
            tv = np.array(np.matrix(vh).H[:, :n_components])
            # kudos @neheller, https://github.com/neheller/TensorFlow-PCA/blob/master/pca.py

        self.mean = mean
        self.tv = tv
        self.elementshape = elementshape
        super(PCA, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PCA, self).build(input_shape)

    def call(self, X):
        pxs_per_element = np.prod(self.elementshape)
        flatX = tf.reshape(X, shape=(-1, pxs_per_element)) - self.mean
        T = tf.matmul(flatX, self.tv)
        flatR = tf.matmul(T, self.tv, transpose_b=True)
        R = tf.reshape(flatR + self.mean, shape=(-1, *self.elementshape))
        return R

    def get_config(self):
        config = {
            'elementshape': self.elementshape,
            'mean': self.mean,
            'tv': self.tv,
        }
        base_config = super(PCA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        elementshape = config['elementshape']
        mean = np.array(config['mean']['value'], dtype=np.float32)
        tv = np.array(config['tv']['value'], dtype=np.float32)
        batch_input_shape = config.get('batch_input_shape')
        return cls(mean=mean, tv=tv, elementshape=elementshape, batch_input_shape=batch_input_shape)
