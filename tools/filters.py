import tensorflow as tf
import numpy as np
import keras.engine.topology

class PCAFilterLayer(keras.engine.topology.Layer):
    def __init__(self, X_train=None, n_components=None, **kwargs):
        if X_train is not None:
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

            self.elementshape = elementshape
            self.pxs_per_element = pxs_per_element
            self.mean = mean
            self.tv = tv
            self.batch_input_shape = None

        if 'elementshape' in kwargs:
            self.elementshape = kwargs['elementshape']
            del kwargs['elementshape']

        if 'pxs_per_element' in kwargs:
            self.pxs_per_element = kwargs['pxs_per_element']
            del kwargs['pxs_per_element']

        if 'mean' in kwargs:
            self.mean = kwargs['mean']
            del kwargs['mean']

        if 'tv' in kwargs:
            self.tv = kwargs['tv']
            del kwargs['tv']

        if 'batch_input_shape' in kwargs:
            self.batch_input_shape = kwargs['batch_input_shape']

        super(PCAFilterLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PCAFilterLayer, self).build(input_shape)

    def call(self, X):
        flatX = tf.reshape(X, shape=(-1, self.pxs_per_element)) - self.mean
        T = tf.matmul(flatX, self.tv)
        flatR = tf.matmul(T, self.tv, transpose_b=True)
        R = tf.reshape(flatR + self.mean, shape=(-1, *self.elementshape))
        return R

    def get_config(self):
        return {
            'elementshape': self.elementshape,
            'pxs_per_element': self.pxs_per_element,
            'mean': self.mean,
            'tv': self.tv,
            'batch_input_shape': self.batch_input_shape,
        }

    @classmethod
    def from_config(cls, config):
        elementshape = config['elementshape']
        pxs_per_element = config['pxs_per_element']
        mean = np.array(config['mean']['value'], dtype=np.float32)
        tv = np.array(config['tv']['value'], dtype=np.float32)
        batch_input_shape = config['batch_input_shape']
        return cls(elementshape=elementshape, pxs_per_element=pxs_per_element, mean=mean, tv=tv, batch_input_shape=batch_input_shape)
