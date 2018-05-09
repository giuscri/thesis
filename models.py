from os.path import exists

import keras

import numpy as np

def save(network):
    if not exists('model/'): os.mkdir('model/')
    network.save('model/latest.h5')
    return network

def load():
    return keras.models.load_model('model/latest.h5')

def fc100_100_10():
    """Create or load FC100-100-10 network."""
    if exists('model/latest.h5'): return load()

    network = keras.models.Sequential([
        keras.layers.Flatten(batch_input_shape=(None, 28, 28)),
        keras.layers.Dense(100),
        keras.layers.Activation('sigmoid'),
        keras.layers.Dense(100),
        keras.layers.Activation('sigmoid'),
        keras.layers.Dense(10),
        keras.layers.Activation('softmax'),
    ])

    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True) # optimizer used by 1704.02654.pdf

    network.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return network

def filtered_fc(network, filterfn):
    """Prepend filter layer to `network`."""
    batch_shape = network.input.shape.as_list()
    dtype = network.input.dtype
    input = keras.layers.Input(batch_shape=batch_shape, dtype=dtype)

    output = keras.layers.Lambda(filterfn)(input)

    for layer in network.layers:
        if type(layer) == keras.layers.Dense: # workaround for (maybe) bugged keras.layers.set_weights
            W, b = layer.weights
            kernel_initializer = lambda _: W
            bias_initializer = lambda _: b
            cloned_layer = keras.layers.Dense(layer.units, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        else:
            cloned_layer = type(layer).from_config(layer.get_config())

        output = cloned_layer(output)

    filtered_network = keras.models.Model(inputs=input, outputs=output)
    filtered_network.compile(
        optimizer=network.optimizer,
        loss=network.loss,
        metrics=network.metrics
    )
    return filtered_network

def train(network, X_train, y_train, epochs=500, batch_size=500, store=True, verbose=True):
    """Train and save model"""
    num_classes = len(np.unique(y_train))
    onehot_y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)

    network.fit(X_train, onehot_y_train, epochs=epochs, batch_size=batch_size, verbose=1 if verbose else 0)
    if store: save(network)
    return network

def evaluate(network, X_test, y_test, verbose=True):
    """Evaluate model."""
    num_classes = network.output.shape.as_list()[-1]
    onehot_y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

    return network.evaluate(X_test, onehot_y_test, verbose=1 if verbose else 0)

def correctly_classified(network, X, y):
    """Filter correctly classified examples."""
    mask = np.argmax(network.predict(X), axis=1) == y
    return X[mask], y[mask]
