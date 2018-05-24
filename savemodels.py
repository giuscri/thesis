#!/usr/bin/env python

from datasets import mnist
from models import train, evaluate, save, fc100_100_10, pcafiltered_fc

import os, sys
from argparse import ArgumentParser

argumentparser = ArgumentParser()

help = {
    '-ep': 'number of epochs to train models',
    '-ow': 'overwrite models',
    '-default': 'use default parameters',
    '-van': 'save trained vanilla network',
    '-retpca': 'save trained pca filtered network with retrain defense',
    '-recpca': 'save trained pca filtered network with reconstruction defense',
}

argumentparser.add_argument('-ow', action='store_true', help=help['-ow'])
argumentparser.add_argument('-default', action='store_true', help=help['-default'])
argumentparser.add_argument('-van', action='store_true', help=help['-van'])
argumentparser.add_argument('-ep', type=int, help=help['-ep'], metavar='epochs')
argumentparser.add_argument('-retpca', nargs='+', type=int, help=help['-retpca'], metavar='n_components')
argumentparser.add_argument('-recpca', nargs='+', type=int, help=help['-recpca'], metavar='n_components')
arguments = argumentparser.parse_args()

default = arguments.default

if default:
    vanilla = True
    retpca = [784, 331, 100, 80, 60, 40, 20]
    recpca = [784, 331, 100, 80, 60, 40, 20]
    overwrite = True
    epochs = 500
else:
    vanilla = arguments.van
    retpca = arguments.retpca or []
    recpca = arguments.recpca or []
    overwrite = arguments.ow
    epochs = arguments.ep if arguments.ep is not None else 500

if vanilla:
    filename = 'model/vanilla.h5'
    if os.path.exists(filename) and not overwrite:
        print(f'{filename} already exists. Use -ow if you really want to overwrite it.')
        sys.exit(-1)

for n_components in retpca:
    filename = f'model/retrain/pca/{n_components}.h5'
    if os.path.exists(filename) and not overwrite:
        print(f'{filename} already exists. Use -ow if you really want to overwrite it.')
        sys.exit(-1)

for n_components in recpca:
    filename = f'model/reconstruction/pca/{n_components}.h5'
    if os.path.exists(filename) and not overwrite:
        print(f'{filename} already exists. Use -ow if you really want to overwrite it.')
        sys.exit(-1)

if not vanilla and not retpca and not recpca:
    print('At least one network to save should be specified.')
    sys.exit(-1)

X_train, y_train, X_test, y_test = mnist()

if vanilla:
    filename = 'model/vanilla.h5'
    network = fc100_100_10()
    train(network, X_train, y_train, epochs=epochs, verbose=True)
    save(network, filename=filename)

if recpca:
    network = fc100_100_10()
    train(network, X_train, y_train, epochs=epochs, verbose=True)

for n_components in recpca:
    filtered_network = pcafiltered_fc(network, X_train, n_components)
    filename = f'model/pca/reconstruction/{n_components}.h5'
    save(filtered_network, filename=filename)

for n_components in retpca:
    network = fc100_100_10()
    filtered_network = pcafiltered_fc(network, X_train, n_components)
    train(filtered_network, X_train, y_train, epochs=epochs, verbose=True)
    filename = f'model/pca/retrain/{n_components}.h5'
    save(filtered_network, filename=filename)
