#!/bin/env python

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import json

import argparse

import itertools

from datasets import mnist
from filters import pcafilter
from models import fc100_100_10, filtered_fc, train, evaluate
from attacks import fastgradientsign

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--save', action='store_true')
parser.add_argument('--verbose', action='store_true')

parser.add_argument('-c', nargs='+', type=int, default=[784, 331, 100, 80, 60, 40, 20])
parser.add_argument('-e', nargs='+', type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25])

args = parser.parse_args()

X_train, y_train, X_test, y_test = mnist()
network = fc100_100_10()
train(network, X_train, y_train, epochs=args.epochs, store=False, verbose=args.verbose)

targeted = {}

for n_components in args.c:
    logging.info(f"filtering input retaining {n_components} principal components")

    filterfn = pcafilter(X_train, n_components=n_components)
    filtered_network = filtered_fc(network, filterfn)

    matrix_byeta = {}

    for eta in args.e:
        logging.info(f'building score matrix for eta={eta}')
        num_classes = network.output.shape.as_list()[-1]
        scorematrix = {}

        for source, destination in itertools.product(range(num_classes), repeat=2):
            logging.info(f'computing score for source={source} -> destination={destination}')
            mask = y_test == source
            X = X_test[mask]
            y_target = np.full(shape=(len(X),), fill_value=destination)
            examples = fastgradientsign(network, X, y_target, eta=eta)

            _, score = evaluate(network, examples, y_target, verbose=False)
            scorematrix[(source, destination)] = score

        matrix_byeta[eta] = tuple(scorematrix.items())

    targeted[n_components] = tuple(matrix_byeta.items())

print(json.dumps(targeted))

if args.save:
    filename = './targeted.json'
    logging.info(f'saving results in {filename}')
    with open(filename, 'w') as f: f.write(json.dumps(targeted))
