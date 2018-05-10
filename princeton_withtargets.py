#!/usr/bin/env python

import numpy as np

import logging, argparse, itertools, sys, pickle, os.path

from datasets import mnist
from filters import pcafilter
from models import fc100_100_10, filtered_fc, train, evaluate
from attacks import fastgradientsign

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--plot', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--pickle', action='store_true', help='print result encoded as pickle')

parser.add_argument('-c', nargs='+', type=int, default=[784, 331, 100, 80, 60, 40, 20], help='number of principal components to retain')
parser.add_argument('-e', nargs='+', type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25], help='value of eta to use for FGS')

parser.add_argument('--source', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
parser.add_argument('--destination', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

args = parser.parse_args()

if args.plot: print('TODO'); sys.exit(0)

filename = 'princeton_withtargets'

X_train, y_train, X_test, y_test = mnist()
network = fc100_100_10()
train(network, X_train, y_train, epochs=args.epochs, verbose=args.verbose)

networkdict = {}

for n_components in args.c:
    logging.info(f'builiding network with filter layer of {n_components}')
    filterfn = pcafilter(X_train, n_components=n_components)
    filtered_network = filtered_fc(network, filterfn)
    networkdict[n_components] = filtered_network

result = {}

for n_components, eta, source, destination in itertools.product(args.c, args.e, args.source, args.destination):
    logging.info(f'computing score for combination: ({n_components}, {eta}, {source}, {destination})')

    mask = y_test == source
    X = X_test[mask]
    y_target = np.full(shape=(len(X),), fill_value=destination)
    examples = fastgradientsign(network, X, y_target, eta=eta)

    _, score = evaluate(network, examples, y_target, verbose=False)
    result[(n_components, eta, source, destination)] = score

if args.pickle:
    sys.stdout.buffer.write(pickle.dumps(result))
else:
    print(result)

if args.save:
    logging.info(f'saving result in {filename}.pkl')
    with open(f'{filename}.pkl', 'wb') as f: pickle.dump(result, f)
