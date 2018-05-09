#!/bin/env python

import numpy as np

import itertools, argparse, pickle, sys, logging

from datasets import mnist
from filters import pcafilter
from models import fc100_100_10, filtered_fc, train
from attacks import fastgradientsign, adversarial_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--plot', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--pickle', action='store_true')

parser.add_argument('-c', nargs='+', type=int, default=[784, 331, 100, 80, 60, 40, 20])
parser.add_argument('-e', nargs='+', type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25])

args = parser.parse_args()

filename = 'princeton.recons' if args.retrain else 'princeton.retrain'

X_train, y_train, X_test, y_test = mnist()
network = fc100_100_10()
train(network, X_train, y_train, epochs=args.epochs, verbose=args.verbose)

networkdict = {}

for n_components in args.c:
    logging.info(f'building network with filter layer of {n_components}')
    filterfn = pcafilter(X_train, n_components=n_components)
    filtered_network = filtered_fc(network, filterfn)

    if args.retrain:
        logging.info('retraining network with filter layer of {n_components}')
        train(filtered_network, X_train, y_train, epochs=args.epochs, verbose=args.verbose)

    networkdict[n_components] = filtered_network

result = {}

for n_components, eta in itertools.product(args.c, args.e):
    logging.info(f'computing score for combination: ({n_components}, {eta})')
    attack = lambda n, X: fastgradientsign(n, X, eta=eta)
    score = int(100 * adversarial_score(networkdict[n_components], X_test, y_test, attack))
    result[(n_components, eta)] = score

if args.pickle:
    sys.stdout.buffer.write(pickle.dumps(result))
else:
    print(result)

if args.save:
    logging.info(f'saving result in {filename}.pkl')
    pickle.dump(result, f'{filename}.pkl')

if args.plot:
    import matplotlib.pyplot
    import matplotlib.figure
    import matplotlib.rc

    logging.info(f'saving plots in {filename}.png')

    for n_components, eta in result:
        score = result[(n_components, eta)]
        pyplot.plot(eta, score, 'o', label=f'{n_components} components')

    matplotlib.rc('text', usetex=True)
    pyplot.style.use('ggplot')
    pyplot.figure(figsize=figaspect(1/2.5))
    pyplot.grid(linestyle='--')
    pyplot.xlabel('$\eta$')
    pyplot.ylabel('Adversarial success (\%)')
    pyplot.legend()
    pyplot.savefig(f'{filename}.png')
