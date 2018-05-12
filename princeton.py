#!/usr/bin/env python

import numpy as np

import itertools, argparse, pickle, sys, logging, os

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
parser.add_argument('--retrain', action='store_true', help='use retrain defense technique')
parser.add_argument('--pickle', action='store_true', help='print result encoded as pickle')

parser.add_argument('-c', nargs='+', type=int, default=[784, 331, 100, 80, 60, 40, 20], help='number of principal components to retain')
parser.add_argument('-e', nargs='+', type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25], help='value of eta to use for FGS')

args = parser.parse_args()

filename = 'princeton.retrain' if args.retrain else 'princeton.recons'

X_train, y_train, X_test, y_test = mnist()
network = fc100_100_10()

if not args.retrain: # train networks only once
    train(network, X_train, y_train, epochs=args.epochs, verbose=args.verbose)

networkdict = {}

for n_components in args.c:
    logging.info(f'building network with filter layer of {n_components}')
    filterfn = pcafilter(X_train, n_components=n_components)
    filtered_network = filtered_fc(network, filterfn)

    if args.retrain:
        logging.info(f'training network with filter layer of {n_components}')
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
    with open(f'{filename}.pkl', 'wb') as f: pickle.dump(result, f)

if args.plot:
    from matplotlib import rc
    rc('text', usetex=True)

    from matplotlib.figure import figaspect
    from matplotlib import pyplot as plt # awful standard alias ...

    plt.style.use('ggplot')
    plt.figure(figsize=figaspect(1/2.5))
    plt.grid(linestyle='--')
    plt.xlabel('$\eta$')
    plt.ylabel('Adversarial success (\%)')

    for n_components in args.c:
        x, y = [], []
        for _, eta in filter(lambda k: k[0] == n_components, result):
            x.append(eta)
            y.append(result[(n_components, eta)])
        plt.plot(x, y, 'o', label=f'{n_components} components')

    plt.legend()

    if args.save:
        logging.info(f'saving plot in {filename}.png')
        plt.savefig(f'{filename}.png')

    plt.show()
