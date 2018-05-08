#!/bin/env python

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import json

import argparse

from datasets import mnist
from filters import pcafilter
from models import fc100_100_10, filtered_fc, train
from attacks import fastgradientsign, adversarial_score

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=500)

parser.add_argument('--plot', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--retrain', action='store_true')

parser.add_argument('-c', nargs='+', type=int, default=[784, 331, 100, 80, 60, 40, 20])
parser.add_argument('-e', nargs='+', type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25])

args = parser.parse_args()

X_train, y_train, X_test, y_test = mnist()
network = fc100_100_10()
train(network, X_train, y_train, epochs=args.epochs, store=False, verbose=args.verbose)

princeton = {}

for n_components in args.c:
    logging.info(f"filtering input retaining {n_components} principal components")

    filterfn = pcafilter(X_train, n_components=n_components)
    filtered_network = filtered_fc(network, filterfn)

    if args.retrain:
        logging.info(f"retraining network with filter layer")
        train(filtered_network, X_train, y_train, epochs=args.epochs, store=False, verbose=args.verbose)

    score_byeta = {}

    for eta in args.e:
        logging.info(f'generating adv/ examples using FGS with eta={round(eta, 3)}')
        attack = lambda n, X: fastgradientsign(n, X, eta=eta)
        score = int(100 * adversarial_score(filtered_network, X_test, y_test, attack))
        score_byeta[eta] = score

    princeton[n_components] = tuple(score_byeta.items())

print(json.dumps(princeton))

if args.save:
    filename = './retrain.json' if args.retrain else './recons.json'
    logging.info(f'saving results in {filename}')
    with open(filename, 'w') as f: f.write(json.dumps(princeton))

if args.plot:
    filename = './retrain.png' if args.retrain else './recons.png'
    logging.info(f'saving plots in f{filename}')

    for n_components in princeton.keys():
        score_byeta = dict(princeton[n_components])
        plt.plot(score_byeta.keys(), score_byeta.values(), 'o', label=f'{n_components} components')

    plt.style.use('ggplot')
    rc('text', usetex=True)
    plt.figure(figsize=figaspect(1/2.5))
    plt.grid(linestyle='--')
    plt.xlabel('$\eta$')
    plt.ylabel('Adversarial success (\%)')
    plt.legend()
    plt.savefig(filename)
