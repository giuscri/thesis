# reproducing some results from 1704.02654.pdf

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect

import numpy as np

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import json

from datasets import mnist
from filters import pcafilter
from models import fc100_100_10, filtered_fc, train
from attacks import fastgradientsign, adversarial_score

X_train, y_train, X_test, y_test = mnist()
network = fc100_100_10()
train(network, X_train, y_train, epochs=500, store=False)

def score(network, eta, X_test, y_test):
    """Score adv/ success rate for specific n_components and eta."""
    logging.info(f'generating adv/ examples using FGS with eta={round(eta, 3)}')
    attack = lambda n, X: fastgradientsign(n, X, eta=eta)
    return int(100 * adversarial_score(filtered_network, X_test, y_test, attack))

plt.style.use('ggplot')
rc('text', usetex=True)
plt.figure(figsize=figaspect(1/2.5))

reconstruction_princeton = {}

for n_components in (784, 331, 100, 80, 60, 40, 20):
    logging.info(f"filtering input retaining {n_components} principal components")

    filterfn = pcafilter(X_train, n_components=n_components)
    filtered_network = filtered_fc(network, filterfn)

    result = {}
    result[0.] = 0

    step = 0.025
    for eta in np.arange(0.025, 0.25 + step, step):
        result[eta] = score(filtered_network, eta, X_test, y_test)

    reconstruction_princeton[n_components] = tuple(result.items())

logging.info(f'saving results in ./reconstruction-princeton.json')
with open('reconstruction-princeton.json', 'w') as f: f.write(json.dumps(reconstruction_princeton))

for n_components in reconstruction_princeton.keys():
    result = dict(reconstruction_princeton[n_components])
    plt.plot(result.keys(), result.values(), 'o', label=f'{n_components} components')

logging.info(f'saving plots in ./reconstruction-princeton.png')
plt.grid(linestyle='--')
plt.xlabel('$\eta$')
plt.ylabel('Adversarial success (\%)')
plt.legend()
plt.savefig('./reconstruction-princeton.png')
