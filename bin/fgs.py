#!/usr/bin/env python

import numpy as np

import itertools, argparse, json, sys, logging, os
from binascii import hexlify

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.datasets import mnist
from tools.models import fc_100_100_10, pca_filtered_model, train, load_from_file
from tools.attacks import fast_gradient_sign, fgs_adversarial_score

PREFIX = os.environ.get('PREFIX', '.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('-plot', action='store_true')
parser.add_argument('-etas', nargs='+', type=float, default=[0.05, 0.1, 0.15, 0.2, 0.25], help='parameters for FGS')
parser.add_argument('-mods', nargs='+', help='HDF5 files to load models from')

arguments = parser.parse_args()
plot = arguments.plot
mods = arguments.mods
etas = arguments.etas

X_train, y_train, X_test, y_test = mnist()

networkdict = {}

for modelname in mods:
    name_without_extension = ''.join(modelname.split('.')[:-1])
    identifier = '-'.join(name_without_extension.split('/')[1:])
    networkdict[identifier] = load_from_file(modelname)

def sortkey(identifier):
    try:
        return int(identifier.split('-')[-1])
    except ValueError:
        return np.inf

IDENTIFIERS = list(networkdict.keys())
IDENTIFIERS.sort(key=sortkey, reverse=True)

result = {}

for identifier, eta in itertools.product(IDENTIFIERS, etas):
    logging.info(f'computing score for combination: ({identifier}, {eta})')
    network = networkdict[identifier]
    score = int(100 * fgs_adversarial_score(network, X_test, y_test, eta))
    inner_result = result.get(identifier, {})
    inner_result[eta] = score
    result[identifier] = inner_result

print(json.dumps(result))

for identifier in IDENTIFIERS:
    fname = identifier.split('-')[-1] + '.json'
    path = f'{PREFIX}/fgs/' + '/'.join(identifier.split('-')[:-1])
    os.makedirs(path, exist_ok=True)
    with open('/'.join([path, fname]), 'w') as f: json.dump(result[identifier], f)

if plot:
    from matplotlib.figure import figaspect
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    plt.figure(figsize=figaspect(1/2.5))
    plt.grid(linestyle='--')
    plt.xlabel('$\eta$')
    plt.ylabel('Adversarial success (\%)')

    for identifier in IDENTIFIERS:
        inner_result = result[identifier]
        x, y = [], []
        for eta in inner_result:
            x.append(eta)
            y.append(inner_result[eta])
        plt.plot(x, y, 'o', label=identifier)

    plt.legend()
    os.makedirs(f'{PREFIX}/fgs/', exist_ok=True)
    plt.savefig(f'{PREFIX}/fgs/{hexlify(os.urandom(32)[:10]).decode()}.png')
    plt.show()
