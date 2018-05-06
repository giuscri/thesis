import os, requests
from os.path import exists

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

import pandas as pd

import numpy as np

def mnist():
    """Fetch, parse and return mnist data."""
    if not exists('mnist'): os.mkdir('mnist/')
    if not exists('mnist/train.csv'):
        logging.info('downloading mnist training set')
        r = requests.get('https://pjreddie.com/media/files/mnist_train.csv')
        with open('mnist/train.csv', 'w') as f: f.write(r.text)

    if not exists('mnist/test.csv'):
        logging.info('downloading mnist test set')
        r = requests.get('https://pjreddie.com/media/files/mnist_test.csv')
        with open('mnist/test.csv', 'w') as f: f.write(r.text)

    names = ['label'] + [f'pixel{i}' for i in range(784)]

    df = pd.read_csv('mnist/train.csv', names=names, dtype=np.float32)
    label, pixels = df['label'], df.drop('label', axis=1)
    X_train = pixels.values.reshape(-1, 28, 28) / 255
    y_train = label.values

    df = pd.read_csv('mnist/test.csv', names=names, dtype=np.float32)
    label, pixels = df['label'], df.drop('label', axis=1)
    X_test = pixels.values.reshape(-1, 28, 28) / 255
    y_test = label.values

    return X_train, y_train, X_test, y_test
