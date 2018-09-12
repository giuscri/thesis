#!/usr/bin/env python

import os, sys, pickle
import keras.backend as K
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import mnist
from models import (train, accuracy, save_to_file, fc_100_100_10,
                    pca_filtered_model, fastica_filtered_model,
                    incrementalpca_filtered_model, nmf_filtered_model,
                    truncatedsvd_filtered_model, kernelpca_filtered_model)

def cached(name):
    filename = f"cache/{name}.pkl"
    if not os.path.exists(filename):
        return None

    with open(filename, "rb") as f:
        return pickle.load(f)

argument_parser = ArgumentParser()
argument_parser.add_argument("--pca", action="store_true",
                            help="use PCA image filter defense")
argument_parser.add_argument("--fastica", action="store_true",
                            help="use FastICA image filter defense")
argument_parser.add_argument("--incrementalpca", action="store_true",
                            help="use IncrementalPCA image filter defense")
argument_parser.add_argument("--nmf", action="store_true",
                            help="use IncrementalPCA image filter defense")
argument_parser.add_argument("--truncatedsvd", action="store_true",
                            help="use TruncatedSVD image filter defense")
argument_parser.add_argument("--kernelpca", action="store_true",
                            help="use KernelPCA image filter defense")
argument_parser.add_argument("--n-components", type=int, nargs="+", default=[],
                            help="number of components for image filters")
argument_parser.add_argument("--epochs", type=int, default=-1,
                            help="default: let the model choose")
argument_parser.add_argument("--random-seed", action="store_true",
                            help="initialize model with random seed")
args = argument_parser.parse_args()

PREFIX = os.environ.get('PREFIX', '.')

X_train, y_train, X_test, y_test = mnist()

if not args.random_seed:
    K.clear_session()
    tf.set_random_seed(1234)
    np.random.seed(1234)

no_defense_model = fc_100_100_10()
print(f"Training {no_defense_model.name}...")
train(no_defense_model, X_train, y_train, args.epochs, verbose=True,
      stop_on_stable_weights=True, reduce_lr_on_plateau=True,
      stop_on_stable_weights_patience=60, reduce_lr_on_plateau_patience=30)

print(f"Saving {no_defense_model.name}...")
save_to_file(no_defense_model, PREFIX)

for n_components in args.n_components:
    if args.pca:
        pca = cached(f"pca-{n_components}")
        filtered_model = pca_filtered_model(no_defense_model, X_train,
                                            n_components, pca=pca)

        print(f"Saving {filtered_model.name}...")
        save_to_file(filtered_model, PREFIX)

    if args.fastica:
        fastica = cached(f"fastica-{n_components}")
        filtered_model = fastica_filtered_model(no_defense_model, X_train,
                                                n_components, fastica=fastica)

        print(f"Saving {filtered_model.name}...")
        save_to_file(filtered_model, PREFIX)

    if args.incrementalpca:
        incrementalpca = cached(f"incrementalpca-{n_components}")
        filtered_model = incrementalpca_filtered_model(no_defense_model, X_train,
                                                       n_components,
                                                       incrementalpca=incrementalpca)

        print(f"Saving {filtered_model.name}...")
        save_to_file(filtered_model, PREFIX)

    if args.nmf:
        nmf = cached(f"nmf-{n_components}")
        filtered_model = nmf_filtered_model(no_defense_model, X_train,
                                            n_components, nmf=nmf)

        print(f"Saving {filtered_model.name}...")
        save_to_file(filtered_model, PREFIX)

    if args.truncatedsvd:
        truncatedsvd = cached(f"truncatedsvd-{n_components}")
        filtered_model = truncatedsvd_filtered_model(no_defense_model, X_train,
                                                     n_components,
                                                     truncatedsvd=truncatedsvd)

        print(f"Saving {filtered_model.name}...")
        save_to_file(filtered_model, PREFIX)

    if args.kernelpca:
        kernelpca = cached(f"kernelpca-{n_components}")
        filtered_model = kernelpca_filtered_model(no_defense_model, X_train,
                                                  n_components, kernelpca=kernelpca)

        print(f"Saving {filtered_model.name}...")
        save_to_file(filtered_model, PREFIX)
