#!/usr/bin/env python

from sklearn.decomposition import (FastICA, IncrementalPCA, KernelPCA, NMF,
                                   TruncatedSVD, PCA)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle, argparse
from datasets import mnist

def name(transformer):
    n_components = transformer.n_components
    return f"{transformer.__class__.__name__.lower()}-{n_components}"

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--all", action="store_true")
argument_parser.add_argument("--fastica", action="store_true")
argument_parser.add_argument("--incrementalpca", action="store_true")
argument_parser.add_argument("--kernelpca", action="store_true")
argument_parser.add_argument("--nmf", action="store_true")
argument_parser.add_argument("--truncatedsvd", action="store_true")
argument_parser.add_argument("--pca", action="store_true")

args = argument_parser.parse_args()

if args.all:
    classes = [FastICA, IncrementalPCA, KernelPCA, NMF, TruncatedSVD, PCA]
else:
    classes = []
    if args.fastica:
        classes.append(FastICA)
    if args.incrementalpca:
        classes.append(IncrementalPCA)
    if args.kernelpca:
        classes.append(KernelPCA)
    if args.nmf:
        classes.append(NMF)
    if args.truncatedsvd:
        classes.append(TruncatedSVD)
    if args.pca:
        classes.append(PCA)

X_train, _, _, _ = mnist()

transformers = []
for n_components in [784, 331, 100, 80, 60, 20]:
    for c in classes:
        if c is TruncatedSVD and n_components == 784:
            continue

        if c is KernelPCA:
            transformer = c(n_components=n_components, fit_inverse_transform=True, n_jobs=-1, random_state=1234)
        elif c is IncrementalPCA:
            transformer = c(n_components=n_components)
        else:
            transformer = c(n_components=n_components, random_state=1234)

        transformers.append(transformer)

for transformer in transformers:
    print(f"Fitting {transformer}...", end="")
    sys.stdout.flush()
    transformer.fit(X_train.reshape(-1, 784))
    print()

os.makedirs("cache/", exist_ok=True)
for transformer in transformers:
    filename = f"cache/{name(transformer)}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(transformer, f)
