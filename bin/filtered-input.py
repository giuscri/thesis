#!/usr/bin/env python

from argparse import ArgumentParser
import sys, os, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import mnist
from sklearn.decomposition import PCA, FastICA, KernelPCA, IncrementalPCA, NMF, TruncatedSVD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

argument_parser = ArgumentParser()
argument_parser.add_argument("--pca", action="store_true",
                             help="use PCA filter")
argument_parser.add_argument("--fastica", action="store_true",
                             help="use FastICA filter")
argument_parser.add_argument("--incrementalpca", action="store_true",
                             help="use IncrementalPCA filter")
argument_parser.add_argument("--kernelpca", action="store_true",
                             help="use KernelPCA filter")
argument_parser.add_argument("--nmf", action="store_true",
                             help="use NMF filter")
argument_parser.add_argument("--truncatedsvd", action="store_true",
                             help="use TruncatedSVD filter")
argument_parser.add_argument("--n-components", type=int, required=True,
                             help="how many components to retain when filtering")


args = argument_parser.parse_args()

X_train, y_train, X_test, y_test = mnist()

original_input = X_test[0]

cached = False
if args.pca and os.path.exists(f"cache/pca-{args.n_components}.pkl"):
    with open(f"cache/pca-{args.n_components}.pkl", "rb") as f:
        sklearn_transformer = pickle.load(f)
    cached = True
elif args.fastica and os.path.exists(f"cache/fastica-{args.n_components}.pkl"):
    with open(f"cache/fastica-{args.n_components}.pkl", "rb") as f:
        sklearn_transformer = pickle.load(f)
    cached = True
elif args.incrementalpca and os.path.exists(f"cache/incrementalpca-{args.n_components}.pkl"):
    with open(f"cache/incrementalpca-{args.n_components}.pkl", "rb") as f:
        sklearn_transformer = pickle.load(f)
    cached = True
elif args.kernelpca and os.path.exists(f"cache/kernelpca-{args.n_components}.pkl"):
    with open(f"cache/kernelpca-{args.n_components}.pkl", "rb") as f:
        sklearn_transformer = pickle.load(f)
    cached = True
elif args.nmf and os.path.exists(f"cache/nmf-{args.n_components}.pkl"):
    with open(f"cache/nmf-{args.n_components}.pkl", "rb") as f:
        sklearn_transformer = pickle.load(f)
    cached = True
elif args.truncatedsvd and os.path.exists(f"cache/truncatedsvd-{args.n_components}.pkl"):
    with open(f"cache/truncatedsvd-{args.n_components}.pkl", "rb") as f:
        sklearn_transformer = pickle.load(f)
    cached = True
elif args.pca:
    sklearn_transformer = PCA(n_components=args.n_components, svd_solver="full", random_state=1234)
elif args.fastica:
    sklearn_transformer = FastICA(n_components=args.n_components, random_state=1234)
elif args.incrementalpca:
    sklearn_transformer = IncrementalPCA(n_components=args.n_components)
elif args.kernelpca:
    sklearn_transformer = KernelPCA(n_components=args.n_components, random_state=1234, n_jobs=-1)
elif args.nmf:
    sklearn_transformer = NMF(n_components=args.n_components, random_state=1234)
elif args.truncatedsvd:
    sklearn_transformer = TruncatedSVD(n_components=args.n_components, random_state=1234)

if not cached:
    print(f"Fitting {sklearn_transformer.__class__.__name__}...", end="")
    sys.stdout.flush()
    sklearn_transformer.fit(X_train.reshape(-1, 784))
    print()

    os.makedirs("cache/", exist_ok=True)
    with open(f"cache/{sklearn_transformer.__class__.__name__.lower()}-{args.n_components}.pkl", "wb") as f:
        pickle.dump(sklearn_transformer, f)

def filter(image):
    batch = [image.reshape(784)]
    filtered_batch = sklearn_transformer.inverse_transform(sklearn_transformer.transform(batch))
    return filtered_batch[0].reshape(28, 28)

def save(image):
    plt.imshow(image)
    sklearn_transformer_name = sklearn_transformer.__class__.__name__
    n_components = sklearn_transformer.n_components
    figure_name = f"filtered-input-{sklearn_transformer_name.lower()}-{n_components}-components.png"
    print(f"Saving {figure_name}...")
    return plt.savefig(figure_name)

def show(image):
    plt.imshow(image)
    plt.show()

filtered_input = filter(original_input)
save(filtered_input)
show(filtered_input)
