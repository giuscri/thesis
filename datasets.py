import os, requests
from os.path import exists

import logging

log = logging.getLogger()
log.setLevel("INFO")
log.setFormatter("%(asctime)s %(message)s")
info = lambda message: log.info(message)

import pandas as pd

import numpy as np

from tqdm import tqdm


def mnist():
    """Fetch, parse and return mnist data."""
    if not exists("mnist"):
        os.mkdir("mnist/")
    if not exists("mnist/train.csv"):
        info("downloading mnist training set")
        r = requests.get(
            "https://pjreddie.com/media/files/mnist_train.csv", stream=True
        )
        contentlength = int(r.headers["Content-Length"])
        nchunks = 420
        chunk_size = contentlength // nchunks
        total = nchunks if contentlength % nchunks == 0 else nchunks + 1

        chunks = []
        for i, chunk in tqdm(
            enumerate(r.iter_content(chunk_size=chunk_size)), total=total
        ):
            chunks.append(chunk)

        text = b"".join(chunks).decode()
        with open("mnist/train.csv", "w") as f:
            f.write(text)

    if not exists("mnist/test.csv"):
        info("downloading mnist test set")
        r = requests.get("https://pjreddie.com/media/files/mnist_test.csv", stream=True)
        contentlength = int(r.headers["Content-Length"])
        nchunks = 420
        chunk_size = contentlength // nchunks
        total = nchunks if contentlength % nchunks == 0 else nchunks + 1

        chunks = []
        for i, chunk in tqdm(
            enumerate(r.iter_content(chunk_size=chunk_size)), total=total
        ):
            chunks.append(chunk)

        text = b"".join(chunks).decode()
        with open("mnist/test.csv", "w") as f:
            f.write(text)

    names = ["label"] + [f"pixel{i}" for i in range(784)]

    df = pd.read_csv("mnist/train.csv", names=names, dtype=np.float32)
    label, pixels = df["label"], df.drop("label", axis=1)
    X_train = pixels.values.reshape(-1, 28, 28) / 255
    y_train = label.values

    df = pd.read_csv("mnist/test.csv", names=names, dtype=np.float32)
    label, pixels = df["label"], df.drop("label", axis=1)
    X_test = pixels.values.reshape(-1, 28, 28) / 255
    y_test = label.values

    X_train.flags.writeable = False
    y_train.flags.writeable = False

    X_test.flags.writeable = False
    y_test.flags.writeable = False

    return X_train, y_train, X_test, y_test
