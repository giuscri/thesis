#!/usr/bin/env python

import os, sys, argparse
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import mnist
from models import train, accuracy, load_from_file

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--model', nargs="+", dest="model_paths",
                             help="models to test the accuracy of")

args = argument_parser.parse_args()

if not args.model_paths:
    model_paths = glob("model/*") # compute accuracy for all the found models
else:
    model_paths = args.model_paths

result = []
_, _, X_test, y_test = mnist()
for path in model_paths:
    if not os.path.exists(path):
        continue

    print(f"Computing for {path}...", end="")
    sys.stdout.flush()
    model = load_from_file(path)
    test_set_accuracy = accuracy(model, X_test, y_test)
    result.append((path, test_set_accuracy))
    print()

result.sort(key=lambda pair: pair[1], reverse=True)
for pair in result:
    path, test_set_accuracy = pair
    print(f"{path} -> {test_set_accuracy}")
