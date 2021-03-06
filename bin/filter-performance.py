#!/usr/bin/env python

from argparse import ArgumentParser
import sys, os, pickle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import mnist
from models import load_from_file, accuracy
from attacks import adversarial_score

def key(triplet):
    _, mnist_accuracy, mnist_adversarial_score = triplet

    # we want to stay close to max accuracy while as far
    # as possible from best adv/ score. This assumes accuracy
    # can't be better than the one for unfiltered model, and the attacker
    # can't perform better than what performs against the unfiltered model.
    return 1 - (0.974 - mnist_accuracy) + (0.81 - mnist_adversarial_score)

argument_parser = ArgumentParser()
argument_parser.add_argument("--model", nargs="+", dest="model_paths",
                             required=True, help="path to model to use")
argument_parser.add_argument("--eta", type=float, default=0.1,
                             help="(default: 0.1) value of eta to use for FGS")

args = argument_parser.parse_args()

_, _, X_test, y_test = mnist()

result = []
for model_path in args.model_paths:
    print(f"Computing for {model_path} ...", end="")
    sys.stdout.flush()
    model = load_from_file(model_path)

    mnist_accuracy = accuracy(model, X_test, y_test)
    mnist_adversarial_score = adversarial_score(model, X_test, y_test, eta=0.1)
    result.append((model_path, mnist_accuracy, mnist_adversarial_score))
    print()

result.sort(key=key)

for triplet in result:
    model_path, mnist_accuracy, mnist_adversarial_score = triplet
    print(f"{model_path}: accuracy {mnist_accuracy}, adversarial_score {mnist_adversarial_score}")
