#!/usr/bin/env python

import numpy as np
import itertools, argparse, json, logging, os
from re import findall
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import mnist
from models import fc_100_100_10, pca_filtered_model, train, load_from_file
from attacks import adversarial_example, adversarial_score
from utils import dump_json_to_file

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--model', nargs="+", dest="model_paths",
                             help='path to models to attack')
argument_parser.add_argument('--eta', nargs='+', type=float, dest="eta_list",
                             default=np.arange(0, 0.25, 0.01),
                             help='values of eta for generating adv/ examples')
args = argument_parser.parse_args()

PREFIX = os.environ.get('PREFIX', '.')

X_train, y_train, X_test, y_test = mnist()

for model_path in args.model_paths:
    model = load_from_file(model_path)
    print(f"Computing adversarial score against {model.name}...", file=sys.stderr)
    adversarial_score_dictionary = {}
    for eta in args.eta_list:
        score = round(adversarial_score(model, X_test, y_test, eta), 3)
        adversarial_score_dictionary[eta] = score

    print(json.dumps(adversarial_score_dictionary))
    dump_json_to_file(adversarial_score_dictionary, f"{PREFIX}/attack/{model.name}/score.json")
