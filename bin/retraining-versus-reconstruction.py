#!/usr/bin/env python

import sys, os, json, argparse
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import mnist
from models import train, fc_100_100_10, pca_filtered_model
from attacks import adversarial_score

def dictionary_difference(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    r = {}
    for k in dict1:
        r[k] = dict1[k] - dict2[k]
    return r

def compare_two_models(first, second):
    distance = []

    print("Computing adversarial score for both models...", end="")
    sys.stdout.flush()

    for eta in np.arange(0, 0.25, 0.01):
        reconstructed_model_score = adversarial_score(reconstructed_model, X_test, y_test, eta)
        retrained_model_score = adversarial_score(retrained_model, X_test, y_test, eta)
        distance.append(retrained_model_score - reconstructed_model_score)

    print()

    if np.mean(distance) > 0:
        return first
    else:
        return second

argument_parser = argparse.ArgumentParser()
args = argument_parser.parse_args()

X_train, y_train, X_test, y_test = mnist()

models_win_counter = {
    "reconstructed_model": 0,
    "retrained_model": 0,
    "reretrained_model": 0,
}

for n_components in [784, 331, 100, 80, 60, 20]:
    reconstructed_model = pca_filtered_model(fc_100_100_10(), X_train, n_components)
    train(reconstructed_model, X_train, y_train, epochs=-1, stop_on_stable_weights=True)

    retrained_model = pca_filtered_model(fc_100_100_10(), X_train, n_components)
    X_train = retrained_model.preprocessing_fn(X_train)
    train(retrained_model, X_train, y_train, epochs=-1, stop_on_stable_weights=True)

    best_model = compare_two_models(reconstructed_model, retrained_model)

    reretrained_model = pca_filtered_model(fc_100_100_10(), X_train, n_components)
    train(reretrained_model, X_train, y_train, epochs=-1, stop_on_stable_weights=True)
    X_train = reretrained_model.preprocessing_fn(X_train)
    train(reretrained_model, X_train, y_train, epochs=-1, stop_on_stable_weights=True)

    best_model = compare_two_models(best_model, reretrained_model)
    if best_model is reretrained_model:
        models_win_counter["reretrained_model"] += 1
    elif best_model is reconstructed_model:
        models_win_counter["reconstructed_model"] += 1
    else:
        models_win_counter["retrained_model"] += 1

winner_model = max(models_win_counter, key=lambda k: models_win_counter[k])
if winner_model is reretrained_model:
    print(f"Training twice is more effective")
elif winner_model is reconstructed_model:
    print(f"Reconstruction is more effective")
else:
    print(f"Retraining is more effective")
