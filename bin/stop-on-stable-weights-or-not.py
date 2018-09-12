#!/usr/bin/env python

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import fc_100_100_10, train, accuracy
from datasets import mnist

kwargs_list = [
    {"epochs": 1000},
    {"epochs": 500},
    {"epochs": 500, "reduce_lr_on_plateau": True,
     "reduce_lr_on_plateau_patience": 30},
    {"epochs": -1, "early_stopping": True,
     "early_stopping_patience": 60}, # 60 because that's a ~whole minute of training on GCP
    {"epochs": -1, "stop_on_stable_weights": True,
     "stop_on_stable_weights_patience": 60},
    {"epochs": -1, "early_stopping": True, "reduce_lr_on_plateau": True,
     "early_stopping_patience": 60},
    {"epochs": -1, "stop_on_stable_weights": True, "reduce_lr_on_plateau": True,
     "reduce_lr_on_plateau_patience": 30},
]
accuracies_list = []
epochs_list = []

X_train, y_train, X_test, y_test = mnist()

for kwargs in kwargs_list:
    model = fc_100_100_10()
    history = train(model, X_train, y_train, **kwargs)
    n_epochs = len(history.epoch)
    test_set_accuracy = accuracy(model, X_test, y_test)
    accuracies_list.append(test_set_accuracy)
    epochs_list.append(n_epochs)

print("#" * 80)
for kwargs, test_set_accuracy, epochs in zip(kwargs_list, accuracies_list, epochs_list):
    print(f"{kwargs} -> {test_set_accuracy}, trained for {epochs} epochs")
    print("#" * 80)
