#!/usr/bin/env python

from argparse import ArgumentParser
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import mnist
from attacks import adversarial_example
from models import load_from_file

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

argument_parser = ArgumentParser()
argument_parser.add_argument('--eta', required=True, type=float,
                             help="value of eta parameter for FGS")
argument_parser.add_argument("--model", dest="model_path", required=True,
                             help="path to the model to generate input against")

args = argument_parser.parse_args()

_, _, X_test, y_test = mnist()

original_input = X_test[0]

model = load_from_file(args.model_path)
adversarial_input = adversarial_example(model, [original_input], [7], eta=args.eta)[0]

plt.imshow(adversarial_input)
figure_name_eta = "".join(f"{args.eta}".split("."))
figure_name = f"adversarial_input.{model.name}.{figure_name_eta}.png"
print(f"Saving {figure_name}...", end="")
plt.savefig(figure_name)
print()

plt.imshow(adversarial_input)
plt.show()
