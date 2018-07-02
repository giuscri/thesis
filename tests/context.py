# https://github.com/kennethreitz/python-guide/blob/c78a2e4f7fbdd3c0843cb1399096563775c4cae7/docs/writing/structure.rst#test-suite

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import attacks, datasets, layers, models
