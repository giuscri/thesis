from binascii import hexlify
import os
import json
import pickle


def dump_pickle_to_file(obj, path):
    dirname, basename = os.path.split(path)
    os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def dump_json_to_file(obj, path):
    dirname, basename = os.path.split(path)
    os.makedirs(dirname, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def load_pickle_from_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json_from_file(path):
    with open(path, "r") as f:
        return json.load(f)
