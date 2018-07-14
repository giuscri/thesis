from .context import utils
from utils import dump_json_to_file
from utils import dump_pickle_to_file
from utils import load_json_from_file
from utils import load_pickle_from_file
import json
import pickle
import os


class Cat:
    def __init__(self):
        self.color = "ginger"


def test_dump_json_to_file():
    obj = {"hello": "world"}
    dump_json_to_file(obj, "/tmp/dump_json_to_file/example.json")

    with open("/tmp/dump_json_to_file/example.json", "r") as f:
        actual = json.loads(f.read())

    assert actual == obj


def test_dump_pickle_to_file():
    cat = Cat()
    dump_pickle_to_file(cat, "/tmp/dump_pickle_to_file/example.pkl")

    with open("/tmp/dump_pickle_to_file/example.pkl", "rb") as f:
        actual = pickle.load(f)

    assert type(actual) is Cat
    assert "color" in actual.__dict__


def test_load_json_from_file():
    os.makedirs("/tmp/load_json_from_file", exist_ok=True)
    with open("/tmp/load_json_from_file/example.json", "w") as f:
        json.dump({"hello": "world"}, f)

    actual = load_json_from_file("/tmp/load_json_from_file/example.json")
    assert actual == {"hello": "world"}


def test_load_pickle_from_file():
    os.makedirs("/tmp/load_pickle_from_file", exist_ok=True)
    with open("/tmp/load_pickle_from_file/example.pkl", "wb") as f:
        pickle.dump(Cat(), f)

    actual = load_pickle_from_file("/tmp/load_pickle_from_file/example.pkl")

    assert type(actual) is Cat
    assert "color" in actual.__dict__
