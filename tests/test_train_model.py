import keras

import subprocess, os, shutil, pytest
from pickle import load
from numpy import allclose
from sklearn.decomposition import PCA

from models import fc_100_100_10, pca_filtered_model


@pytest.fixture(autouse=True)
def teardown():
    yield
    shutil.rmtree("/tmp/model/", ignore_errors=True)


def test_pca_when_reconstruction_is_saved(environ):
    command = ["python", "bin/train-model.py", "--epochs", "0", "--pca",
               "--n-components", "100", "20"]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0

    created_files = [
        "/tmp/model/pca-filtered-model-100-components/weights.h5",
        "/tmp/model/pca-filtered-model-100-components/pca.pkl",
        "/tmp/model/pca-filtered-model-20-components/weights.h5",
        "/tmp/model/pca-filtered-model-20-components/pca.pkl",
    ]
    for f in created_files:
        assert os.path.exists(f)


def test_no_defense_network_is_saved(environ):
    command = ["python", "bin/train-model.py", "--epochs", "0"]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0

    assert os.path.exists("/tmp/model/fc-100-100-10/weights.h5")


def test_models_with_same_random_state_have_same_loss_and_accuracy(environ, mnist):
    X_train, _, X_test, y_test = mnist
    one_hot_y_test = keras.utils.to_categorical(y_test, 10)
    command = ["python", "bin/train-model.py", "--pca",
               "--n-components", "20", "--epochs", "2"]

    process = subprocess.run(command, stdout=subprocess.PIPE)
    model = fc_100_100_10()
    model.load_weights("/tmp/model/pca-filtered-model-20-components/weights.h5")
    with open("/tmp/model/pca-filtered-model-20-components/pca.pkl", "rb") as f:
        pca = load(f)
    model = pca_filtered_model(model, X_train, pca=pca)
    expected = model.evaluate(X_test, one_hot_y_test)

    shutil.rmtree("/tmp/model/pca-filtered-model-20-components/")

    process = subprocess.run(command, stdout=subprocess.PIPE)
    model = fc_100_100_10()
    model.load_weights("/tmp/model/pca-filtered-model-20-components/weights.h5")
    with open("/tmp/model/pca-filtered-model-20-components/pca.pkl", "rb") as f:
        pca = load(f)
    model = pca_filtered_model(model, X_train, pca=pca)
    actual = model.evaluate(X_test, one_hot_y_test)

    assert allclose(expected, actual, atol=0.001)
