import keras

import subprocess, os, shutil, pytest
from pickle import load
from numpy import allclose
from sklearn.decomposition import PCA

from models import fc_100_100_10, pca_filtered_model


@pytest.fixture(autouse=True)
def teardown():
    shutil.rmtree("/tmp/model/", ignore_errors=True)


def test_pca_when_reconstruction_is_saved(environ):
    command = ["python", "bin/train-model", "--reconstruction", "--epochs", "0", "--pca", "100", "20"]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    assert os.path.exists("/tmp/model/")
    assert os.path.exists("/tmp/model/reconstruction/pca-filtered-model-100-components/")
    assert os.path.exists("/tmp/model/reconstruction/pca-filtered-model-100-components/weights.h5")
    assert os.path.exists("/tmp/model/reconstruction/pca-filtered-model-100-components/pca.pkl")
    assert os.path.exists("/tmp/model/reconstruction/pca-filtered-model-20-components/")
    assert os.path.exists("/tmp/model/reconstruction/pca-filtered-model-20-components/weights.h5")
    assert os.path.exists("/tmp/model/reconstruction/pca-filtered-model-20-components/pca.pkl")


def test_pca_when_retraining_is_saved(environ):
    command = ["python", "bin/train-model", "--retraining", "--epochs", "0", "--pca", "100", "20"]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    assert os.path.exists("/tmp/model/")
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-100-components/")
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-100-components/weights.h5")
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-100-components/pca.pkl")
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-20-components/")
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-20-components/weights.h5")
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-20-components/pca.pkl")


def test_no_defense_network_is_saved(environ):
    command = ["python", "bin/train-model", "--no-defense", "--epochs", "0"]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    assert os.path.exists("/tmp/model/")
    assert os.path.exists("/tmp/model/fc-100-100-10/")
    assert os.path.exists("/tmp/model/fc-100-100-10/weights.h5")


def test_models_with_same_random_state_have_same_loss_and_accuracy(environ, mnist):
    X_train, _, X_test, y_test = mnist
    one_hot_y_test = keras.utils.to_categorical(y_test, 10)
    command = ["python", "bin/train-model", "--retraining", "--pca", "20", "--epochs", "2", "--random-state", "1234"]

    process = subprocess.run(command, stdout=subprocess.PIPE)
    model = fc_100_100_10()
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-20-components/weights.h5")
    model.load_weights("/tmp/model/retraining/pca-filtered-model-20-components/weights.h5")
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-20-components/pca.pkl")
    with open("/tmp/model/retraining/pca-filtered-model-20-components/pca.pkl", "rb") as f:
        pca = load(f)
    model = pca_filtered_model(model, X_train, pca=pca)
    expected = model.evaluate(X_test, one_hot_y_test)

    shutil.rmtree("/tmp/model/retraining/pca-filtered-model-20-components/")

    process = subprocess.run(command, stdout=subprocess.PIPE)
    model = fc_100_100_10()
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-20-components/weights.h5")
    model.load_weights("/tmp/model/retraining/pca-filtered-model-20-components/weights.h5")
    assert os.path.exists("/tmp/model/retraining/pca-filtered-model-20-components/pca.pkl")
    with open("/tmp/model/retraining/pca-filtered-model-20-components/pca.pkl", "rb") as f:
        pca = load(f)
    model = pca_filtered_model(model, X_train, pca=pca)
    actual = model.evaluate(X_test, one_hot_y_test)

    assert allclose(expected, actual, atol=0.001)