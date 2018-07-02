from .context import tools
from tools import models
from tools import layers
import keras

import subprocess, os, shutil, pytest
from numpy import allclose

@pytest.fixture(autouse=True)
def teardown():
    shutil.rmtree('/tmp/model/', ignore_errors=True)

def test_pca_when_reconstruction_is_saved(environ):
    command = ['python', 'bin/save_model', '--reconstruction', '--epochs', '0', '--pca', '100', '20']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    assert os.path.exists('/tmp/model/')
    assert os.path.exists('/tmp/model/reconstruction/pca-filtered-model-100-components.h5')
    assert os.path.exists('/tmp/model/reconstruction/pca-filtered-model-20-components.h5')

def test_pca_when_retraining_is_saved(environ):
    command = ['python', 'bin/save_model', '--retraining', '--epochs', '0', '--pca', '100', '20']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    assert os.path.exists('/tmp/model/')
    assert os.path.exists('/tmp/model/retraining/pca-filtered-model-100-components.h5')
    assert os.path.exists('/tmp/model/retraining/pca-filtered-model-20-components.h5')

def test_no_defense_network_is_saved(environ):
    command = ['python', 'bin/save_model', '--no-defense', '--epochs', '0']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    assert os.path.exists('/tmp/model/')
    assert os.path.exists('/tmp/model/fc-100-100-10.h5')

def test_models_with_same_random_state_have_same_loss_and_accuracy(environ, mnist):
    _, _, X_test, y_test = mnist
    one_hot_y_test = keras.utils.to_categorical(y_test, 10)
    command = ['python', 'bin/save_model', '--retraining', '--pca', '20', '--epochs', '2', '--random-state', '1234']

    process = subprocess.run(command, stdout=subprocess.PIPE)
    model = keras.models.load_model('/tmp/model/retraining/pca-filtered-model-20-components.h5', custom_objects={'PCA': layers.PCA})
    expected = model.evaluate(X_test, one_hot_y_test)

    os.remove('/tmp/model/retraining/pca-filtered-model-20-components.h5')

    process = subprocess.run(command, stdout=subprocess.PIPE)
    model = keras.models.load_model('/tmp/model/retraining/pca-filtered-model-20-components.h5', custom_objects={'PCA': layers.PCA})
    actual = model.evaluate(X_test, one_hot_y_test)

    assert allclose(expected, actual, atol=0.001)
