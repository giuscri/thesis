import subprocess, json, os, pytest, shutil
import numpy as np


@pytest.fixture(autouse=True)
def prepare_and_teardown():
    os.environ["PREFIX"] = "/tmp"
    command = ["python", "bin/train-model", "--epochs", "10", "--retraining",
               "--pca", "--n-components", "784", "100"]
    subprocess.run(command)
    yield
    shutil.rmtree("/tmp/model", ignore_errors=True)
    shutil.rmtree("/tmp/attack", ignore_errors=True)


def test_adversarial_score_when_using_reconstruction_defense(environ):
    command = [
        "python",
        "bin/attack-via-fgs",
        "--model",
        "/tmp/model/pca-filtered-model-784-components-reconstruction",
        "/tmp/model/pca-filtered-model-100-components-reconstruction",
        "--eta",
        "0.05",
        "0.1",
        "0.2",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    result = json.loads(process.stdout.decode())

    assert len(result.keys()) == 2

    expected = np.array([22, 53, 93])
    scoredict = result["pca-filtered-model-784-components-reconstruction"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected)

    expected = np.array([15, 37, 82])
    scoredict = result["pca-filtered-model-100-components-reconstruction"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected)


def test_adversarial_score_when_using_retrain_defense(environ):
    command = [
        "python",
        "bin/attack-via-fgs",
        "--model",
        "/tmp/model/pca-filtered-model-784-components-retraining",
        "/tmp/model/pca-filtered-model-100-components-retraining",
        "--eta",
        "0.05",
        "0.1",
        "0.2",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    result = json.loads(process.stdout)

    assert len(result.keys()) == 2

    expected = np.array([21, 61, 95])
    scoredict = result["pca-filtered-model-784-components-retraining"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected)

    expected = np.array([15, 44, 87])
    scoredict = result["pca-filtered-model-100-components-retraining"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected)
