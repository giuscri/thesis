import subprocess, json, os, pytest, shutil
import numpy as np


@pytest.fixture(autouse=True)
def teardown():
    shutil.rmtree("/tmp/attack", ignore_errors=True)


def test_fgs_adversarial_score_when_using_reconstruction_defense(environ):
    command = [
        "python",
        "bin/attack-via-fgs",
        "--model",
        "tests/model/pca-filtered-model-784-components-reconstruction",
        "tests/model/pca-filtered-model-100-components-reconstruction",
        "--eta",
        "0.05",
        "0.1",
        "0.2",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    result = json.loads(process.stdout.decode())

    assert len(result.keys()) == 2

    expected = np.array([22, 53, 94])
    scoredict = result["pca-filtered-model-784-components-reconstruction"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected, atol=5)

    expected = np.array([17, 43, 88])
    scoredict = result["pca-filtered-model-100-components-reconstruction"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected, atol=5)


def test_fgs_adversarial_score_when_using_retrain_defense(environ):
    command = [
        "python",
        "bin/attack-via-fgs",
        "--model",
        "tests/model/pca-filtered-model-784-components-retraining",
        "tests/model/pca-filtered-model-100-components-retraining",
        "--eta",
        "0.05",
        "0.1",
        "0.2",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    result = json.loads(process.stdout)

    assert len(result.keys()) == 2

    score_when_reconstruction = np.array([22, 53, 94])
    scoredict = result["pca-filtered-model-784-components-retraining"]
    score_when_retraining = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    tolerance = 5
    assert np.all(score_when_retraining < score_when_reconstruction + tolerance)

    score_when_reconstruction = np.array([17, 43, 88])
    scoredict = result["pca-filtered-model-100-components-retraining"]
    score_when_retraining = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    tolerance = 5
    assert np.all(score_when_retraining < score_when_reconstruction + tolerance)
