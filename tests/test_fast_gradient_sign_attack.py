import subprocess, json, os, pytest, shutil
import numpy as np


@pytest.fixture(autouse=True)
def teardown():
    shutil.rmtree("/tmp/fast-gradient-sign", ignore_errors=True)


def test_fgs_adversarial_score_when_using_reconstruction_defense(environ):
    command = [
        "python",
        "bin/fast_gradient_sign",
        "--model",
        "tests/model/reconstruction/pca-filtered-model-784-components.h5",
        "tests/model/reconstruction/pca-filtered-model-100-components.h5",
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
    scoredict = result["reconstruction/pca-filtered-model-784-components"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected, atol=5)

    expected = np.array([17, 43, 88])
    scoredict = result["reconstruction/pca-filtered-model-100-components"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected, atol=5)


def test_fgs_adversarial_score_when_using_retrain_defense(environ):
    command = [
        "python",
        "bin/fast_gradient_sign",
        "--model",
        "tests/model/retraining/pca-filtered-model-784-components.h5",
        "tests/model/retraining/pca-filtered-model-100-components.h5",
        "--eta",
        "0.05",
        "0.1",
        "0.2",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    result = json.loads(process.stdout)

    assert len(result.keys()) == 2

    expected = np.array([21, 61, 94])
    scoredict = result["retraining/pca-filtered-model-784-components"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected, atol=5)

    expected = np.array([16, 48, 91])
    scoredict = result["retraining/pca-filtered-model-100-components"]
    actual = np.array([scoredict["0.05"], scoredict["0.1"], scoredict["0.2"]])
    assert np.allclose(actual, expected, atol=5)


@pytest.mark.skipif("DISPLAY" in os.environ, reason="blocks test suite inside X")
def test_trying_to_plot_will_raise_an_error(environ):
    command = [
        "python",
        "bin/fast_gradient_sign",
        "--model",
        "tests/model/reconstruction/pca-filtered-model-784-components.h5",
        "--eta",
        "0.05",
        "--plot",
    ]
    process = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    assert process.returncode == 1

    assert (
        "DISPLAY" in os.environ or b"no $DISPLAY environment variable" in process.stderr
    )
