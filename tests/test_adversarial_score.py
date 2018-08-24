import subprocess, json, os, pytest, shutil
import numpy as np


@pytest.fixture(autouse=True)
def prepare_and_teardown():
    os.environ["PREFIX"] = "/tmp"
    command = ["python", "bin/train-model", "--epochs", "10",
               "--pca", "--n-components", "784"]
    subprocess.run(command)
    yield
    shutil.rmtree("/tmp/model", ignore_errors=True)
    shutil.rmtree("/tmp/attack", ignore_errors=True)


def test_adversarial_score_when_using_reconstruction_defense(environ):
    command = [
        "python",
        "bin/adversarial-score",
        "--model",
        "/tmp/model/pca-filtered-model-784-components-reconstruction",
        "--eta",
        "0.05",
        "0.1",
        "0.2",
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    adversarial_score_dictionary = json.loads(process.stdout)

    assert len(adversarial_score_dictionary.keys()) == 3

    expected = np.array([0.22, 0.53, 0.93])
    actual = np.array([adversarial_score_dictionary["0.05"],
                       adversarial_score_dictionary["0.1"],
                       adversarial_score_dictionary["0.2"]])
    assert np.allclose(actual, expected, atol=0.01)
