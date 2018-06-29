from .context import tools
from tools import models

import subprocess, json, os, pytest, shutil
import numpy as np

@pytest.fixture(autouse=True)
def teardown():
    shutil.rmtree('/tmp/fgs', ignore_errors=True)

def test_fgs_adversarial_score_when_using_reconstruction_defense(environ):
    command = ['python', 'bin/fgs.py', '-mods', 'tests/model/pca/reconstruction/784.h5', 'tests/model/pca/reconstruction/100.h5', '-etas', '0.05', '0.1', '0.2']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    result = json.loads(process.stdout.decode())

    assert len(result.keys()) == 2

    expected = np.array([22, 53, 94])
    scoredict = result['model-pca-reconstruction-784']
    actual = np.array([scoredict['0.05'], scoredict['0.1'], scoredict['0.2']])
    assert np.allclose(actual, expected, atol=5)

    expected = np.array([17, 43, 88])
    scoredict = result['model-pca-reconstruction-100']
    actual = np.array([scoredict['0.05'], scoredict['0.1'], scoredict['0.2']])
    assert np.allclose(actual, expected, atol=5)

def test_fgs_adversarial_score_when_using_retrain_defense(environ):
    command = ['python', 'bin/fgs.py', '-mods', 'tests/model/pca/retrain/784.h5', 'tests/model/pca/retrain/100.h5', '-etas', '0.05', '0.1', '0.2']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    result = json.loads(process.stdout)

    assert len(result.keys()) == 2

    expected = np.array([21, 54, 94])
    scoredict = result['model-pca-retrain-784']
    actual = np.array([scoredict['0.05'], scoredict['0.1'], scoredict['0.2']])
    assert np.allclose(actual, expected, atol=5)

    expected = np.array([16, 42, 88])
    scoredict = result['model-pca-retrain-100']
    actual = np.array([scoredict['0.05'], scoredict['0.1'], scoredict['0.2']])
    assert np.allclose(actual, expected, atol=5)

@pytest.mark.skipif('DISPLAY' in os.environ, reason='blocks test suite inside X')
def test_plot_is_saved(environ):
    command = 'python bin/fgs.py -mods tests/model/pca/reconstruction/784.h5 -etas 0.05 -plot'
    command = ['python', 'bin/fgs.py', '-mods', 'tests/model/pca/reconstruction/784.h5', '-etas', '0.05', '-plot']
    process = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    assert process.returncode == 1

    assert 'DISPLAY' in os.environ or b'no $DISPLAY environment variable' in process.stderr # check fgs.py will try to call plt.show()
