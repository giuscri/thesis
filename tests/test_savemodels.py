from .context import tools
from tools import models

import subprocess, os, shutil, pytest

@pytest.fixture
def environ():
    yield os.environ
    if 'PREFIX' in os.environ:
        del os.environ['PREFIX']

def test_common_run(environ):
    assert not 'PREFIX' in environ
    environ['PREFIX'] = '/tmp'
    command = ['python', 'bin/savemodels.py', '-ep', '0', '-ow', '-retpca', '784', '331', '100', '80', '60', '40', '20', '-recpca', '784', '331', '100', '80', '60', '40', '20']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    os.path.exists('/tmp/model/')
    os.path.exists('/tmp/model/vanilla.h5')
    os.path.exists('/tmp/model/pca/retrain/784.h5')
    os.path.exists('/tmp/model/pca/retrain/331.h5')
    os.path.exists('/tmp/model/pca/retrain/100.h5')
    os.path.exists('/tmp/model/pca/retrain/80.h5')
    os.path.exists('/tmp/model/pca/retrain/60.h5')
    os.path.exists('/tmp/model/pca/retrain/40.h5')
    os.path.exists('/tmp/model/pca/retrain/20.h5')
    os.path.exists('/tmp/model/pca/reconstruction/784.h5')
    os.path.exists('/tmp/model/pca/reconstruction/331.h5')
    os.path.exists('/tmp/model/pca/reconstruction/100.h5')
    os.path.exists('/tmp/model/pca/reconstruction/80.h5')
    os.path.exists('/tmp/model/pca/reconstruction/60.h5')
    os.path.exists('/tmp/model/pca/reconstruction/40.h5')
    os.path.exists('/tmp/model/pca/reconstruction/20.h5')
    shutil.rmtree('/tmp/model/')

def test_no_networks_specified(environ):
    assert not 'PREFIX' in environ
    environ['PREFIX'] = '/tmp'
    command = ['python', 'bin/savemodels.py']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 1

def test_vanilla_network_is_saved(environ):
    assert not 'PREFIX' in environ
    environ['PREFIX'] = '/tmp'
    command = ['python', 'bin/savemodels.py', '-van', '-ep', '1']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    os.path.exists('/tmp/model/')
    os.path.exists('/tmp/model/vanilla.h5')
    shutil.rmtree('/tmp/model/')

def test_do_not_overwrite_if_not_forced(environ):
    assert not 'PREFIX' in environ
    environ['PREFIX'] = '/tmp'
    command = ['python', 'bin/savemodels.py', '-ep', '0', '-van', '-retpca', '100', '-recpca', '100']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 0
    os.path.exists('/tmp/model/')
    os.path.exists('/tmp/model/vanilla.h5')
    os.path.exists('/tmp/model/pca/retrain/100.h5')
    os.path.exists('/tmp/model/pca/reconstruction/100.h5')

    command = ['python', 'bin/savemodels.py', '-ep', '0', '-van', '-retpca', '100', '-recpca', '100']
    process = subprocess.run(command, stdout=subprocess.PIPE)
    assert process.returncode == 1

    os.path.exists('/tmp/model/')
