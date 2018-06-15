from .context import tools
from tools import models

import subprocess, os, shutil, pytest

def test_savemodels():
    assert not 'PREFIX' in os.environ
    os.environ['PREFIX'] = '/tmp'
    command = 'python bin/savemodels.py -ep 0 -ow -retpca 784 331 100 80 60 40 20 -recpca 784 331 100 80 60 40 20'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
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
    del os.environ['PREFIX']

def test_nonetworks():
    assert not 'PREFIX' in os.environ
    os.environ['PREFIX'] = '/tmp'
    command = 'python bin/savemodels.py'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    assert process.returncode == 255
    del os.environ['PREFIX']

def test_vanilla():
    assert not 'PREFIX' in os.environ
    os.environ['PREFIX'] = '/tmp'
    command = 'python bin/savemodels.py -van -ep 1'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    os.path.exists('/tmp/model/')
    os.path.exists('/tmp/model/vanilla.h5')
    shutil.rmtree('/tmp/model/')
    del os.environ['PREFIX']

def test_donot_overwrite():
    assert not 'PREFIX' in os.environ
    os.environ['PREFIX'] = '/tmp'
    command = 'python bin/savemodels.py -ep 0 -van -retpca 100 -recpca 100'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    assert process.returncode == 0
    os.path.exists('/tmp/model/')
    os.path.exists('/tmp/model/vanilla.h5')
    os.path.exists('/tmp/model/pca/retrain/100.h5')
    os.path.exists('/tmp/model/pca/reconstruction/100.h5')

    command = 'python bin/savemodels.py -ep 0 -van -retpca 100 -recpca 100'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    assert process.returncode == 255

    os.path.exists('/tmp/model/')
    del os.environ['PREFIX']
