from .context import models

import subprocess, os, shutil, pytest

@pytest.mark.skipif('DOCKER' not in os.environ, reason='overwrites trained models')
def test_savemodels():
    command = 'python savemodels.py -ep 0 -ow -retpca 784 331 100 80 60 40 20 -recpca 784 331 100 80 60 40 20'
    process = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    os.path.exists('model/')
    os.path.exists('model/vanilla.v5')
    os.path.exists('model/pca/retrain/784.v5')
    os.path.exists('model/pca/retrain/331.v5')
    os.path.exists('model/pca/retrain/100.v5')
    os.path.exists('model/pca/retrain/80.v5')
    os.path.exists('model/pca/retrain/60.v5')
    os.path.exists('model/pca/retrain/40.v5')
    os.path.exists('model/pca/retrain/20.v5')
    os.path.exists('model/pca/reconstruction/784.v5')
    os.path.exists('model/pca/reconstruction/331.v5')
    os.path.exists('model/pca/reconstruction/100.v5')
    os.path.exists('model/pca/reconstruction/80.v5')
    os.path.exists('model/pca/reconstruction/60.v5')
    os.path.exists('model/pca/reconstruction/40.v5')
    os.path.exists('model/pca/reconstruction/20.v5')
    shutil.rmtree('model/')
