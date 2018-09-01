from .context import models

import os, shutil
import pytest

from sklearn.decomposition import (PCA, FastICA, IncrementalPCA,
                                   KernelPCA, TruncatedSVD, NMF)

from datasets import mnist
from models import (fc_100_100_10, pca_filtered_model, train, accuracy,
                    fastica_filtered_model, kernelpca_filtered_model,
                    incrementalpca_filtered_model, nmf_filtered_model,
                    truncatedsvd_filtered_model, save_to_file, load_from_file)

@pytest.fixture(autouse=True, scope="module")
def save_models():
    X_train, y_train, X_test, y_test = mnist()
    prefix = "/tmp"

    model = pca_filtered_model(fc_100_100_10(), X_train, 10)
    save_to_file(model, prefix)

    model = fastica_filtered_model(fc_100_100_10(), X_train, 10)
    save_to_file(model, prefix)

    model = truncatedsvd_filtered_model(fc_100_100_10(), X_train, 10)
    save_to_file(model, prefix)

    model = kernelpca_filtered_model(fc_100_100_10(), X_train[:1000], 10)
    save_to_file(model, prefix)

    model = incrementalpca_filtered_model(fc_100_100_10(), X_train, 10)
    save_to_file(model, prefix)

    model = nmf_filtered_model(fc_100_100_10(), X_train, 10)
    save_to_file(model, prefix)

    yield

    shutil.rmtree("/tmp/model/pca-filtered-model-10-components")
    shutil.rmtree("/tmp/model/fastica-filtered-model-10-components")
    shutil.rmtree("/tmp/model/truncatedsvd-filtered-model-10-components")
    shutil.rmtree("/tmp/model/kernelpca-filtered-model-10-components")
    shutil.rmtree("/tmp/model/incrementalpca-filtered-model-10-components")
    shutil.rmtree("/tmp/model/nmf-filtered-model-10-components")

def test_pca_filtered_model_is_loaded_correctly():
    model = load_from_file("/tmp/model/pca-filtered-model-10-components")
    assert type(model.sklearn_transformer) is PCA

def test_fastica_filtered_model_is_loaded_correctly():
    model = load_from_file("/tmp/model/fastica-filtered-model-10-components")
    assert type(model.sklearn_transformer) is FastICA

def test_nmf_filtered_model_is_loaded_correctly():
    model = load_from_file("/tmp/model/nmf-filtered-model-10-components")
    assert type(model.sklearn_transformer) is NMF

def test_kernelpca_filtered_model_is_loaded_correctly():
    model = load_from_file("/tmp/model/kernelpca-filtered-model-10-components")
    assert type(model.sklearn_transformer) is KernelPCA

def test_truncatedsvd_filtered_model_is_loaded_correctly():
    model = load_from_file("/tmp/model/truncatedsvd-filtered-model-10-components")
    assert type(model.sklearn_transformer) is TruncatedSVD

def test_incrementalpca_filtered_model_is_loaded_correctly():
    model = load_from_file("/tmp/model/incrementalpca-filtered-model-10-components")
    assert type(model.sklearn_transformer) is IncrementalPCA
