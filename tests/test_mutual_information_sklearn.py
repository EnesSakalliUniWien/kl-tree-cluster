import numpy as np

from kl_clustering_analysis.information_metrics.mutual_information import (
    _mi_binary_vec,
    _mi_binary_vec_sklearn,
)


def test_mi_dispatcher_uses_sklearn():
    rng = np.random.default_rng(123)
    n_samples = 100
    n_vectors = 7

    x = rng.integers(0, 2, size=n_samples, dtype=np.uint8)
    Y = rng.integers(0, 2, size=(n_vectors, n_samples), dtype=np.uint8)

    mi_dispatched = _mi_binary_vec(x, Y)
    mi_sklearn = _mi_binary_vec_sklearn(x, Y)

    np.testing.assert_allclose(mi_dispatched, mi_sklearn, rtol=0, atol=0)

