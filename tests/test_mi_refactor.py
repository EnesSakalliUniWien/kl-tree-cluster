import numpy as np

# import pytest
from kl_clustering_analysis.information_metrics.mutual_information.mi_numba import (
    _mi_binary_vec_numba,
    _NUMBA_AVAILABLE,
)
from kl_clustering_analysis.information_metrics.mutual_information.mi_numpy import (
    _mi_binary_vec_numpy,
)


def test_mi_numba_vs_numpy():
    if not _NUMBA_AVAILABLE:
        print("Numba not available, skipping test")
        return

    np.random.seed(42)
    n_samples = 100
    n_vectors = 50

    x = np.random.randint(0, 2, n_samples).astype(np.uint8)
    Y = np.random.randint(0, 2, (n_vectors, n_samples)).astype(np.uint8)

    mi_numpy = _mi_binary_vec_numpy(x, Y)
    mi_numba = _mi_binary_vec_numba(x, Y)

    np.testing.assert_allclose(mi_numba, mi_numpy, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    test_mi_numba_vs_numpy()
    print("Test passed!")
