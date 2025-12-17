import numpy as np
from kl_clustering_analysis.information_metrics.mutual_information.mi import (
    _mi_binary_vec,
)
from kl_clustering_analysis.information_metrics.mutual_information.mi_numpy import (
    _mi_binary_vec_numpy,
)
from kl_clustering_analysis.information_metrics.mutual_information.mi_sklearn import (
    _mi_binary_vec_sklearn,
)


def test_sklearn_vs_numpy():
    """
    Verify that the Sklearn implementation matches the NumPy implementation.
    Both should compute exact MI for discrete binary data.
    """
    np.random.seed(42)
    n_samples = 200
    n_vectors = 20

    x = np.random.randint(0, 2, n_samples).astype(np.uint8)
    Y = np.random.randint(0, 2, (n_vectors, n_samples)).astype(np.uint8)

    mi_numpy = _mi_binary_vec_numpy(x, Y)
    mi_sklearn = _mi_binary_vec_sklearn(x, Y)

    # Sklearn returns nats. Our numpy implementation also returns nats (log base e).
    # Let's verify they are close.
    np.testing.assert_allclose(mi_sklearn, mi_numpy, rtol=1e-10, atol=1e-10)


def test_dispatcher_uses_sklearn():
    """
    Verify that the main dispatcher _mi_binary_vec uses the sklearn implementation
    (or at least returns the same result).
    """
    np.random.seed(123)
    n_samples = 100
    n_vectors = 10

    x = np.random.randint(0, 2, n_samples).astype(np.uint8)
    Y = np.random.randint(0, 2, (n_vectors, n_samples)).astype(np.uint8)

    mi_dispatched = _mi_binary_vec(x, Y)
    mi_sklearn = _mi_binary_vec_sklearn(x, Y)

    np.testing.assert_allclose(mi_dispatched, mi_sklearn, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    test_sklearn_vs_numpy()
    test_dispatcher_uses_sklearn()
    print("Sklearn integration tests passed!")
