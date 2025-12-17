import numpy as np
from kl_clustering_analysis.information_metrics.mutual_information.cmi import (
    _cmi_binary_vec,
)
from kl_clustering_analysis.information_metrics.mutual_information.permutation import (
    _perm_test_cmi_binary,
)


def test_discrete_cmi():
    """Test CMI with non-binary discrete data."""
    print("Testing discrete CMI...")

    # Case 1: X = Y, Z is random (3 categories)
    # I(X;Y|Z) should be high
    N = 100
    x = np.random.randint(0, 5, N)  # 5 categories
    y = x.copy()
    z = np.random.randint(0, 3, N)  # 3 categories

    cmi_val = _cmi_binary_vec(x, y.reshape(1, -1), z)[0]
    print(f"CMI(X;X|Z_random) [5 cats]: {cmi_val:.4f}")
    assert cmi_val > 0.5, "CMI should be high for identical X,Y"

    # Case 2: X indep Y given Z
    # Z determines X and Y partially, but X and Y are conditionally independent
    # Let's make X and Y random given Z
    x = np.zeros(N, dtype=int)
    y = np.zeros(N, dtype=int)
    z = np.random.randint(0, 3, N)

    for val in [0, 1, 2]:
        mask = z == val
        n_stratum = mask.sum()
        x[mask] = np.random.randint(0, 4, n_stratum)
        y[mask] = np.random.randint(0, 4, n_stratum)

    cmi_val_indep = _cmi_binary_vec(x, y.reshape(1, -1), z)[0]
    print(f"CMI(X;Y|Z) [indep]: {cmi_val_indep:.4f}")
    assert cmi_val_indep < 0.2, "CMI should be low for conditionally independent vars"

    # Test Permutation Test
    print("\nTesting Permutation Test with discrete data...")
    obs, p_val = _perm_test_cmi_binary(x, y, z, permutations=100, random_state=42)
    print(f"Permutation Test (Indep): Obs={obs:.4f}, p-val={p_val:.4f}")
    assert p_val > 0.05, "Should fail to reject null hypothesis for independent data"

    # Test Permutation Test (Dependent)
    y_dep = x.copy()
    obs_dep, p_val_dep = _perm_test_cmi_binary(
        x, y_dep, z, permutations=100, random_state=42
    )
    print(f"Permutation Test (Dep): Obs={obs_dep:.4f}, p-val={p_val_dep:.4f}")
    assert p_val_dep < 0.05, "Should reject null hypothesis for dependent data"


if __name__ == "__main__":
    test_discrete_cmi()
