import numpy as np

# Parameters
M = 2000  # High dim
k = 50  # Low dim

print(f"Testing Dimensions: M={M}, k={k}")

# 1. Generate random unit vector Z (Fixed vector)
Z = np.random.randn(M)
Z = Z / np.linalg.norm(Z)

norms_orth = []
norms_gauss = []

for _ in range(500):
    # Gaussian Matrix G
    G = np.random.randn(k, M)

    # QR Decomposition to get R (Orthonormal rows)
    # QR of G^T gives Q (M x M or M x k). We need the column space.
    Q_full, _ = np.linalg.qr(G.T)
    # Q_full is M x k (if mode='reduced' default). Columns are orthonormal.
    # So R_proj = Q_full.T is k x M. Rows are orthonormal.
    R = Q_full.T

    Y_o = R @ Z
    Y_g = G @ Z

    norms_orth.append(np.linalg.norm(Y_o) ** 2)
    norms_gauss.append(np.linalg.norm(Y_g) ** 2)

mean_orth = np.mean(norms_orth)
mean_gauss = np.mean(norms_gauss)

print(f"Mean Squared Norm (Orthogonal Projection): {mean_orth:.5f}")
print(f"Theory (k/M): {k / M:.5f}")
print(f"Mean Squared Norm (Raw Gaussian): {mean_gauss:.5f}")
print(f"Theory (k): {k:.5f}")

# Distribution Check for Gaussian
# Expect chi-square with k degrees of freedom (mean k, var 2k)
var_gauss = np.var(norms_gauss)
print(f"Var Gaussian: {var_gauss:.5f}, Theory (2k): {2 * k:.5f}")

# Distribution Check for Orthogonal
# Expect Beta(k/2, (M-k)/2) ... mean k/M.
# Beta variance: (alpha*beta) / ((alpha+beta)^2 (alpha+beta+1))
# alpha = k/2, beta = (M-k)/2. alpha+beta = M/2.
# Var = (k/2 * (M-k)/2) / (M^2/4 * (M/2 + 1))
#     = (k(M-k)/4) / (M^2 (M+2)/ 8) ... roughly 2k/M^2
var_orth = np.var(norms_orth)
print(f"Var Orthogonal: {var_orth:.8f}")
