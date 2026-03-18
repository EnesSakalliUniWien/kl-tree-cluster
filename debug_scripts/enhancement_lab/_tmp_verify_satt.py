"""Quick verification: Satterthwaite == whitened when k=1."""

import numpy as np
from scipy.stats import chi2

# k=1 case
lam = np.array([3.5])
w = np.array([2.1])
Tw = float(np.sum(w**2 / lam))
pw = chi2.sf(Tw, 1)
c = float(np.sum(lam**2) / np.sum(lam))
nu = float(np.sum(lam) ** 2 / np.sum(lam**2))
Ts = float(np.sum(w**2))
ps = chi2.sf(Ts / c, nu)
print(f"k=1: whitened_p={pw:.10f} satt_p={ps:.10f} same={abs(pw - ps) < 1e-15}")

# k=2 case (non-uniform eigenvalues -> different)
lam2 = np.array([5.0, 1.0])
w2 = np.array([2.0, 1.5])
Tw2 = float(np.sum(w2**2 / lam2))
pw2 = chi2.sf(Tw2, 2)
c2 = float(np.sum(lam2**2) / np.sum(lam2))
nu2 = float(np.sum(lam2) ** 2 / np.sum(lam2**2))
Ts2 = float(np.sum(w2**2))
ps2 = chi2.sf(Ts2 / c2, nu2)
print(f"k=2: whitened_p={pw2:.10f} satt_p={ps2:.10f} differ={abs(pw2 - ps2) > 1e-6}")

# k=2 uniform eigenvalues -> should be same again
lam3 = np.array([3.0, 3.0])
w3 = np.array([2.0, 1.5])
Tw3 = float(np.sum(w3**2 / lam3))
pw3 = chi2.sf(Tw3, 2)
c3 = float(np.sum(lam3**2) / np.sum(lam3))
nu3 = float(np.sum(lam3) ** 2 / np.sum(lam3**2))
Ts3 = float(np.sum(w3**2))
ps3 = chi2.sf(Ts3 / c3, nu3)
print(f"k=2_uniform: whitened_p={pw3:.10f} satt_p={ps3:.10f} same={abs(pw3 - ps3) < 1e-15}")
print("\nConclusion: Satterthwaite degenerates to whitened when k=1 (c=lam, nu=1)")
print("For k>1 with non-uniform eigenvalues, they differ.")
