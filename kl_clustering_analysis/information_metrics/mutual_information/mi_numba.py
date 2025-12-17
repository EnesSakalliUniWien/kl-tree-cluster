"""Numba-accelerated Mutual Information (MI) calculations for binary data."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange  # type: ignore

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:

    @njit(fastmath=True)  # type: ignore[misc]
    def _calculate_marginal_probabilities_x(
        x_vector: np.ndarray, n_samples: int
    ) -> tuple[float, float, int]:
        """
        Compute marginal probabilities and count of ones for the reference vector X.

        Parameters
        ----------
        x_vector : np.ndarray
            The binary reference vector X.
        n_samples : int
            Total number of samples (length of the vector).

        Returns
        -------
        tuple[float, float, int]
            - prob_x_zero: Probability P(X=0).
            - prob_x_one: Probability P(X=1).
            - count_x_ones: Total count of 1s in X.
        """
        # Accumulate the number of 1s in X
        count_x_ones = 0
        for k in range(n_samples):
            count_x_ones += x_vector[k]

        # Calculate marginal probabilities
        prob_x_one = count_x_ones / n_samples
        prob_x_zero = 1.0 - prob_x_one
        return prob_x_zero, prob_x_one, count_x_ones

    @njit(fastmath=True)  # type: ignore[misc]
    def _count_joint_and_y_ones(
        x_vector: np.ndarray, y_vectors: np.ndarray, row_idx: int, n_samples: int
    ) -> tuple[int, int]:
        """
        Compute count of ones in a specific Y vector and joint ones (X=1, Y=1).

        Parameters
        ----------
        x_vector : np.ndarray
            The binary reference vector X.
        y_vectors : np.ndarray
            Matrix of Y vectors.
        row_idx : int
            Index of the current Y vector to process.
        n_samples : int
            Total number of samples.

        Returns
        -------
        tuple[int, int]
            - count_y_ones: Total count of 1s in the current Y vector.
            - count_joint_ones: Count where both X=1 and Y=1.
        """
        count_y_ones = 0
        count_joint_ones = 0

        # Iterate through samples to count events
        for k in range(n_samples):
            y_value = y_vectors[row_idx, k]
            count_y_ones += y_value
            # Bitwise AND is equivalent to multiplication for binary {0,1}
            count_joint_ones += y_value & x_vector[k]

        return count_y_ones, count_joint_ones

    @njit(fastmath=True)  # type: ignore[misc]
    def _calculate_mi_term_contribution(
        joint_count: int, prob_x_marginal: float, prob_y_marginal: float, n_samples: int
    ) -> float:
        """
        Compute a single term of the Mutual Information sum.

        Formula: p(x,y) * log( p(x,y) / (p(x) * p(y)) )

        Parameters
        ----------
        joint_count : int
            Count of the specific joint event (e.g., X=0, Y=0).
        prob_x_marginal : float
            Marginal probability of X for this state.
        prob_y_marginal : float
            Marginal probability of Y for this state.
        n_samples : int
            Total number of samples.

        Returns
        -------
        float
            The contribution of this term to the total MI. Returns 0.0 if the joint
            probability is 0, handling the limit lim(p->0) p*log(p) = 0.
        """
        # Only compute if the joint event occurred and marginals are non-zero
        if joint_count > 0 and prob_x_marginal > 0.0 and prob_y_marginal > 0.0:
            prob_joint = joint_count / n_samples
            return prob_joint * np.log(prob_joint / (prob_x_marginal * prob_y_marginal))
        return 0.0

    @njit(parallel=True, fastmath=True)  # type: ignore[misc]
    def _mi_binary_vec_numba(
        x_vector: np.ndarray, y_vectors: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """
        Numba-accelerated vectorized MI calculation for binary data.

        Calculates Mutual Information between a single binary vector X and a set of
        binary vectors Y using parallel execution.

        Parameters
        ----------
        x_vector : np.ndarray
            Binary vector X of shape (n_samples,). Values in {0, 1}.
        y_vectors : np.ndarray
            Matrix of binary vectors Y of shape (n_vectors, n_samples).

        Returns
        -------
        np.ndarray
            Array of shape (n_vectors,) containing MI values in nats.
        """
        n_vectors = y_vectors.shape[0]
        n_samples = y_vectors.shape[1]
        mi_output = np.zeros(n_vectors, dtype=np.float64)

        if n_samples == 0:
            return mi_output

        # --- Precompute X statistics ---
        # prob_x_zero: P(X=0)
        # prob_x_one: P(X=1)
        # count_x_ones: Count(X=1)
        prob_x_zero, prob_x_one, count_x_ones = _calculate_marginal_probabilities_x(
            x_vector, n_samples
        )

        # --- Parallel Loop over Y vectors ---
        for i in prange(n_vectors):
            # Compute counts for current Y vector
            # count_y_ones: Count(Y=1)
            # count_joint_ones: Count(X=1, Y=1) aka n11
            count_y_ones, count_joint_ones = _count_joint_and_y_ones(
                x_vector, y_vectors, i, n_samples
            )

            # Derive other joint counts using inclusion-exclusion
            # count_x1_y0: Count(X=1, Y=0) = Count(X=1) - Count(X=1, Y=1)
            count_x1_y0 = count_x_ones - count_joint_ones

            # count_x0_y1: Count(X=0, Y=1) = Count(Y=1) - Count(X=1, Y=1)
            count_x0_y1 = count_y_ones - count_joint_ones

            # count_x0_y0: Count(X=0, Y=0) = Total - (n11 + n10 + n01)
            count_x0_y0 = n_samples - (count_joint_ones + count_x1_y0 + count_x0_y1)

            # Compute Y marginals
            prob_y_one = count_y_ones / n_samples
            prob_y_zero = 1.0 - prob_y_one

            # Accumulate MI contributions for all 4 combinations of (x, y)
            mutual_info = 0.0

            # Term 1: x=0, y=0
            mutual_info += _calculate_mi_term_contribution(
                count_x0_y0, prob_x_zero, prob_y_zero, n_samples
            )
            # Term 2: x=0, y=1
            mutual_info += _calculate_mi_term_contribution(
                count_x0_y1, prob_x_zero, prob_y_one, n_samples
            )
            # Term 3: x=1, y=0
            mutual_info += _calculate_mi_term_contribution(
                count_x1_y0, prob_x_one, prob_y_zero, n_samples
            )
            # Term 4: x=1, y=1
            mutual_info += _calculate_mi_term_contribution(
                count_joint_ones, prob_x_one, prob_y_one, n_samples
            )

            mi_output[i] = mutual_info
        return mi_output

else:
    # Stub when Numba is unavailable
    def _mi_binary_vec_numba(x_vector: np.ndarray, y_vectors: np.ndarray) -> np.ndarray:  # type: ignore
        """Stub - Numba path not available."""
        raise RuntimeError("Numba path not available")
