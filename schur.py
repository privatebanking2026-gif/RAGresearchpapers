"""
Schur Complement Method for Computing Conditional Entropy H(R|I).

For a block matrix:
    W_FULL = [ W_II  W_IR ]
             [ W_RI  W_RR ]

The Schur complement is:
    S = W_RR - W_RI @ inv(W_II + eI) @ W_IR

Key Properties:
    1. det(W_FULL) = det(W_II) * det(S)
    2. H(I,R) = H(I) + H(R|I) where H(R|I) = entropy(S)
    3. GUARANTEES H(R|I) >= 0

The Schur complement S represents the "residual" similarity in results
AFTER factoring out the variance from interpretation differences.

References:
    - Zhang, F. (Ed.). (2006). The Schur complement and its applications.
    - Paper Section 3.3: Decomposing Uncertainty with the Schur Complement.
"""

import networkx as nx
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SchurComplementEntropy:
    """
    Compute conditional entropy H(R|I) using Schur complement method.

    Guarantees non-negative H(R|I) by construction.
    """

    def __init__(self, kle_instance):
        """
        Initialize with a KernelLaplacianEntropy instance.

        Args:
            kle_instance: Instance of KernelLaplacianEntropy for entropy computation.
        """
        self.kle = kle_instance

    def compute_schur_complement(
        self,
        w_full: np.ndarray,
        n_i: int,
        n_r: int,
        epsilon: float = 1e-1,
    ) -> np.ndarray:
        """
        Compute Schur complement: S = W_RR - W_RI @ inv(W_II + eI) @ W_IR.

        Args:
            w_full: Full similarity matrix (n_i+n_r x n_i+n_r).
            n_i: Number of interpretations.
            n_r: Number of results.
            epsilon: Regularization for W_II inversion stability.

        Returns:
            Schur complement matrix (n_r x n_r).
        """
        if w_full.shape != (n_i + n_r, n_i + n_r):
            raise ValueError(
                f"Expected W_FULL shape {(n_i + n_r, n_i + n_r)}, got {w_full.shape}"
            )

        w_ii = w_full[:n_i, :n_i]
        w_ir = w_full[:n_i, n_i:]
        w_ri = w_full[n_i:, :n_i]
        w_rr = w_full[n_i:, n_i:]

        logger.info(f"Computing Schur complement for {n_i} interpretations and {n_r} results")

        # Regularize W_II for numerical stability
        w_ii_reg = w_ii + epsilon * np.eye(n_i)

        cond_number = np.linalg.cond(w_ii_reg)
        logger.info(f"W_II condition number: {cond_number:.2e}")

        if cond_number > 1e10:
            logger.warning(f"High condition number ({cond_number:.2e}), using pseudo-inverse")
            w_ii_inv = np.linalg.pinv(w_ii_reg)
        else:
            try:
                w_ii_inv = np.linalg.inv(w_ii_reg)
            except np.linalg.LinAlgError as e:
                logger.warning(f"Matrix inversion failed ({e}), using pseudo-inverse")
                w_ii_inv = np.linalg.pinv(w_ii_reg)

        # S = W_RR - W_RI @ inv(W_II) @ W_IR
        projection_term = w_ri @ w_ii_inv @ w_ir
        schur = w_rr - projection_term

        # Log projection magnitude
        projection_frobenius = np.linalg.norm(projection_term, "fro")
        wrr_frobenius = np.linalg.norm(w_rr, "fro")
        projection_ratio = projection_frobenius / wrr_frobenius if wrr_frobenius > 0 else 0
        logger.info(f"Interpretation-explained variance: {projection_ratio:.2%} of W_RR")

        # Ensure PSD by clipping negative eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(schur)
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue < -1e-10:
            num_negative = np.sum(eigenvalues < 0)
            logger.warning(
                f"Schur complement has {num_negative} negative eigenvalues (min={min_eigenvalue:.6f}). "
                f"Projecting to PSD."
            )
            eigenvalues_corrected = np.maximum(eigenvalues, 0)
            schur = eigenvectors @ np.diag(eigenvalues_corrected) @ eigenvectors.T

        return schur

    def compute_naive_h_r_given_i(
        self,
        w_full: np.ndarray,
        n_i: int,
        n_r: int,
        h_i: float,
    ) -> float:
        """
        Compute H(R|I) via naive subtraction: H(I,R)_joint - H(I).

        WARNING: Can yield NEGATIVE values. Provided for comparison only.
        """
        n_total = n_i + n_r
        G_full = nx.Graph()
        for i in range(n_total):
            for j in range(i, n_total):
                if w_full[i, j] > 1e-10:
                    G_full.add_edge(i, j, weight=w_full[i, j])

        K_full = self.kle.heat_kernel(G_full)
        K_full_norm = K_full / np.trace(K_full) if np.trace(K_full) > 0 else K_full
        h_ir_joint = self.kle.vn_entropy(K_full_norm)

        h_r_given_i_naive = h_ir_joint - h_i

        logger.info(
            f"  Naive subtraction: H(I,R)={h_ir_joint:.6f}, H(I)={h_i:.6f}, "
            f"H(R|I)_naive={h_r_given_i_naive:.6f}"
            f"{' (NEGATIVE)' if h_r_given_i_naive < 0 else ''}"
        )

        return h_r_given_i_naive

    def compute_entropies_schur(
        self,
        w_full: np.ndarray,
        n_i: int,
        n_r: int,
        epsilon: float = 1e-1,
    ) -> Tuple[float, float, float, float, float]:
        """
        Compute all CLUES entropy metrics using Schur complement.

        Returns:
            (h_i, h_r, h_qir, h_r_given_i, h_r_given_i_naive):
                - h_i: H(I), interpretation ambiguity
                - h_r: H(R), marginal result entropy
                - h_qir: H(I,R), total system entropy (via chain rule)
                - h_r_given_i: H(R|I), conditional entropy (Schur, >= 0)
                - h_r_given_i_naive: H(R|I) via naive subtraction (may be negative)
        """
        logger.info("=" * 60)
        logger.info("CLUES: Schur Complement Entropy Decomposition")
        logger.info("=" * 60)

        w_ii = w_full[:n_i, :n_i]
        w_rr = w_full[n_i:, n_i:]

        # Step 1: H(I) from W_II
        G_i = nx.Graph()
        for i in range(n_i):
            for j in range(i, n_i):
                if w_ii[i, j] > 1e-10:
                    G_i.add_edge(i, j, weight=w_ii[i, j])

        K_i = self.kle.heat_kernel(G_i)
        K_i_norm = K_i / np.trace(K_i) if np.trace(K_i) > 0 else K_i
        h_i = self.kle.vn_entropy(K_i_norm)
        logger.info(f"  H(I) = {h_i:.6f} (interpretation ambiguity)")

        # Step 2: H(R) from W_RR
        G_r = nx.Graph()
        for i in range(n_r):
            for j in range(i, n_r):
                if w_rr[i, j] > 1e-10:
                    G_r.add_edge(i, j, weight=w_rr[i, j])

        K_r = self.kle.heat_kernel(G_r)
        K_r_norm = K_r / np.trace(K_r) if np.trace(K_r) > 0 else K_r
        h_r = self.kle.vn_entropy(K_r_norm)
        logger.info(f"  H(R) = {h_r:.6f} (marginal result entropy)")

        # Step 3: Schur complement
        schur = self.compute_schur_complement(w_full, n_i, n_r, epsilon)

        # Step 4: H(R|I) from Schur complement
        G_schur = nx.Graph()
        for i in range(n_r):
            for j in range(i, n_r):
                if schur[i, j] > 1e-10:
                    G_schur.add_edge(i, j, weight=schur[i, j])

        if len(G_schur.edges()) == 0:
            logger.warning("Schur complement graph has no edges - H(R|I) = 0")
            h_r_given_i = 0.0
        else:
            K_schur = self.kle.heat_kernel(G_schur)
            K_schur_norm = K_schur / np.trace(K_schur) if np.trace(K_schur) > 0 else K_schur
            h_r_given_i = self.kle.vn_entropy(K_schur_norm)

        logger.info(f"  H(R|I) = {h_r_given_i:.6f} (instability, via Schur complement)")

        # Step 5: Joint entropy via chain rule
        h_qir = h_i + h_r_given_i
        logger.info(f"  H(I,R) = {h_qir:.6f} (total, via chain rule)")

        # Step 6: Naive subtraction for comparison
        h_r_given_i_naive = self.compute_naive_h_r_given_i(w_full, n_i, n_r, h_i)

        logger.info("=" * 60)
        logger.info("DECOMPOSITION SUMMARY:")
        if h_qir > 1e-10:
            logger.info(f"  H(I)   = {h_i:.6f} ({h_i / h_qir * 100:.1f}% of total)")
            logger.info(f"  H(R|I) = {h_r_given_i:.6f} ({h_r_given_i / h_qir * 100:.1f}% of total)")
        else:
            logger.info(f"  H(I)   = {h_i:.6f}")
            logger.info(f"  H(R|I) = {h_r_given_i:.6f}")
        logger.info(f"  H(R)   = {h_r:.6f} (marginal)")
        logger.info("=" * 60)

        return h_i, h_r, h_qir, h_r_given_i, h_r_given_i_naive
