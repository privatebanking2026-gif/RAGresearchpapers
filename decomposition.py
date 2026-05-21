"""
Uncertainty Decomposition orchestrator for CLUES framework.

Coordinates the full pipeline: similarity computation -> block matrix construction
-> Schur complement entropy decomposition -> regime classification.

Paper Reference:
    Section 3: Methodology - A Decomposition Framework for Semantic Uncertainty.
"""

from typing import List, Optional, Tuple
import asyncio
import logging
import networkx as nx
import numpy as np
import timeit

from clues.config import UncertaintyResult, UncertaintyThresholds
from clues.entropy import KernelLaplacianEntropy
from clues.schur import SchurComplementEntropy
from clues.llm import GeminiClient

logger = logging.getLogger(__name__)


class UncertaintyDecomposition:
    """
    Full CLUES uncertainty decomposition pipeline.

    Computes similarity matrices, constructs the bipartite semantic graph,
    and decomposes uncertainty into ambiguity (H_I) and instability (H_{R|I}).

    Args:
        t: Heat kernel diffusion parameter (default 10.0).
        norm_laplacian: Whether to use normalized Laplacian (default True).
        gamma: Input kernel sharpening parameter for W_II (default 1.0).
    """

    def __init__(
        self,
        t: float = 10.0,
        norm_laplacian: bool = True,
        gamma: float = 1.0,
    ):
        self.t = t
        self.norm_laplacian = norm_laplacian
        self.gamma = gamma
        self.kle = KernelLaplacianEntropy(t=t, norm_lapl=norm_laplacian, gamma=gamma)
        self.schur = SchurComplementEntropy(self.kle)

        if gamma != 1.0:
            logger.info(f"Input Kernel Sharpening ENABLED: gamma={gamma}")

    def compute_entropies(
        self, w_full: np.ndarray, n_i: int, n_r: int
    ) -> Tuple[float, float, float, float, float]:
        """Compute all entropy metrics via Schur complement."""
        return self.schur.compute_entropies_schur(w_full, n_i, n_r)

    async def compute_decomposed_uncertainty(
        self,
        interpretations: List[str],
        results: List[str],
        mapping: List[int],
        client: Optional[GeminiClient] = None,
        max_concurrency: int = 25,
        interpretation_similarity_prompt: Optional[str] = None,
        thresholds: Optional[UncertaintyThresholds] = None,
    ) -> dict:
        """
        Compute the full CLUES uncertainty decomposition.

        Args:
            interpretations: List of interpretation texts.
            results: List of result texts.
            mapping: List where mapping[i] = j means result i came from interpretation j.
            client: GeminiClient for similarity computation. If None, creates one.
            max_concurrency: Maximum concurrent API calls.
            interpretation_similarity_prompt: Optional custom prompt for W_II computation.

        Returns:
            Dict with matrices, entropies, contributions, uncertainty_result, graphs, parameters.
        """
        if client is None:
            client = GeminiClient()

        start_time = timeit.default_timer()

        logger.info("Computing interpretation and result similarity matrices in parallel...")

        # Compute W_II and W_RR in parallel
        task_ii = self.compute_similarity_matrix(
            interpretations,
            client,
            max_concurrency=max_concurrency,
            custom_entailment_prompt=interpretation_similarity_prompt,
        )
        task_rr = self.compute_similarity_matrix(
            results,
            client,
            max_concurrency=max_concurrency,
        )

        (w_ii, g_ii), (w_rr, g_rr) = await asyncio.gather(task_ii, task_rr)

        # Apply kernel sharpening to W_II if gamma != 1.0
        w_ii_original = w_ii.copy()
        if self.gamma != 1.0:
            logger.info(f"Applying Input Kernel Sharpening to W_II: gamma={self.gamma}")
            w_ii = KernelLaplacianEntropy.sharpen_similarity_matrix(w_ii, self.gamma)

        # Construct block matrix
        w_ir = self.create_mapping_matrix(len(interpretations), len(results), mapping)
        w_ri = w_ir.T
        w_full = self.construct_full_matrix(w_ii, w_rr, w_ir, w_ri)

        # Compute entropy decomposition
        h_i, h_r, h_qir, h_r_given_i, h_r_given_i_naive = self.compute_entropies(
            w_full, len(interpretations), len(results)
        )

        h_r_given_i_raw = h_r_given_i
        if h_r_given_i < 0:
            logger.warning(f"Negative H(R|I)={h_r_given_i:.6f}, enforcing non-negativity.")
            h_r_given_i = 0.0

        # Compute contributions
        if h_qir > 0:
            if h_r_given_i_raw < 0:
                ambiguity_contribution = 1.0
                result_contribution = 0.0
            else:
                ambiguity_contribution = h_i / h_qir
                result_contribution = h_r_given_i / h_qir
        else:
            ambiguity_contribution = float("nan")
            result_contribution = float("nan")

        # Create result with regime classification
        entropies_dict = {
            "H_I": h_i,
            "H_R": h_r,
            "H_QIR": h_qir,
            "H(R|I)_raw": h_r_given_i_raw,
            "H(R|I)": h_r_given_i,
            "H(R|I)_naive": h_r_given_i_naive,
        }
        uncertainty_result = UncertaintyResult.from_metrics(entropies_dict, thresholds=thresholds)

        elapsed_time = timeit.default_timer() - start_time
        logger.info(f"Total computation time: {elapsed_time:.2f} seconds")
        logger.info(
            f"Regime: {uncertainty_result.regime.value} -> {uncertainty_result.recommended_action}"
        )

        return {
            "matrices": {
                "W_II": w_ii,
                "W_II_original": w_ii_original,
                "W_RR": w_rr,
                "W_IR": w_ir,
                "W_RI": w_ri,
                "W_full": w_full,
            },
            "entropies": entropies_dict,
            "contributions": {
                "ambiguity": ambiguity_contribution,
                "result": result_contribution,
            },
            "uncertainty_result": uncertainty_result,
            "graphs": {"G_II": g_ii, "G_RR": g_rr},
            "parameters": {"gamma": self.gamma, "t": self.t},
        }

    async def compute_decomposed_uncertainty_from_matrices(
        self,
        w_ii: np.ndarray,
        w_rr: np.ndarray,
        mapping: List[int],
        thresholds: Optional[UncertaintyThresholds] = None,
    ) -> dict:
        """
        Compute CLUES decomposition from pre-computed similarity matrices (offline mode).

        Args:
            w_ii: Pre-computed interpretation similarity matrix (N x N).
            w_rr: Pre-computed result similarity matrix (NM x NM).
            mapping: List where mapping[i] = j means result i came from interpretation j.

        Returns:
            Same dict structure as compute_decomposed_uncertainty.
        """
        n_i = w_ii.shape[0]
        n_r = w_rr.shape[0]

        # Apply kernel sharpening if needed
        w_ii_original = w_ii.copy()
        if self.gamma != 1.0:
            w_ii = KernelLaplacianEntropy.sharpen_similarity_matrix(w_ii, self.gamma)

        # Build placeholder graphs from matrices
        g_ii = self._matrix_to_graph(w_ii)
        g_rr = self._matrix_to_graph(w_rr)

        # Construct block matrix
        w_ir = self.create_mapping_matrix(n_i, n_r, mapping)
        w_ri = w_ir.T
        w_full = self.construct_full_matrix(w_ii, w_rr, w_ir, w_ri)

        # Compute entropy decomposition
        h_i, h_r, h_qir, h_r_given_i, h_r_given_i_naive = self.compute_entropies(
            w_full, n_i, n_r
        )

        h_r_given_i_raw = h_r_given_i
        if h_r_given_i < 0:
            h_r_given_i = 0.0

        if h_qir > 0:
            if h_r_given_i_raw < 0:
                ambiguity_contribution = 1.0
                result_contribution = 0.0
            else:
                ambiguity_contribution = h_i / h_qir
                result_contribution = h_r_given_i / h_qir
        else:
            ambiguity_contribution = float("nan")
            result_contribution = float("nan")

        entropies_dict = {
            "H_I": h_i,
            "H_R": h_r,
            "H_QIR": h_qir,
            "H(R|I)_raw": h_r_given_i_raw,
            "H(R|I)": h_r_given_i,
            "H(R|I)_naive": h_r_given_i_naive,
        }
        uncertainty_result = UncertaintyResult.from_metrics(entropies_dict, thresholds=thresholds)

        return {
            "matrices": {
                "W_II": w_ii,
                "W_II_original": w_ii_original,
                "W_RR": w_rr,
                "W_IR": w_ir,
                "W_RI": w_ri,
                "W_full": w_full,
            },
            "entropies": entropies_dict,
            "contributions": {
                "ambiguity": ambiguity_contribution,
                "result": result_contribution,
            },
            "uncertainty_result": uncertainty_result,
            "graphs": {"G_II": g_ii, "G_RR": g_rr},
            "parameters": {"gamma": self.gamma, "t": self.t},
        }

    async def compute_similarity_matrix(
        self,
        texts: List[str],
        client: GeminiClient,
        max_concurrency: int = 25,
        custom_entailment_prompt: Optional[str] = None,
    ) -> Tuple[np.ndarray, nx.Graph]:
        """
        Compute similarity matrix via prompt-based entailment.

        Args:
            texts: List of text strings to compare.
            client: GeminiClient for LLM calls.
            max_concurrency: Maximum concurrent API calls.
            custom_entailment_prompt: Optional custom prompt.

        Returns:
            (similarity_matrix, networkx_graph)
        """
        graph = await self.kle.get_prompt_similarity_graph(
            contents=texts,
            client=client,
            max_concurrency=max_concurrency,
            custom_prompt=custom_entailment_prompt,
        )

        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if graph.has_edge(i, j):
                    similarity_matrix[i, j] = graph[i][j]["weight"]

        return similarity_matrix, graph

    @staticmethod
    def create_mapping_matrix(
        n_interp: int, n_results: int, mapping: List[int]
    ) -> np.ndarray:
        """
        Create binary mapping matrix W_IR.

        Args:
            n_interp: Number of interpretations.
            n_results: Number of results.
            mapping: mapping[result_idx] = interpretation_idx.

        Returns:
            W_IR: (n_interp x n_results) binary matrix.
        """
        w_ir = np.zeros((n_interp, n_results))
        for result_idx, interp_idx in enumerate(mapping):
            w_ir[interp_idx, result_idx] = 1.0
        return w_ir

    def construct_full_matrix(
        self,
        w_ii: np.ndarray,
        w_rr: np.ndarray,
        w_ir: np.ndarray,
        w_ri: np.ndarray,
    ) -> np.ndarray:
        """
        Construct full block similarity matrix W (Eq. 1 from paper).

        W = [W_II  W_IR]
            [W_RI  W_RR]
        """
        n_interp = w_ii.shape[0]
        n_results = w_rr.shape[0]

        w_full = np.zeros((n_interp + n_results, n_interp + n_results))
        w_full[:n_interp, :n_interp] = w_ii
        w_full[n_interp:, n_interp:] = w_rr
        w_full[:n_interp, n_interp:] = w_ir
        w_full[n_interp:, :n_interp] = w_ri

        eigenvalues = np.linalg.eigvalsh(w_full)
        min_eig = np.min(eigenvalues)
        logger.info(f"W_FULL eigenvalue range: [{min_eig:.6f}, {np.max(eigenvalues):.6f}]")

        return w_full

    @staticmethod
    def _matrix_to_graph(matrix: np.ndarray) -> nx.Graph:
        """Convert a similarity matrix to a NetworkX graph."""
        G = nx.Graph()
        n = matrix.shape[0]
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i, n):
                if matrix[i, j] > 0:
                    G.add_edge(i, j, weight=matrix[i, j])
        return G
