from __future__ import annotations

from typing import Dict, List, Set
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2

from .local_kl_utils import get_local_kl_value


class ClusterDecomposer:
    """Annotate a hierarchy with significance tests and carve it into clusters.

    The decomposer walks a :class:`~tree.poset_tree.PosetTree` top-down and decides
    whether to split or merge at each internal node based on two statistical gates:

    #. **Local divergence gate** – both children must significantly diverge from the
       parent according to the local KL (child‖parent) chi-square test. If either
       child fails, the subtree is merged into a single cluster.
    #. **Sibling independence gate** – provided the local gate passes, children must
       be independent given the parent (conditional mutual information BH test).
       If independence is rejected, the children are merged; otherwise the walk
       recurses into each child.

    Nodes that pass both gates become cluster boundaries. Leaves under the same
    boundary node are assigned the same cluster identifier. The resulting report
    captures the cluster root node, member leaves, and cluster size.
    """

    def __init__(
        self,
        tree: nx.DiGraph,
        results_df: pd.DataFrame | None = None,
        *,
        n_features: int | None = None,  # if None, inferred from distributions
        alpha_local: float = 0.05,  # χ² test level for KL(child‖parent)
        significance_column: str = "Are_Features_Dependent",
        parent_gate: str = "off",  # "off" or "strict"
    ):
        """Configure decomposition thresholds and pre-compute reusable metadata.

        Parameters
        ----------
        tree
            Directed hierarchy (typically a :class:`~tree.poset_tree.PosetTree`).
        results_df
            DataFrame of statistical annotations (e.g., columns produced by
            ``hierarchy_analysis.statistics`` helpers). May be ``None`` if the caller
            plans to rely on on-the-fly calculations.
        n_features
            Total number of feature dimensions; inferred from node distributions when
            ``None``.
        alpha_local
            Significance level used when the local KL gate falls back to raw
            chi-square tests.
        significance_column
            Column name in ``results_df`` representing node-level significance for the
            optional parent gate.
        parent_gate
            ``"off"`` to ignore parent-level significance, ``"strict"`` to require
            parents to be marked significant before splitting.
        """
        self.tree = tree
        self.results_df = results_df if results_df is not None else pd.DataFrame()
        self.significance_column = significance_column
        self.alpha_local = float(alpha_local)
        self.parent_gate = parent_gate

        # ----- root -----
        roots = [n for n, d in self.tree.in_degree() if d == 0]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one root; found {len(roots)}: {roots}")
        self._root = roots[0]

        # ----- pre-cache distributions & feature count -----
        nf = None
        self._dist: Dict[str, np.ndarray] = {}
        self._is_leaf: Dict[str, bool] = {}
        self._label: Dict[str, str] = {}
        for n in self.tree.nodes:
            node = self.tree.nodes[n]
            if "is_leaf" not in node:
                raise ValueError(f"Node '{n}' missing 'is_leaf'.")
            if "distribution" not in node:
                raise ValueError(f"Node '{n}' missing 'distribution'.")
            arr = np.asarray(node["distribution"], dtype=float).ravel()
            if arr.ndim != 1:
                raise ValueError(
                    f"Node '{n}' distribution must be 1D, got shape {arr.shape}."
                )
            self._dist[n] = arr
            self._is_leaf[n] = bool(node["is_leaf"])
            self._label[n] = node.get("label", n)
            nf = len(arr) if nf is None else nf
        self.n_features = int(n_features if n_features is not None else nf)

        # ----- descendant sets & leaf counts (poset view) -----
        self._desc_sets: Dict[str, frozenset] = self._compute_descendant_sets()
        self._leaf_count_cache: Dict[str, int] = {}
        for n in self.tree.nodes:
            if "leaf_count" in self.tree.nodes[n]:
                self._leaf_count_cache[n] = int(self.tree.nodes[n]["leaf_count"])
            else:
                self._leaf_count_cache[n] = len(self._desc_sets.get(n, ()))

        # ----- results_df → fast dict lookups (no .loc in hot paths) -----
        self._parent_sig: Dict[str, bool] = {}
        self._local_sig: Dict[str, bool] = {}
        self._sibling_indep: Dict[str, bool] = {}
        if not self.results_df.empty:
            if self.significance_column in self.results_df.columns:
                self._parent_sig = (
                    self.results_df[self.significance_column]
                    .fillna(False)
                    .astype(bool)
                    .to_dict()
                )
            if "Local_BH_Significant" in self.results_df.columns:
                self._local_sig = (
                    self.results_df["Local_BH_Significant"]
                    .fillna(False)
                    .astype(bool)
                    .to_dict()
                )
            elif "Local_Are_Features_Dependent" in self.results_df.columns:
                self._local_sig = (
                    self.results_df["Local_Are_Features_Dependent"]
                    .fillna(False)
                    .astype(bool)
                    .to_dict()
                )
            if "Sibling_BH_Independent" in self.results_df.columns:
                self._sibling_indep = (
                    self.results_df["Sibling_BH_Independent"]
                    .fillna(False)
                    .astype(bool)
                    .to_dict()
                )

        # ----- caches for local tests -----
        self._local_p_cache: Dict[tuple[str, str], float] = {}
        self._local_diverge_cache: Dict[tuple[str, str], bool] = {}

        # Precompute children list (avoids rebuilding generator repeatedly)
        self._children: Dict[str, List[str]] = {
            n: list(self.tree.successors(n)) for n in self.tree.nodes
        }

    # ---------- poset helpers ----------

    def _compute_descendant_sets(self) -> dict[str, frozenset]:
        desc: Dict[str, frozenset] = {}
        # process children before parents
        for node in nx.topological_sort(self.tree.reverse()):
            if self._is_leaf[node]:
                desc[node] = frozenset([self._label[node]])
            else:
                child_sets = [desc[c] for c in self.tree.successors(node)]
                desc[node] = (
                    frozenset().union(*child_sets) if child_sets else frozenset()
                )
        return desc

    # ---------- utilities ----------

    def _get_all_leaves(self, node_id: str) -> set[str]:
        """Return the set of leaf labels beneath ``node_id`` using cached posets."""
        fs = self._desc_sets.get(node_id, frozenset())
        if fs:
            return set(fs)
        # fallback for leaf nodes with no entry
        if self._is_leaf[node_id]:
            return {self._label[node_id]}
        return set()

    def is_significant(self, node_id: str) -> bool:
        """Convenience accessor for parent-level significance gate."""
        return bool(self._parent_sig.get(node_id, False))

    # ---------- local KL (child vs parent) ----------

    def _leaf_count(self, node_id: str) -> int:
        """Retrieve cached leaf-count for ``node_id`` (populated during init)."""
        return self._leaf_count_cache[node_id]

    def _child_parent_pvalue(self, child: str, parent: str) -> float:
        """Compute or reuse the chi-square p-value for ``child`` vs ``parent``."""
        key = (child, parent)
        if key in self._local_p_cache:
            return self._local_p_cache[key]

        # Prefer precomputed local KL if present on node attributes
        kl_local = get_local_kl_value(
            self.tree,
            child,
            parent,
            child_dist=self._dist[child],
            parent_dist=self._dist[parent],
        )
        n_leaves = self._leaf_count(child)
        chi2_stat = 2.0 * n_leaves * float(kl_local)
        p = float(chi2.sf(chi2_stat, df=self.n_features))
        self._local_p_cache[key] = p
        return p

    def _child_diverges_from_parent(self, child: str, parent: str) -> bool:
        """Determine whether the local KL test flags ``child`` as divergent."""
        key = (child, parent)
        if key in self._local_diverge_cache:
            return self._local_diverge_cache[key]

        # 1) Use annotated local decision when available
        if child in self._local_sig:
            res = bool(self._local_sig[child])
            self._local_diverge_cache[key] = res
            return res

        # 2) Compute χ² p-value on the fly
        p = self._child_parent_pvalue(child, parent)
        res = bool(p < self.alpha_local)
        self._local_diverge_cache[key] = res
        return res

    # ---------- core decomposition (iterative, no recursion) ----------

    def _should_split(self, parent: str) -> bool:
        """Evaluate both gates and return ``True`` when the parent should split."""
        # Optional: gate by parent significance
        if (
            self.parent_gate == "strict"
            and parent != self._root
            and self.is_significant(parent)
        ):
            return False

        children = self._children[parent]
        if len(children) != 2:
            # non-binary → treat as terminal cluster
            return False
        c1, c2 = children

        # 1) local KL gate: both children must significantly diverge from parent
        if not (
            self._child_diverges_from_parent(c1, parent)
            and self._child_diverges_from_parent(c2, parent)
        ):
            return False

        # 2) sibling independence decision (annotated)
        indep = self._sibling_indep.get(parent, None)
        return bool(indep is True)

    def _collect_cluster_leaves(self, node_id: str) -> set[str]:
        """Gather all leaves under ``node_id`` to form a cluster record."""
        return self._get_all_leaves(node_id)

    def decompose_tree(self) -> dict[str, object]:
        """Return cluster assignments by iteratively traversing the hierarchy."""
        stack = [self._root]
        final_leaf_sets: List[set[str]] = []

        while stack:
            node = stack.pop()
            children = self._children[node]

            # leaf or unary → cluster here
            if len(children) < 2:
                final_leaf_sets.append(self._collect_cluster_leaves(node))
                continue

            # non-binary → merge at parent
            if len(children) != 2:
                final_leaf_sets.append(self._collect_cluster_leaves(node))
                continue

            if self._should_split(node):
                # split → push children
                c1, c2 = children
                stack.append(c2)
                stack.append(c1)
            else:
                # merge here
                final_leaf_sets.append(self._collect_cluster_leaves(node))

        cluster_assignments: dict[int, dict[str, object]] = {}
        for i, leaf_set in enumerate(final_leaf_sets):
            if not leaf_set:
                continue
            cluster_assignments[i] = {
                "root_node": self._find_cluster_root(leaf_set),
                "leaves": sorted(leaf_set),
                "size": len(leaf_set),
            }
        return {
            "cluster_assignments": cluster_assignments,
            "num_clusters": len(cluster_assignments),
            "independence_analysis": {
                "alpha_local": self.alpha_local,
                "decision_mode": "cmi_only",
            },
        }

    # ---------- LCA ----------

    def _find_cluster_root(self, leaf_labels: Set[str]) -> str:
        """Identify the lowest common ancestor for a collection of leaf labels."""
        # Map labels to leaf node ids (small set → simple scan)
        leaf_nodes: List[str] = []
        for n in self.tree.nodes:
            if self._is_leaf[n] and self._label[n] in leaf_labels:
                leaf_nodes.append(n)
        if not leaf_nodes:
            return self._root
        if len(leaf_nodes) == 1:
            return leaf_nodes[0]

        common = set(nx.ancestors(self.tree, leaf_nodes[0]))
        common.add(leaf_nodes[0])
        for lf in leaf_nodes[1:]:
            anc = set(nx.ancestors(self.tree, lf))
            anc.add(lf)
            common &= anc
            if not common:
                return self._root
        # choose the “lowest”: one that has no descendant also in `common`
        for anc in common:
            if not (set(nx.descendants(self.tree, anc)) & common):
                return anc
        return self._root

    # ---------- reporting / compat ----------

    def generate_report(self, decomposition_results: dict[str, object]) -> pd.DataFrame:
        """
        Convenience wrapper to build a per-sample cluster assignment DataFrame.

        Mirrors hierarchy_analysis.decomposition_utils.generate_decomposition_report
        so existing code that called `decomposer.generate_report(...)` continues to work.
        """
        from .decomposition_utils import generate_decomposition_report

        return generate_decomposition_report(decomposition_results)

    @property
    def independence_threshold(self) -> float:
        """
        Backwards-compat property used by older scripts. The current algorithm
        uses CMI-based statistical decisions rather than a fixed threshold; we
        return NaN to indicate 'not applicable'.
        """
        try:
            return float("nan")
        except Exception:  # pragma: no cover
            return None  # type: ignore[return-value]
