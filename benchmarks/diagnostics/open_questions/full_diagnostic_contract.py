"""Full diagnostic contracts for the KL-TE open mathematical questions.

This module converts the open-question ledger into executable diagnostic
requirements. A row marked ``fully_specified`` is not a solved method claim.
It means the question has a concrete diagnostic target, required inputs,
required outputs, acceptance criteria, and a stated blocker before promotion.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from benchmarks.shared.util.time import format_timestamp_utc

STUDY_ROLE = "open_question_full_diagnostic_contract_not_method_claim"
SCHEMA_VERSION = "open_question_full_diagnostic_contract/v1"
CONTRACT_STATUS = "fully_specified"
REQUIRED_FIELDS = (
    "question_id",
    "current_evidence_status",
    "diagnostic_contract_status",
    "diagnostic_family",
    "diagnostic_scale",
    "existing_entrypoint",
    "required_inputs",
    "required_outputs",
    "acceptance_criteria",
    "blocks_method_claim_until",
    "study_role",
)


@dataclass(frozen=True)
class DiagnosticContract:
    """Diagnostic contract for one open mathematical question."""

    question_id: str
    current_evidence_status: str
    diagnostic_family: str
    diagnostic_scale: str
    existing_entrypoint: str
    required_inputs: str
    required_outputs: str
    acceptance_criteria: str
    blocks_method_claim_until: str
    diagnostic_contract_status: str = CONTRACT_STATUS
    study_role: str = STUDY_ROLE


CONTRACTS: tuple[DiagnosticContract, ...] = (
    DiagnosticContract(
        "Q1",
        "partially_diagnosed",
        "selected_tail_law",
        "cloud_panel",
        "benchmarks/diagnostics/calibration/selected_tail_law_q5_validation.py",
        "Admissible selected-null contexts with tree reconstruction, edge action, focal sibling context, feature family, parent sizes, projection dimension, spectral summaries, and barycentric balance.",
        "Context-level held-out tail calibration, admissibility status, residual-tail error, and failure reasons.",
        "Pass only when every promoted context has >=499 matching simulations, <=5% c-hat simulation SE, and residual-tail exceedance within alpha plus two binomial SEs on held-out folds.",
        "No external calibration law may be used outside contexts passing the selected-tail admissibility and held-out absolute-tail checks.",
    ),
    DiagnosticContract(
        "Q2",
        "partially_diagnosed",
        "ancestor_path_selection",
        "cloud_panel",
        "benchmarks/diagnostics/calibration/selected_hierarchy_geometry_covariates.py",
        "Non-root selected records with ancestor Tree-BH path actions, depths, edge p-values, selected hierarchy ratios, and matched simulation ids.",
        "Path-conditioned c-hat, selected-ratio tail quantiles, support counts, and comparison against path-relaxed contexts.",
        "Pass when adding ancestor-path variables changes c-hat and q95 ratio by less than 10% in all admissible non-root contexts; otherwise path conditioning is mandatory.",
        "Non-root external laws cannot borrow root or path-relaxed calibration until path-stability passes.",
    ),
    DiagnosticContract(
        "Q3",
        "partially_answered",
        "full_selection_baseline",
        "local_and_cloud",
        "benchmarks/validation/selected_edge_type1_geometry.py",
        "Fixed-tree, ancestor-path, focal-blocker, and full-reconstruction runs with common seeds and null/planted labels.",
        "Layer attribution table with false-split rate, selected-ratio distortion, edge rejection, and support failures by baseline.",
        "Pass when the full-reconstruction baseline is present and every shortcut baseline is explicitly marked conservative, anti-conservative, or attribution-only.",
        "Selection diagnostics must report full reconstruction before any shortcut diagnostic can support a method claim.",
    ),
    DiagnosticContract(
        "Q4",
        "partially_diagnosed",
        "gaussian_selected_tail",
        "cloud_panel",
        "benchmarks/diagnostics/calibration/selected_tail_admissibility_domain.py",
        "Gaussian null and planted Gaussian contexts across parent size, projection dimension, covariance condition, and barycentric balance.",
        "Gaussian admissibility table, c-hat precision, selected-ratio q95/q99, and blocker explanation by support status.",
        "Pass when Gaussian blocker contexts become admissible and held-out exceedance is calibrated at sibling alpha 0.01.",
        "Gaussian blocker claims remain diagnostic until the selected-tail law explains support-failure contexts without overborrowing.",
    ),
    DiagnosticContract(
        "Q5",
        "partially_answered",
        "selected_tail_law",
        "cloud_panel",
        "benchmarks/diagnostics/calibration/selected_tail_law_q5_validation.py",
        "Broad selected-geometry records including edge action, p/n, parent size, projection dimension, feature family, spectral geometry, and barycentric variables.",
        "Six-model Q5 validation table plus source-family, non-root, categorical, continuous, and phylogenetic transfer summaries.",
        "Pass when the compact barycentric edge/spectral model or successor has calibrated held-out absolute tails in every promoted family.",
        "Q5 may guide diagnostics but not production external calibration until broad transfer passes.",
    ),
    DiagnosticContract(
        "Q6",
        "implemented_internal_open_external",
        "calibration_decision_contract",
        "local_and_cloud",
        "kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/types/inflation_model.py",
        "Internal calibration decisions plus external selected-tail context rows carrying admissibility and held-out tail metrics.",
        "Unified decision table with internal_admissible, undefined support failures, and external_admissible states.",
        "Pass when external_admissible is only returned for contexts satisfying Q1/Q7 tail and support contracts.",
        "The external decision branch must remain disabled until promotion criteria are implemented and tested.",
    ),
    DiagnosticContract(
        "Q7",
        "partially_answered",
        "external_support_contract",
        "cloud_panel",
        "benchmarks/diagnostics/calibration/selected_hierarchy_external_calibration_contract.py",
        "Independent matching simulations, matched selected records, c-hat SE, empirical tail resolution, and held-out folds per context.",
        "Admissibility table with support, precision, tail resolution, and held-out exceedance pass/fail columns.",
        "Pass when all promoted contexts satisfy >=499 simulations, >=499 records, <=5% relative c-hat SE, and valid held-out tail precision.",
        "No selected-tail estimate is production-admissible without this row-level contract.",
    ),
    DiagnosticContract(
        "Q8",
        "partially_answered",
        "context_schema",
        "cloud_panel",
        "benchmarks/diagnostics/calibration/selected_tail_topology_refinement.py",
        "Predeclared context variables and candidate stratifiers: case, family, n, p, projection, parent-size bin, edge action, spectral summaries, topology, and barycentric leverage.",
        "Exact-match versus modeled-tail comparison with support fragmentation and tail calibration metrics.",
        "Pass when the selected context schema reaches support without hidden data-adaptive bins and preserves held-out absolute-tail calibration.",
        "Context variables remain descriptive until the schema is predeclared and support-admissible.",
    ),
    DiagnosticContract(
        "Q9",
        "partially_diagnosed",
        "internal_support_threshold",
        "local_and_cloud",
        "benchmarks/diagnostics/calibration/internal_support_threshold_validation.py",
        "Internal calibration decisions with positive-weight records, effective sample size, max weight share, leave-one stability, strict/stopped support, and barycentric balance.",
        "Threshold grid reporting false-split rate, undefined rate, c-hat stability, and planted-signal retention.",
        "Pass when one threshold tuple controls null false splits while preserving planted power across support regimes.",
        "Sparse-context enforcement must stay opt-in until the threshold grid passes.",
    ),
    DiagnosticContract(
        "Q10",
        "diagnostic_only",
        "weight_rule_validation",
        "cloud_panel",
        "benchmarks/diagnostics/calibration/sibling_null_weight_rule_validation.py",
        "Mixed null/signal sibling records with true support labels, edge p-values, selected ratios, and calibration outcomes.",
        "Weight-rule grid with labeled-support calibration, effective support, false-split rate, and stability.",
        "Pass when a rule improves support stability without increasing labeled-null false splits or selected-nonnull leakage.",
        "The current product-BH rule remains default until labeled support validation selects a replacement.",
    ),
    DiagnosticContract(
        "Q11",
        "partially_diagnosed",
        "bandwidth_stability",
        "cloud_panel",
        "kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/empirical_null_inflation_estimation.py",
        "Calibration decisions over exact and relaxed axes for parent size, family, projection, spectral geometry, and barycentric balance/leverage.",
        "Bandwidth grid with c-hat variance, undefined rate, false-split rate, and support concentration.",
        "Pass when a bandwidth rule has stable c-hat under leave-one-record and matched-context perturbations in admissible contexts.",
        "Bandwidth remains implementation default rather than validated method constant.",
    ),
    DiagnosticContract(
        "Q12",
        "partially_diagnosed",
        "fixed_projection_reference",
        "theory_plus_simulation",
        "benchmarks/validation/selected_pca_projected_wald_calibration.py",
        "Fixed projection matrices, known covariance nulls, independent subtree samples, and feature-family covariance models.",
        "Proof checklist plus simulated p-value uniformity and chi-square QQ summaries under fixed projection.",
        "Pass when assumptions are stated as theorem hypotheses and simulations show calibrated p-values under those hypotheses.",
        "Projected chi-square claims must be scoped to fixed, valid projections until selected-basis theory passes.",
    ),
    DiagnosticContract(
        "Q13",
        "partially_diagnosed",
        "selected_projection_reference",
        "theory_plus_simulation",
        "benchmarks/validation/selected_pca_projected_wald_calibration.py",
        "Selected PCA bases with and without internal rows, selected barycenter maps, and fixed-subspace controls.",
        "Selected-PCA distortion table with p-value calibration, basis-selection variables, and tail inflation.",
        "Pass when selected-basis p-values are calibrated or a correction law is validated against fixed-subspace controls.",
        "Fixed-projection proof is insufficient for selected-PCA method claims.",
    ),
    DiagnosticContract(
        "Q14",
        "partially_diagnosed",
        "mp_selection_law",
        "theory_plus_cloud",
        "benchmarks/diagnostics/spectral/local_mp_identity_law_diagnostic.py",
        "Local spectra, effective independent rows, backend eigenvalue scale, selected-tree variables, and barycentric balance.",
        "Identity/deformed/finite-null MP comparison with projection dimension, calibration, and power endpoints.",
        "Pass when an MP threshold rule controls null projection selection and preserves planted contrast power after selection.",
        "The MP rule remains a diagnostic/default heuristic until selected/deformed MP validation passes.",
    ),
    DiagnosticContract(
        "Q15",
        "partially_diagnosed",
        "minimum_dimension_floor",
        "full_suite",
        "raw/assets/benchmark-results/mp_kmin_q14_q15_smoke_20260604/mp_kmin_contract_smoke.csv",
        "Full benchmark grid over kmin values with zero-dimensional semantics, calibration support, ARI, and false-split labels.",
        "Floor grid reporting support failures, over/under-splits, null false splits, ARI, and dimension distributions.",
        "Pass when one floor has acceptable null control, no unsupported semantic gaps, and competitive planted power.",
        "Do not change kmin default from smoke evidence alone.",
    ),
    DiagnosticContract(
        "Q16",
        "not_resolved",
        "all_selected_nonnull_support",
        "cloud_panel",
        "kl_clustering_analysis/hierarchy_analysis/statistics/sibling_divergence/inflation_correction/inflation_adjusted_sibling_tests.py",
        "Leaf-only selected-nonnull contexts with no strict/stopped support, external admissibility rows, and planted/null labels.",
        "Decision table comparing fail-closed, external selected-tail, and candidate all-selected support policies.",
        "Pass only if a policy controls null false splits without borrowing selected-nonnull rows as empirical null support.",
        "All-selected-nonnull contexts must fail closed until this contract passes.",
    ),
    DiagnosticContract(
        "Q17",
        "diagnostic_only",
        "sibling_projection_dimension",
        "cloud_panel",
        "benchmarks/diagnostics/spectral/mp_projection_dimension_behavior_sweep.py",
        "Selected-geometry rows crossed with null/planted labels, dimension-rule candidates, spectral summaries, and barycentric balance.",
        "Dimension-rule grid with raw selected-tree p-value behavior, null false-split, power, support failure, and rule-difference metrics.",
        "Pass when a rule is calibrated under null and retains planted sibling contrast power across balance strata.",
        "Current behavior grid remains diagnostic because raw selected-tree sibling p-values are not calibrated production decisions.",
    ),
    DiagnosticContract(
        "Q18",
        "partially_diagnosed",
        "categorical_covariance",
        "cloud_panel",
        "benchmarks/validation/feature_covariance_calibration.py",
        "Fixed-tree multinomial one-hot nulls over cardinality, imbalance, parent size, and drop-last covariance conditioning.",
        "Categorical projected-Wald p-value calibration by cardinality and simplex-face proximity.",
        "Pass when fixed-tree categorical p-values are calibrated before same-data selection is introduced.",
        "Categorical selected-tree claims are blocked by fixed-tree multinomial calibration failures.",
    ),
    DiagnosticContract(
        "Q19",
        "partially_diagnosed",
        "categorical_failure_decomposition",
        "full_suite",
        "benchmarks/validation/feature_covariance_calibration.py",
        "High-cardinality categorical panels crossing covariance model, projection rule, sibling FDR layer, edge alpha, and barycentric boundary distance.",
        "Factorial attribution table for covariance, FDR, projection, selection, and simplex-boundary effects.",
        "Pass when the dominant failure source is isolated with stable effect sizes across seeds and cardinalities.",
        "Do not tune categorical defaults until the failure source is separated.",
    ),
    DiagnosticContract(
        "Q20",
        "partially_diagnosed",
        "continuous_covariance",
        "cloud_panel",
        "benchmarks/validation/feature_covariance_calibration.py",
        "Continuous Gaussian nulls over n, p, covariance condition number, dense covariance guardrails, and selected-tree settings.",
        "Dense covariance calibration table with p-value uniformity, runtime feasibility, and support behavior.",
        "Pass when dense empirical covariance is calibrated and computationally admissible for the claimed continuous domain.",
        "p >> n continuous production claims require a validated alternative covariance model.",
    ),
    DiagnosticContract(
        "Q21",
        "idea_only",
        "regularized_continuous_covariance",
        "prototype_then_cloud",
        "not_implemented",
        "Continuous p >> n null/planted panels for shrinkage, low-rank, diagonal, and Wasserstein/Gaussian-barycenter covariance candidates.",
        "Candidate covariance comparison with calibration, power, stability, and runtime/memory metrics.",
        "Pass when one regularized model dominates dense fallback on calibration and power in p >> n contexts.",
        "No regularized continuous covariance model may be claimed until implemented and validated.",
    ),
    DiagnosticContract(
        "Q22",
        "not_resolved",
        "discretized_gaussian_contract",
        "cloud_panel",
        "benchmarks/validation/feature_covariance_calibration.py",
        "Discretized Gaussian data with known latent and observed nulls, Bernoulli/categorical approximations, and selected spectra.",
        "Approximation-contract table comparing Bernoulli, categorical, and latent-Gaussian references.",
        "Pass when one reference family gives calibrated p-values and stable selected spectra over discretization levels.",
        "Discretized Gaussian cases remain benchmark variants without production distributional interpretation.",
    ),
    DiagnosticContract(
        "Q23",
        "partially_formalized",
        "mixed_feature_contract",
        "cloud_panel",
        "benchmarks/validation/feature_covariance_calibration.py",
        "Mixed Bernoulli, categorical, and continuous blocks with per-block nulls, cross-block independence/dependence settings, and planted signals.",
        "Mixed-feature calibration table with blockwise and aggregate projected-Wald p-values.",
        "Pass when blockwise contracts compose without inflating aggregate false splits.",
        "Mixed-feature method claims need this block-composition validation.",
    ),
    DiagnosticContract(
        "Q24",
        "partially_diagnosed",
        "traversal_fdr",
        "cloud_panel",
        "benchmarks/validation/traversal_sibling_fdr_null.py",
        "Traversal trees with valid synthetic p-values, fixed-tree Wald p-values, selected-tree Wald p-values, and active calibration decisions.",
        "Layered traversal FDP, false rejection, support failure, and final-cluster false-split metrics.",
        "Pass when the active traversal layer controls the declared target under null and reports planted recovery under signal.",
        "Traversal-aligned FDR remains unproved until active-layer null simulations pass.",
    ),
    DiagnosticContract(
        "Q25",
        "not_resolved",
        "hierarchical_error_target",
        "theory_plus_simulation",
        "benchmarks/validation/traversal_sibling_fdr_null.py",
        "Candidate error targets over edge path, traversal depth, selected hierarchy, and final clusters.",
        "Target comparison table with estimand definition, FDP/FWER estimate, and method compatibility.",
        "Pass when one target is mathematically defined and empirically estimable with valid diagnostics.",
        "Sibling FDR language must stay qualified until the target is chosen.",
    ),
    DiagnosticContract(
        "Q26",
        "idea_only",
        "pass_through_same_evidence",
        "full_suite",
        "not_implemented",
        "Pass-through runs with same-evidence scores, descendant evidence, parent size, projection dimension, and barycentric balance.",
        "Ablation table for baseline pass-through, no pass-through, same-evidence penalty, and barycentric-balance penalty.",
        "Pass when the penalty reduces null over-splitting without losing planted descendant recovery.",
        "Barycentric pass-through claims need an implemented ablation.",
    ),
    DiagnosticContract(
        "Q27",
        "not_resolved",
        "pass_through_threshold_function",
        "prototype_then_cloud",
        "not_implemented",
        "Predeclared tau functions using same evidence, descendant evidence, n, dimension, and barycentric balance.",
        "Threshold-function grid with null over-split, planted under-split, and sensitivity metrics.",
        "Pass when one tau function has stable null and planted behavior across families.",
        "No tau functional form should enter production without grid validation.",
    ),
    DiagnosticContract(
        "Q28",
        "not_resolved",
        "pass_through_ablation",
        "full_suite",
        "not_implemented",
        "Benchmark suite with pass-through on/off, threshold variants, null and planted labels, and final cluster counts.",
        "Under-split/over-split/ARI table by traversal policy.",
        "Pass when pass-through improves under-splits without unacceptable null over-splitting under a declared criterion.",
        "Pass-through remains an algorithmic policy until this ablation is locked.",
    ),
    DiagnosticContract(
        "Q29",
        "idea_only",
        "phylogenetic_tree_space_null",
        "prototype_then_cloud",
        "not_implemented",
        "Tree-valued phylogenetic objects with Frechet/BHV barycenters, clade nulls, and Euclidean-control embeddings.",
        "Tree-space false-split calibration and comparison to ordinary Euclidean summaries.",
        "Pass when tree-space barycenter variance controls direct sibling false splits inside null clades.",
        "Phylogenetic claims remain future work until tree-valued nulls exist.",
    ),
    DiagnosticContract(
        "Q30",
        "partially_diagnosed",
        "failure_taxonomy",
        "full_suite",
        "benchmarks/diagnostics/oracle/run_gate_path_trace.py",
        "All benchmark cases with oracle recoverability, gate traces, metric/linkage settings, calibration statuses, and final outcomes.",
        "Failure taxonomy manifest assigning gate, metric, linkage, calibration, representation, or recoverability labels.",
        "Pass when every failed case has exactly one primary label and optional secondary labels with supporting metrics.",
        "Benchmark failure narratives remain incomplete until taxonomy coverage is total.",
    ),
    DiagnosticContract(
        "Q31",
        "partially_diagnosed",
        "tree_unrecoverable_checklist",
        "full_suite",
        "benchmarks/diagnostics/oracle/run_oracle_tree_recoverability.py",
        "Tree-unrecoverable cases with oracle cuts, alternate metrics/linkage, gate traces, and representation settings.",
        "Required diagnostic checklist and manifest before metric/linkage changes.",
        "Pass when every tree_unrecoverable case has oracle, metric, linkage, and gate diagnostics.",
        "Do not change metrics/linkage for unrecoverable cases without the checklist.",
    ),
    DiagnosticContract(
        "Q32",
        "partially_diagnosed",
        "representation_metric_linkage_gate",
        "full_suite",
        "benchmarks/diagnostics/analysis/analyze_relationships.py",
        "Diffuse Gaussian, heavy-overlap, and SBM cases crossed by representation, metric, linkage, gate constants, and calibration state.",
        "Ablation table separating representation, distance, linkage, and gate contributions.",
        "Pass when each case family has a stable primary failure source across seeds.",
        "Case-family fixes should wait for factor attribution.",
    ),
    DiagnosticContract(
        "Q33",
        "not_resolved",
        "locked_manuscript_evidence",
        "manual_manifest",
        "benchmarks/validation/method_constants_manifest.py",
        "Candidate benchmark outputs, code commit, run commands, seeds, endpoints, confidence intervals, and limitations.",
        "Locked manuscript-results manifest with include/exclude rationale and frozen artifact paths.",
        "Pass when every manuscript result references a locked artifact with reproducible provenance.",
        "No result should be called manuscript-ready without locked provenance.",
    ),
    DiagnosticContract(
        "Q34",
        "partially_diagnosed",
        "method_constant_validation",
        "full_suite",
        "benchmarks/validation/method_constants_manifest.py",
        "Method constants with benchmark-performance grids, null calibration grids, planted power, and decision rules.",
        "Constant evidence manifest separating benchmark default, calibrated default, and implementation default.",
        "Pass when each constant has complete required fields and a declared decision status.",
        "Constants remain implementation defaults or benchmark defaults until evidence manifests are complete.",
    ),
    DiagnosticContract(
        "Q35",
        "partially_diagnosed",
        "edge_alpha_type1",
        "cloud_panel",
        "benchmarks/validation/selected_edge_type1_geometry.py",
        "Null feature families with fixed-tree and selected-tree controls across edge alpha and hierarchy reconstruction modes.",
        "Edge Type-I table with frontier rejection, all-edge rejection, final false splits, and categorical fixed-tree diagnostics.",
        "Pass when edge alpha controls the declared null edge target under the selected-tree regime or a correction is validated.",
        "Edge alpha remains benchmark-supported rather than Type-I justified.",
    ),
    DiagnosticContract(
        "Q36",
        "partially_diagnosed",
        "sibling_alpha_null_power",
        "cloud_panel",
        "benchmarks/validation/traversal_sibling_fdr_null.py",
        "Sibling alpha grid with null selected-tree calibration, planted sibling contrasts, support decisions, and final clusters.",
        "Null/power curve for sibling alpha with active calibration and support-failure accounting.",
        "Pass when alpha 0.01 or a replacement controls null target and retains planted power.",
        "Sibling alpha remains benchmark-supported but not null/power validated.",
    ),
    DiagnosticContract(
        "Q37",
        "not_resolved",
        "final_cluster_count_control",
        "cloud_panel",
        "benchmarks/validation/alpha_grid_search.py",
        "Null and planted panels with final cluster counts, true cluster counts, over/under split labels, and active calibration decisions.",
        "Cluster-count control simulation with family-wise over-split and under-split endpoints.",
        "Pass when final cluster-count error meets a declared control target under null and planted settings.",
        "Local p-value diagnostics do not imply final cluster-count control.",
    ),
    DiagnosticContract(
        "Q38",
        "partially_diagnosed",
        "planted_power",
        "cloud_panel",
        "benchmarks/validation/alpha_grid_search.py",
        "Predeclared planted structures, effect sizes, sample sizes, feature families, and calibration states.",
        "Power curves for split recovery, ARI, final cluster count, and support failure by effect size.",
        "Pass when planted power is reported with uncertainty and method constants fixed before evaluation.",
        "Benchmark ARI grids are not a locked power study.",
    ),
    DiagnosticContract(
        "Q39",
        "not_resolved",
        "biological_application_selection",
        "manual_selection",
        "wiki/analyses/manuscript-life-science-readiness.md",
        "Candidate biological datasets with preprocessing feasibility, ground truth, baselines, licensing, and manuscript relevance.",
        "Application selection matrix and one locked analysis plan.",
        "Pass when one dataset is selected with reproducible preprocessing and comparison baselines.",
        "No real-data result can be manuscript-ready without this selection.",
    ),
    DiagnosticContract(
        "Q40",
        "not_resolved",
        "results_promotion_manifest",
        "manual_manifest",
        "benchmarks/validation/method_constants_manifest.py",
        "Generated outputs proposed for manuscript use with provenance, commit, seeds, run commands, and limitations.",
        "Promoted-results manifest and wiki index coverage.",
        "Pass when every promoted output has locked provenance and matches manuscript claims.",
        "Generated outputs remain candidate evidence until promoted.",
    ),
    DiagnosticContract(
        "Q41",
        "partially_diagnosed",
        "manuscript_overclaim_audit",
        "manual_manifest",
        "manuscript/sections/method/assumptions_validation.tex",
        "Manuscript claims, assumption table rows, source-ledger statuses, and diagnostic-contract statuses.",
        "Overclaim audit mapping every claim to theorem, validation artifact, diagnostic-only evidence, or future work.",
        "Pass when every approximation is labeled and no diagnostic-only result is phrased as production calibration.",
        "Submission text must wait for the overclaim audit.",
    ),
    DiagnosticContract(
        "Q42",
        "partially_diagnosed",
        "selected_region_barycenter_derivative",
        "theory_plus_simulation",
        "benchmarks/diagnostics/calibration/root_selected_region_margins.py",
        "Root selected-region margins, selected barycenter memberships, null-whitened signed distances, tie cells, and active sets.",
        "Derivative/tangent-cone diagnostic comparing first-order selected-barycenter predictions to simulations.",
        "Pass when the selected barycenter map derivative predicts selected-region boundary behavior within a declared tolerance.",
        "Root margin geometry remains first-order descriptive until derivative validation passes.",
    ),
    DiagnosticContract(
        "Q43",
        "partially_diagnosed",
        "full_selected_law",
        "theory_plus_cloud",
        "benchmarks/diagnostics/calibration/root_selected_region_margins.py",
        "Selected barycenter membership changes, selected projections, Tree-BH cells, sibling FDR/inflation decisions, and non-root focal contexts.",
        "Full selected-law diagnostic with contribution attribution for projection, Tree-BH, sibling FDR, and barycenter membership.",
        "Pass when the combined selected-law variables explain held-out selected-ratio tails in admissible contexts.",
        "The barycentric identity remains local until lifted to the full selected law.",
    ),
    DiagnosticContract(
        "Q44",
        "partially_diagnosed",
        "internal_vs_selected_support_mixing",
        "cloud_panel",
        "benchmarks/diagnostics/calibration/internal_vs_selected_hierarchy_inflation.py",
        "Matched selected-null and planted-nonnull panels with internal support labels, selected-hierarchy tails, and support-mixing controls.",
        "Comparison table separating real selected-tail inflation from overpenalty caused by selected-nonnull support mixing.",
        "Pass when blockers are classified as true selected-tail inflation, support-mixing overpenalty, or unsupported context with stable criteria.",
        "Internal empirical-inflation blockers cannot be interpreted without this separation.",
    ),
)


def diagnostic_contract_table() -> pd.DataFrame:
    """Return the full Q1-Q44 diagnostic contract table."""
    rows = [asdict(contract) for contract in CONTRACTS]
    return pd.DataFrame.from_records(rows)[list(REQUIRED_FIELDS)]


def diagnostic_contract_summary(table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return counts by evidence status, diagnostic scale, and family."""
    contract_table = diagnostic_contract_table() if table is None else table
    rows: list[dict[str, object]] = []
    for summary_type, column in (
        ("current_evidence_status", "current_evidence_status"),
        ("diagnostic_scale", "diagnostic_scale"),
        ("diagnostic_family", "diagnostic_family"),
        ("diagnostic_contract_status", "diagnostic_contract_status"),
    ):
        counts = Counter(str(value) for value in contract_table[column])
        for value, count in sorted(counts.items()):
            rows.append(
                {
                    "summary_type": summary_type,
                    "value": value,
                    "n_questions": int(count),
                    "study_role": STUDY_ROLE,
                }
            )
    return pd.DataFrame.from_records(rows)


def validate_diagnostic_contracts(table: pd.DataFrame | None = None) -> list[str]:
    """Return contract completeness errors."""
    contract_table = diagnostic_contract_table() if table is None else table
    errors: list[str] = []
    missing_columns = set(REQUIRED_FIELDS) - set(contract_table.columns)
    if missing_columns:
        errors.append(f"missing required columns: {sorted(missing_columns)!r}")
        return errors

    expected_questions = {f"Q{index}" for index in range(1, 45)}
    observed_questions = set(contract_table["question_id"].astype(str))
    if observed_questions != expected_questions:
        errors.append(
            "question_id set mismatch: "
            f"missing={sorted(expected_questions - observed_questions)!r}, "
            f"extra={sorted(observed_questions - expected_questions)!r}"
        )
    if contract_table["question_id"].duplicated().any():
        duplicates = sorted(
            contract_table.loc[contract_table["question_id"].duplicated(), "question_id"]
            .astype(str)
            .unique()
        )
        errors.append(f"duplicate question ids: {duplicates!r}")

    for column in REQUIRED_FIELDS:
        empty = contract_table[column].astype(str).str.strip().eq("")
        if bool(empty.any()):
            bad_ids = contract_table.loc[empty, "question_id"].astype(str).tolist()
            errors.append(f"{column} has empty values for {bad_ids!r}")
    invalid_status = contract_table["diagnostic_contract_status"].ne(CONTRACT_STATUS)
    if bool(invalid_status.any()):
        bad_ids = contract_table.loc[invalid_status, "question_id"].astype(str).tolist()
        errors.append(f"non-fully-specified diagnostic contracts: {bad_ids!r}")
    invalid_role = contract_table["study_role"].ne(STUDY_ROLE)
    if bool(invalid_role.any()):
        bad_ids = contract_table.loc[invalid_role, "question_id"].astype(str).tolist()
        errors.append(f"invalid study role for {bad_ids!r}")
    return errors


def run_open_question_full_diagnostic_contract(*, output_dir: Path) -> dict[str, Path]:
    """Write the full diagnostic contract CSV, summary CSV, and manifest."""
    table = diagnostic_contract_table()
    errors = validate_diagnostic_contracts(table)
    if errors:
        raise ValueError("Invalid open-question diagnostic contracts: " + "; ".join(errors))

    summary = diagnostic_contract_summary(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "open_question_full_diagnostic_contract.csv"
    summary_path = output_dir / "open_question_full_diagnostic_summary.csv"
    manifest_path = output_dir / "manifest.json"
    table.to_csv(contract_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest: Mapping[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": format_timestamp_utc(),
        "study_role": STUDY_ROLE,
        "n_questions": int(table.shape[0]),
        "diagnostic_contract_status": CONTRACT_STATUS,
        "outputs": {
            "contract": str(contract_path),
            "summary": str(summary_path),
        },
        "interpretation": (
            "Every open question has a fully specified diagnostic contract. "
            "This does not promote any diagnostic result to a method claim."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"contract": contract_path, "summary": summary_path, "manifest": manifest_path}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_open_question_full_diagnostic_contract(output_dir=args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CONTRACTS",
    "CONTRACT_STATUS",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "STUDY_ROLE",
    "diagnostic_contract_summary",
    "diagnostic_contract_table",
    "run_open_question_full_diagnostic_contract",
    "validate_diagnostic_contracts",
]
