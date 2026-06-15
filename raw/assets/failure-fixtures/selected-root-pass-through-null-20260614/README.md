# Selected Root Pass-Through Null Failure Fixture

This fixture is a compact global-null Bernoulli dataset that reproduces the current selected-root/pass-through failure.

- Case geometry: `overlap_mod_4c_small` dimensions only.
- Data role: null, generated as iid Bernoulli(0.5), so the true cluster count is 1.
- Matrix size: 400 samples x 80 binary features.
- Seed: `20310054`.
- Profile: `fixed_coordinate_global_passthrough_refined_v1`.

Observed replay result:

- `found_clusters = 3` despite `true_clusters = 1`.
- `root_sibling_p_value ~= 6.72e-6`, showing selected-root anti-conservatism in the raw root statistic.
- `root_sibling_open_after_guards = false`.
- `root_stability_guard_blocked = true`.
- One descendant sibling remains open after guards and produces the false split.

Replay command:

```bash
python -m benchmarks.diagnostics.calibration.fixed_sibling_gate_profile_validation \
  --output-dir /tmp/klte_clean_failure_fixture_replay \
  --suite full \
  --case-names overlap_mod_4c_small \
  --profiles fixed_coordinate_global_passthrough_refined_v1 \
  --data-roles null \
  --sibling-alpha 0.01 \
  --edge-alpha 0.001 \
  --replicates 1 \
  --base-seed 20310054
```

Files:

- `feature_matrix.csv`: binary feature matrix.
- `labels.csv`: all-zero truth labels and replay predicted labels.
- `node_annotations.csv`: full node-level gate annotations from the replay.
- `decision_path.csv`: root plus open sibling nodes after guards.
- `tree_edges.csv`: selected average-linkage Hamming tree edges.
- `fixed_sibling_gate_profile_validation_rows.csv`: public runner row verifying the same failure.

This fixture isolates the traversal failure. It intentionally uses the fixed-coordinate profile, so it does not demonstrate the adaptive PCA/MP projected-Wald failure. The MP issue is a separate selected-projection environment: the fixed-subspace chi-square law is valid only when the projection is fixed or conditionally independent, while same-sample adaptive projection and selected topology require their own null law.
