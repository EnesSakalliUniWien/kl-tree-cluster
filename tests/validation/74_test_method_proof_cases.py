from benchmarks.shared.cases import get_test_cases_by_suite
from benchmarks.shared.generators.generate_case_data import generate_case_data


def test_method_proof_suite_lists_traceable_cases():
    cases = get_test_cases_by_suite("method_proof")

    assert len(cases) == 10
    assert {case["name"] for case in cases} >= {
        "bar_binary_balanced_4c",
        "cat_simplex_face_rare_20cat",
        "cont_lowrank_pggn_shrinkage",
    }
    for case in cases:
        assert case["benchmark_role"]
        assert case["mathematical_target"]
        assert case["expected_failure_modes"]
        assert "selected_ratio" in case["required_trace_columns"]
        assert case["admissibility_rule"] == (
            "fail_closed_until_trace_support_and_heldout_precision_pass"
        )


def test_method_proof_generators_satisfy_case_data_contract():
    cases = {case["name"]: case for case in get_test_cases_by_suite("method_proof")}
    selected_names = [
        "bar_binary_balanced_4c",
        "mp_spike_below_bbp_continuous",
        "cat_simplex_face_rare_20cat",
        "phylo_brownian_null_16taxa",
    ]

    for name in selected_names:
        data_df, labels, original, metadata = generate_case_data(cases[name])
        assert data_df.shape[0] == metadata["n_samples"]
        assert data_df.shape[0] == labels.shape[0]
        assert original.shape[0] == labels.shape[0]
        assert metadata["source_family"]
        assert metadata["feature_representation"]
        if metadata["feature_representation"] == "continuous":
            assert metadata["requires_precomputed_kl_distance"] is True
            assert metadata["precomputed_distance_condensed"] is not None
