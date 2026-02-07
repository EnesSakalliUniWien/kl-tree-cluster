"""Stochastic Block Model (graph-based) test cases."""

SBM_CASES = {
    "sbm_graphs": [
        {
            "name": "sbm_clear_small",
            "generator": "sbm",
            "sizes": [30, 30],
            "p_intra": 0.12,
            "p_inter": 0.005,
            "seed": 123,
        },
        {
            "name": "sbm_moderate",
            "generator": "sbm",
            "sizes": [50, 40, 30],
            "p_intra": 0.08,
            "p_inter": 0.02,
            "seed": 124,
        },
        {
            "name": "sbm_hard",
            "generator": "sbm",
            "sizes": [40, 40, 40],
            "p_intra": 0.05,
            "p_inter": 0.04,
            "seed": 125,
        },
    ],
}
