#!/usr/bin/env python3
"""
Quick test to verify global weighting is working by directly modifying config.py file.
"""

import os
import sys
import shutil
from datetime import datetime

# Backup original config
config_file = "kl_clustering_analysis/config.py"
backup_file = f"{config_file}.backup"

print("Creating backup of config.py...")
shutil.copy(config_file, backup_file)

try:
    # Read config
    with open(config_file, "r") as f:
        config_content = f.read()

    # Test with DISABLED
    print("\n" + "=" * 80)
    print("TEST 1: Global weighting DISABLED")
    print("=" * 80 + "\n")

    # Modify to False
    modified_content = config_content.replace(
        "USE_GLOBAL_DIVERGENCE_WEIGHTING = True",
        "USE_GLOBAL_DIVERGENCE_WEIGHTING = False",
    )
    with open(config_file, "w") as f:
        f.write(modified_content)

    # Import fresh
    if "kl_clustering_analysis" in sys.modules:
        # Remove all kl_clustering_analysis modules
        to_remove = [
            k for k in sys.modules.keys() if k.startswith("kl_clustering_analysis")
        ]
        for k in to_remove:
            del sys.modules[k]

    from kl_clustering_analysis import config
    from kl_clustering_analysis.benchmarking.run_clustering_benchmark import (
        main as benchmark_main,
    )

    print(
        f"Config loaded: USE_GLOBAL_DIVERGENCE_WEIGHTING = {config.USE_GLOBAL_DIVERGENCE_WEIGHTING}"
    )
    print(f"               GLOBAL_WEIGHT_METHOD = {config.GLOBAL_WEIGHT_METHOD}")
    print()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    off_file = f"results/benchmark_global_OFF_{timestamp}.csv"
    benchmark_main(output_file=off_file)
    print(f"\n✅ OFF run complete: {off_file}\n")

    # Test with ENABLED
    print("\n" + "=" * 80)
    print("TEST 2: Global weighting ENABLED")
    print("=" * 80 + "\n")

    # Restore original (True)
    with open(config_file, "w") as f:
        f.write(config_content)

    # Remove modules again
    to_remove = [
        k for k in sys.modules.keys() if k.startswith("kl_clustering_analysis")
    ]
    for k in to_remove:
        del sys.modules[k]

    # Import fresh again
    from kl_clustering_analysis import config as config2
    from kl_clustering_analysis.benchmarking.run_clustering_benchmark import (
        main as benchmark_main2,
    )

    print(
        f"Config loaded: USE_GLOBAL_DIVERGENCE_WEIGHTING = {config2.USE_GLOBAL_DIVERGENCE_WEIGHTING}"
    )
    print(f"               GLOBAL_WEIGHT_METHOD = {config2.GLOBAL_WEIGHT_METHOD}")
    print()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    on_file = f"results/benchmark_global_ON_{timestamp}.csv"
    benchmark_main2(output_file=on_file)
    print(f"\n✅ ON run complete: {on_file}\n")

    print("\n" + "=" * 80)
    print("FILES CREATED")
    print("=" * 80)
    print(f"OFF: {off_file}")
    print(f"ON:  {on_file}")
    print("\nNow run comparison with: python compare_global_weighting_impact.py")

finally:
    # Restore original config
    print("\nRestoring original config.py...")
    shutil.move(backup_file, config_file)
    print("Done!")
