import pandas as pd
import numpy as np
import re
from pathlib import Path


def parse_numpy_array_string(s):
    """
    Parses a string representation of a numpy array like '[0. 2. 0. ...]'
    into a real numpy array. Handles newlines and multiple spaces.
    """
    if not isinstance(s, str):
        return np.array([])

    # Remove brackets
    s = s.strip("[]")
    # Replace newlines with spaces
    s = s.replace("\n", " ")
    # Remove multiple spaces
    s = re.sub(r"\s+", " ", s)
    s = s.strip()

    if not s:
        return np.array([])

    try:
        return np.fromstring(s, sep=" ")
    except Exception as e:
        print(f"Error parsing array: {e}")
        return np.array([])


def analyze_drivers():
    csv_path = Path("results/audit/case_62_kl_divergence_stats.csv")
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Find Root Node (Max leaf count)
    root_idx = df["leaf_count"].idxmax()
    root_row = df.loc[root_idx]
    root_id = root_row["node_id"]

    print(f"=== Analysis for Root Node {root_id} ({root_row['leaf_count']} leaves) ===")
    div_str = root_row.get("kl_divergence_per_column_local", "")
    div_arr = parse_numpy_array_string(div_str)

    if div_arr.size > 0:
        top_k_indices = np.argsort(div_arr)[-10:][::-1]
        print("Top 10 Features driving Root Split:")
        for idx in top_k_indices:
            print(f"  Feature {idx:3d}: KL={div_arr[idx]:.4f}")
    else:
        print("No local divergence data for root.")

    # Find a "Deep" node that split (non-leaf in the decomposition, small leaf count)
    # We want a node that is NOT a leaf in the CSV (is_leaf=False) but has small leaf_count
    # Actually, is_leaf=False in audit means it's an internal node in the full tree.
    # We want one that had a significant split.

    deep_splitters = df[
        (df["is_leaf"] == False)
        & (df["leaf_count"] < 20)
        & (df["Sibling_BH_Different"] == True)
    ].sort_values("leaf_count")

    if not deep_splitters.empty:
        # Pick the smallest one that still split
        deep_node = deep_splitters.iloc[0]
        d_id = deep_node["node_id"]
        print(
            f"\n=== Analysis for Deep Over-Split Node {d_id} ({deep_node['leaf_count']} leaves) ==="
        )
        print(f"Split P-Value: {deep_node['Sibling_Divergence_P_Value']:.2e}")

        div_str = deep_node.get("kl_divergence_per_column_local", "")
        div_arr = parse_numpy_array_string(div_str)

        if div_arr.size > 0:
            top_k_indices = np.argsort(div_arr)[-10:][::-1]
            print("Top 10 Features driving this deep split:")
            for idx in top_k_indices:
                print(f"  Feature {idx:3d}: KL={div_arr[idx]:.4f}")
        else:
            print("No local divergence data.")
    else:
        print("\nNo deep significant splits found (maybe checked wrong column?)")


if __name__ == "__main__":
    analyze_drivers()
