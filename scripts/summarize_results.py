import pandas as pd
import os


def summarize():
    input_path = "benchmarks/results/full_benchmark_alpha_0p001_now.csv"
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    print(f"Loading results from {input_path}...")
    df = pd.read_csv(input_path)

    # Check for duplicates
    # We expect unique (case_id, method) pairs
    duplicates = df[df.duplicated(subset=["case_id", "method"], keep=False)]
    if not duplicates.empty:
        print(
            f"Found {len(duplicates)} duplicate entries. Keeping the last execution for each."
        )
        # Sort by something? We don't have a timestamp, but usually last appended is newest.
        # We can just drop duplicates keeping 'last'.
        df = df.drop_duplicates(subset=["case_id", "method"], keep="last")

    methods_to_test = ["kl", "kl_rogerstanimoto"]
    df = df[df["method"].isin(methods_to_test)]

    # Calculate Mean ARI
    print("\nMean ARI by Method:")
    mean_ari = df.groupby("method")["ari"].mean()
    print(mean_ari)

    # Create Pivot Table for direct comparison
    pivot = df.pivot(index="case_id", columns="method", values="ari")

    # Drop rows where we don't have both results (if any)
    pivot = pivot.dropna()

    pivot["diff (kl - rt)"] = pivot["kl"] - pivot["kl_rogerstanimoto"]

    # Identify cases with significant differences
    # Using a small epsilon for float comparison
    significant_diff = pivot[pivot["diff (kl - rt)"].abs() > 1e-4].copy()
    significant_diff = significant_diff.sort_values("diff (kl - rt)", ascending=False)

    print(f"\nTotal Cases Compared: {len(pivot)}")
    print(f"Cases with different scores: {len(significant_diff)}")

    print("\nTop 5 Pro-Hamming Cases (KL > RT):")
    print(significant_diff.head(5))

    print("\nTop 5 Pro-Rogers-Tanimoto Cases (RT > KL):")
    print(significant_diff.tail(5))

    output_pivot_path = "benchmarks/results/benchmark_summary_pivot.csv"
    pivot.to_csv(output_pivot_path)
    print(f"\nSummary pivot saved to {output_pivot_path}")


if __name__ == "__main__":
    summarize()
