"""
Calculate configuration entropy for clusters produced by cluster_technology_stack_config.py.

This script measures how diverse the configuration patterns are within each config cluster.
High entropy = projects configure options in diverse ways
Low entropy = projects configure options consistently
"""

import argparse
import pandas as pd
import numpy as np
from math import log
from pathlib import Path


def calculate_entropy(counts):
    """Calculate Shannon entropy from a count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def main():
    parser = argparse.ArgumentParser(
        description="Calculate configuration entropy for each config cluster within an ecosystem."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Name of the directory of a company (e.g., 'netflix', 'disney').",
    )
    parser.add_argument(
        "--ecosystem",
        type=int,
        default=None,
        help="Ecosystem ID to process. If not provided, processes all available ecosystems.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="cluster",
        help="Prefix used in clustering output files (default: 'cluster').",
    )

    args = parser.parse_args()

    data_dir = Path(f"../../data/{args.input}/clustering/technology_stack_config")

    # Find all available ecosystems
    if args.ecosystem is not None:
        ecosystems = [args.ecosystem]
    else:
        # Auto-detect ecosystems from available files
        assignment_files = list(data_dir.glob(f"{args.prefix}_*_config_project_clusters.csv"))
        ecosystems = []
        for f in assignment_files:
            # Extract ecosystem ID from filename like "cluster_0_config_project_clusters.csv"
            parts = f.stem.replace(f"{args.prefix}_", "").replace("_config_project_clusters", "")
            try:
                ecosystems.append(int(parts))
            except ValueError:
                continue
        ecosystems = sorted(ecosystems)

        if not ecosystems:
            print(f"No clustering result files found with prefix '{args.prefix}'.")
            print("Run cluster_technology_stack_config.py first.")
            return

    print(f"Processing ecosystems: {ecosystems}")

    all_entropies = []

    for ecosystem in ecosystems:
        print(f"\n{'='*80}")
        print(f"Processing ecosystem: {ecosystem}")
        print(f"{'='*80}")

        # Determine file paths
        assign_path = data_dir / f"{args.prefix}_{ecosystem}_config_project_clusters.csv"
        config_path = data_dir / f"{args.prefix}_{ecosystem}_config_matrix.csv"
        output_path = data_dir / f"{args.prefix}_{ecosystem}_config_entropy.csv"

        # Check if files exist
        if not assign_path.exists():
            print(f"  Skipping: {assign_path} not found")
            continue
        if not config_path.exists():
            print(f"  Skipping: {config_path} not found")
            continue

        # Load data
        assign = pd.read_csv(assign_path)
        config = pd.read_csv(config_path)

        # Merge assignments with configuration matrix
        df = pd.merge(assign, config, on="project")

        # Get configuration option columns (all columns except project and config_cluster)
        config_cols = [c for c in df.columns if c not in ("project", "config_cluster")]

        print(f"Found {len(config_cols)} configuration options")
        print(f"Found {len(df)} projects across {df['config_cluster'].nunique()} clusters")

        cluster_entropies = []

        for cluster_id, group in df.groupby("config_cluster"):
            # Sum up configuration option usage in this cluster
            config_usage = group[config_cols].sum()

            # Calculate entropy
            H = calculate_entropy(config_usage)

            # Normalize by maximum possible entropy (log of number of options)
            H_max = log(len(config_cols)) if len(config_cols) > 1 else 1.0
            H_norm = H / H_max

            cluster_entropies.append({
                "ecosystem": ecosystem,
                "config_cluster": int(cluster_id),
                "entropy": H,
                "entropy_normalized": H_norm,
                "num_projects": len(group),
                "num_options": len(config_cols),
            })

        entropy_df = pd.DataFrame(cluster_entropies)

        # Sort by entropy (descending)
        entropy_df = entropy_df.sort_values(by="entropy", ascending=False)

        print(f"\nCluster Configuration Entropy:")
        print(entropy_df.to_string(index=False))

        # Save to CSV
        entropy_df.to_csv(output_path, index=False)
        print(f"\nWrote entropy results to: {output_path}")

        all_entropies.append(entropy_df)

    # If processing multiple ecosystems, also write a combined summary
    if len(all_entropies) > 1:
        combined_df = pd.concat(all_entropies, ignore_index=True)
        combined_path = data_dir / f"{args.prefix}_all_config_entropy.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\n{'='*80}")
        print(f"Wrote combined entropy results to: {combined_path}")


if __name__ == "__main__":
    main()
