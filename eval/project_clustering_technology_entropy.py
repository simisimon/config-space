"""
Calculate configuration entropy for clusters of projects using a specific technology.

This script measures how diverse the configuration patterns are within each cluster.
High entropy = projects configure the technology in diverse ways
Low entropy = projects configure the technology consistently
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
        description="Calculate configuration entropy for each cluster of a technology."
    )
    parser.add_argument(
        "--technology",
        type=str,
        required=True,
        help="Technology name (e.g., 'docker-compose', 'nodejs', 'npm')",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Clustering method (e.g., 'agglomerative', 'hdbscan'). "
             "If not provided, processes all available methods.",
    )
    parser.add_argument(
        "--data-dir",
        default="../data/project_clustering_technology",
        help="Directory containing clustering results (default: ../data/project_clustering_technology).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tech_safe = args.technology.replace("/", "_").replace(" ", "_")

    # Find all available clustering methods for this technology
    if args.method:
        methods = [args.method]
    else:
        # Auto-detect methods from available files
        assignment_files = list(data_dir.glob(f"{tech_safe}_cluster_assignments_*.csv"))
        methods = [f.stem.replace(f"{tech_safe}_cluster_assignments_", "") for f in assignment_files]

        if not methods:
            print(f"No clustering result files found for technology: {args.technology}")
            print(f"Run project_clustering_technology.py first with --technology {args.technology}")
            return

    print(f"Processing technology: {args.technology}")
    print(f"Processing methods: {methods}")

    for method in methods:
        print(f"\n{'='*80}")
        print(f"Processing method: {method}")
        print(f"{'='*80}")

        # Determine file paths
        assign_path = data_dir / f"{tech_safe}_cluster_assignments_{method}.csv"
        config_path = data_dir / f"{tech_safe}_config_matrix_{method}.csv"
        output_path = data_dir / f"{tech_safe}_cluster_entropy_{method}.csv"

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

        # Get configuration option columns (all columns except project, cluster, num_options, num_files)
        config_cols = [c for c in df.columns if c not in ("project", "cluster", "num_options", "num_files")]

        print(f"Found {len(config_cols)} configuration options")
        print(f"Found {len(df)} projects across {df['cluster'].nunique()} clusters")

        cluster_entropies = []

        for cluster_id, group in df.groupby("cluster"):
            # Sum up configuration option usage in this cluster
            config_usage = group[config_cols].sum()

            # Calculate entropy
            H = calculate_entropy(config_usage)

            # Normalize by maximum possible entropy (log of number of options)
            H_norm = H / log(len(config_cols)) if len(config_cols) > 0 else 0.0

            cluster_entropies.append({
                "cluster": int(cluster_id),
                "entropy": H,
                "entropy_normalized": H_norm,
                "num_projects": len(group)
            })

        entropy_df = pd.DataFrame(cluster_entropies)

        # Sort by entropy (descending)
        entropy_df = entropy_df.sort_values(by="entropy", ascending=False)

        print(f"\nCluster Configuration Entropy ({method}):")
        print(entropy_df.to_string(index=False))

        # Save to CSV
        entropy_df.to_csv(output_path, index=False)
        print(f"\nWrote entropy results to: {output_path}")


if __name__ == "__main__":
    main()
