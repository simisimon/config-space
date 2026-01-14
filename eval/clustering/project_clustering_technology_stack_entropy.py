import argparse
import pandas as pd
import numpy as np
from math import log
from pathlib import Path

def calculate_entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def main():
    parser = argparse.ArgumentParser(
        description="Calculate technology entropy for each ecosystem across different clustering methods."
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Clustering method suffix (e.g., 'louvain', 'hdbscan', 'agglomerative'). "
             "If not provided, processes all available methods.",
    )
    parser.add_argument(
        "--data-dir",
        default="../data/project_clustering_technology_stack",
        help="Directory containing clustering results (default: ../data/project_clustering_technologies).",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Find all available clustering methods
    if args.method:
        methods = [args.method]
    else:
        # Auto-detect methods from available files
        assignment_files = list(data_dir.glob("ecosystems_project_assignments_*.csv"))
        methods = [f.stem.replace("ecosystems_project_assignments_", "") for f in assignment_files]

        # Also check for files without suffix (legacy format)
        if (data_dir / "ecosystems_project_assignments.csv").exists():
            methods.append("default")

        if not methods:
            print("No clustering result files found. Run project_clustering_technologies.py first.")
            return

    print(f"Processing methods: {methods}")

    for method in methods:
        print(f"\nProcessing method: {method}")

        # Determine file paths
        if method == "default":
            assign_path = data_dir / "ecosystems_project_assignments.csv"
            tech_path = data_dir / "ecosystems_tech_matrix.csv"
            output_path = data_dir / "ecosystems_entropy.csv"
        else:
            assign_path = data_dir / f"ecosystems_project_assignments_{method}.csv"
            tech_path = data_dir / f"ecosystems_tech_matrix_{method}.csv"
            output_path = data_dir / f"ecosystems_entropy_{method}.csv"

        # Check if files exist
        if not assign_path.exists():
            print(f"  Skipping: {assign_path} not found")
            continue
        if not tech_path.exists():
            print(f"  Skipping: {tech_path} not found")
            continue

        # Load data
        assign = pd.read_csv(assign_path)
        tech = pd.read_csv(tech_path)

        df = pd.merge(assign, tech, on="project")

        tech_cols = [c for c in df.columns if c not in ("project", "ecosystem")]

        ecosystem_entropies = []

        for ecosys, group in df.groupby("ecosystem"):
            tech_usage = group[tech_cols].sum()
            H = calculate_entropy(tech_usage)
            H_norm = H / log(len(tech_cols)) if len(tech_cols) > 0 else 0.0
            ecosystem_entropies.append((ecosys, H, H_norm, len(group)))

        entropy_df = pd.DataFrame(ecosystem_entropies,
                                columns=["ecosystem", "entropy", "entropy_normalized", "num_projects"])

        print(f"\n{entropy_df.to_string()}")

        entropy_df = entropy_df.sort_values(by="entropy", ascending=False)
        entropy_df.to_csv(output_path, index=False)
        print(f"\nWrote entropy results to: {output_path}")


if __name__ == "__main__":
    main()