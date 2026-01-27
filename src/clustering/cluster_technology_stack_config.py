import argparse
import csv
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, adjusted_rand_score


def get_ecosystem_projects(ecosystem_assignments_csv, ecosystem_id, projects_dir):
    """
    Get all projects belonging to a specific ecosystem cluster.

    Args:
        ecosystem_assignments_csv: Path to ecosystems_project_assignments.csv
        ecosystem_id: The ecosystem cluster ID to extract
        projects_dir: Directory containing project JSON files

    Returns:
        List of paths to JSON files for projects in this ecosystem
    """
    ecosystem_projects = []
    df = pd.read_csv(ecosystem_assignments_csv)

    df_projects = df[df["ecosystem"] == ecosystem_id]["project"].tolist()
    ecosystem_projects.extend(df_projects)

    json_files = []
    missing_files = []

    for project_name in ecosystem_projects:
        # Try different naming patterns
        patterns = [
            f"{project_name}_commit.json",
            f"{project_name}_last_commit.json",
        ]
        found = False
        for pattern in patterns:
            json_path = os.path.join(projects_dir, pattern)
            if os.path.exists(json_path):
                json_files.append(json_path)
                found = True
                break
        if not found:
            missing_files.append(project_name)

    print(f"Ecosystem {ecosystem_id}: found {len(json_files)} projects")
    if missing_files:
        print(f"  Warning: {len(missing_files)} projects missing JSON files")

    return json_files


# Types for which we do NOT want to keep the raw value (noise/sensitive/too specific)
VALUELESS_TYPES = {
    "PASSWORD",
    "UNKNOWN",
    "COMMAND",
    "EMAIL",
    "IMAGE",
}

def load_latest_config_pairs(json_path):
    """Load all (concept, option, value, type) pairs from the latest config-related commit."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    project_name = data.get("project_name", os.path.basename(json_path))

    # Handle different JSON structures
    if "config_data" in data:
        # New format: config_data at top level
        cfg_files = data["config_data"].get("config_file_data", [])
    elif "latest_commit_data" in data:
        # Old format: latest_commit_data.network_data
        latest_commit = data.get("latest_commit_data", {})
        if not latest_commit:
            return project_name, []
        network_data = latest_commit.get("network_data", {})
        cfg_files = network_data.get("config_file_data", [])
    else:
        return project_name, []

    pairs = []
    for cfg in cfg_files:
        concept = cfg.get("concept")
        for p in cfg.get("pairs", []):
            option = p.get("option")
            value = p.get("value")
            ptype = p.get("type")
            if concept is None or option is None:
                continue
            pairs.append((concept, option, value, ptype))

    # Deduplicate identical tuples
    pairs = list(set(pairs))
    return project_name, pairs


def build_feature_key(concept, option, value, ptype, keep_values=True):
    """
    Build a feature key for a given (concept, option, value, type).

    Strategy:
      - For some types (VALUELESS_TYPES), ignore the concrete value to avoid
        overly specific or sensitive features: key = concept::option
      - Otherwise, if keep_values is True, include value: key = concept::option::value
    """
    if not keep_values or ptype in VALUELESS_TYPES or value is None:
        return f"{concept}::{option}"
    else:
        return f"{concept}::{option}::{str(value)}"


def build_project_feature_matrix(
        json_files_list,
        min_feature_frequency=3,
        keep_values=True,
        concepts_filter=None,
):
    """
    Build a project × feature binary matrix from JSON config extraction files.

    Args:
        json_files_list: List of paths to JSON files to process
        min_feature_frequency: Minimum number of projects a feature must appear in
        keep_values: Whether to include concrete values in features
        concepts_filter: Set of concepts to keep (e.g., {"maven", "docker-compose"}),
                        or None to keep all.
    """
    project_features = {}
    feature_counter = Counter()

    json_files = sorted(json_files_list)

    if not json_files:
        raise ValueError(f"No JSON files found")

    for path in json_files:
        project_name, pairs = load_latest_config_pairs(path)

        feats = set()
        for concept, option, value, ptype in pairs:
            if concepts_filter is not None and concept not in concepts_filter:
                continue
            key = build_feature_key(concept, option, value, ptype, keep_values=keep_values)
            feats.add(key)

        project_features[project_name] = feats
        feature_counter.update(feats)

    # Filter features by minimum project frequency
    kept_features = sorted(
        [f for f, c in feature_counter.items() if c >= min_feature_frequency]
    )
    feature_index = {f: i for i, f in enumerate(kept_features)}

    # Build binary matrix
    n_projects = len(project_features)
    n_features = len(kept_features)
    X = np.zeros((n_projects, n_features), dtype=int)
    projects = sorted(project_features.keys())

    for row_idx, proj in enumerate(projects):
        feats = project_features[proj]
        for f in feats:
            j = feature_index.get(f)
            if j is not None:
                X[row_idx, j] = 1

    return projects, kept_features, X


def cluster_with_k(dist_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Run agglomerative clustering with a precomputed distance matrix.
    """
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )
    except TypeError:
        # Newer sklearn versions use 'metric' instead of 'affinity'
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    labels = model.fit_predict(dist_matrix)
    return labels


def stability_for_k(dist_matrix: np.ndarray,
                    labels_full: np.ndarray,
                    k: int,
                    subsample_fraction: float,
                    n_repeats: int,
                    random_state: int = 42):
    """
    Estimate clustering stability for a given k via subsampling + ARI.

    - labels_full: clustering of the full dataset for this k
    - For each repeat:
        * sample a subset of projects
        * recluster the subset
        * compute ARI between subset labels and full-data labels restricted to subset
    """
    rng = np.random.RandomState(random_state + k)
    n_samples = dist_matrix.shape[0]
    subset_size = max(int(subsample_fraction * n_samples), k + 2)
    subset_size = min(subset_size, n_samples)

    aris = []
    for _ in range(n_repeats):
        if subset_size <= k:
            break
        subset_idx = rng.choice(n_samples, size=subset_size, replace=False)
        dist_sub = dist_matrix[np.ix_(subset_idx, subset_idx)]
        labels_sub = cluster_with_k(dist_sub, k)
        ari = adjusted_rand_score(labels_full[subset_idx], labels_sub)
        aris.append(ari)

    return np.array(aris)


def cluster_projects(X, n_clusters):
    """
    Cluster projects using Jaccard distance and AgglomerativeClustering.
    Returns both labels and distance matrix.
    """
    if X.shape[1] == 0:
        raise ValueError("No configuration features left; try lowering --min-feature-frequency.")

    dist_matrix = pairwise_distances(X, metric="jaccard")
    labels = cluster_with_k(dist_matrix, n_clusters)
    return labels, dist_matrix


def summarize_clusters(projects, features, X, labels, top_n=15):
    """
    For each cluster, compute:
      - number of projects
      - top features (most common option/values)
    """
    df = pd.DataFrame(X, columns=features)
    df.insert(0, "project", projects)
    df["config_cluster"] = labels

    summaries = []
    for cluster_id, group in df.groupby("config_cluster"):
        n_proj = len(group)
        # feature frequency within cluster
        freq = group[features].sum().sort_values(ascending=False)
        top_feats = freq.head(top_n)
        top_str = "; ".join(f"{feat} ({int(cnt)})" for feat, cnt in top_feats.items())
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "num_projects": int(n_proj),
                "top_features": top_str,
            }
        )

    summary_df = pd.DataFrame(summaries).sort_values("cluster_id")
    return df[["project", "config_cluster"]], summary_df


def plot_pca_embedding(X, labels, projects, output_path, random_state=42, max_example_projects=3):
    """
    2D PCA embedding of the project configuration profiles.

    Args:
        X: Feature matrix
        labels: Cluster labels for each project
        projects: List of project names
        output_path: Path to save the plot
        random_state: Random seed for PCA
        max_example_projects: Maximum number of example projects to show per cluster in legend
    """
    if X.shape[1] < 2:
        print("Skipping PCA embedding: fewer than 2 features.")
        return

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)

    unique_clusters = np.unique(labels)
    plt.figure(figsize=(12, 8))

    for cid in unique_clusters:
        mask = labels == cid
        cluster_projects = [p for p, m in zip(projects, mask) if m]
        n_projects = len(cluster_projects)

        # Select example projects to show in legend
        example_projects = cluster_projects[:max_example_projects]
        examples_str = ", ".join(example_projects)
        if n_projects > max_example_projects:
            examples_str += ", ..."

        label_text = f"Cluster {cid} (n={n_projects})\n  {examples_str}"

        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=label_text,
            alpha=0.7,
            s=20,
        )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("Configuration-profile clusters within tecchnology ecosystems")
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize="small",
        frameon=True,
        ncol=min(3, len(unique_clusters)),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote PCA embedding plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster projects within an ecosystem based on concrete configuration option–value pairs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Name of the directory of a company (e.g., 'netflix', 'disney').",
    )
    parser.add_argument(
        "--ecosystem",
        type=int,
        required=True,
        help="Ecosystem cluster ID (0-19) to analyze. Automatically selects "
             "projects from this ecosystem using the assignments CSV.",
    )
    parser.add_argument(
        "--method",
        default="louvain",
        choices=["louvain", "agglomerative", "hdbscan"],
        help="Clustering method used to determine ecosystem assignments (default: louvain).",
    )
    parser.add_argument(
        "--ecosystem-assignments",
        default=None,
        help="Path to ecosystem assignments CSV. "
             "If not provided, defaults to data/{input}/clustering/technology_stack/ecosystems_project_assignments_{method}.csv",
    )
    parser.add_argument(
        "--projects-dir",
        default=None,
        help="Directory containing all project JSON files. "
             "If not provided, defaults to data/{input}/latest_commit",
    )
    parser.add_argument(
        "--prefix",
        default="cluster",
        help="Prefix for output files (default: cluster)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of configuration-profile clusters to compute. "
             "If not specified, uses stability-based k selection.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=3,
        help="Minimum number of clusters to consider for stability analysis (default: 3).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=15,
        help="Maximum number of clusters to consider for stability analysis (default: 15).",
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.8,
        help="Fraction of projects to sample for each stability run (default: 0.8).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="Number of subsampling runs per k to estimate stability (default: 20).",
    )
    parser.add_argument(
        "--min-feature-frequency",
        type=int,
        default=3,
        help="Minimum number of projects a feature must appear in to be kept (default: 3).",
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="",
        help="Comma-separated list of concepts/technologies to include (e.g., 'maven,docker'). "
             "If empty, all concepts are used.",
    )
    parser.add_argument(
        "--no-values",
        action="store_true",
        help="If set, ignore concrete values and cluster only by (concept,option) presence.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for PCA (default: 42).",
    )

    args = parser.parse_args()

    # Set default paths based on input if not provided
    if args.ecosystem_assignments is None:
        args.ecosystem_assignments = f"../../data/{args.input}/clustering/technology_stack/ecosystems_project_assignments_{args.method}.csv"
    if args.projects_dir is None:
        args.projects_dir = f"../../data/{args.input}/latest_commit"

    # Create output directory if it doesn't exist
    output_dir = f"../../data/{args.input}/clustering/technology_stack_config"
    os.makedirs(output_dir, exist_ok=True)

    concepts_filter = None
    if args.concepts.strip():
        concepts_filter = {c.strip() for c in args.concepts.split(",") if c.strip()}

    # Get list of JSON files from the specified ecosystem
    print(f"Selecting projects from ecosystem {args.ecosystem}...")
    json_files_list = get_ecosystem_projects(
        ecosystem_assignments_csv=args.ecosystem_assignments,
        ecosystem_id=args.ecosystem,
        projects_dir=args.projects_dir,
    )
    if not json_files_list:
        raise ValueError(f"No projects found for ecosystem {args.ecosystem}")

    print(f"Building project × configuration-feature matrix from ecosystem {args.ecosystem} ...")
    projects, features, X = build_project_feature_matrix(
        json_files_list=json_files_list,
        min_feature_frequency=args.min_feature_frequency,
        keep_values=not args.no_values,
        concepts_filter=concepts_filter,
    )
    print(f"  Projects: {len(projects)}")
    print(f"  Features: {len(features)} (min frequency ≥ {args.min_feature_frequency})")

    # Save matrix
    matrix_out = f"{output_dir}/{args.prefix}_{args.ecosystem}_config_matrix.csv"
    df_matrix = pd.DataFrame(X, columns=features)
    df_matrix.insert(0, "project", projects)
    df_matrix.to_csv(matrix_out, index=False)
    print(f"Wrote configuration matrix to: {matrix_out}")

    # Compute distance matrix once
    if X.shape[1] == 0:
        raise ValueError("No configuration features left; try lowering --min-feature-frequency.")
    dist_matrix = pairwise_distances(X, metric="jaccard")
    print("Computed Jaccard distance matrix.")

    # Determine k: either use manual override or stability-based selection
    if args.n_clusters is not None:
        print(f"Using manually specified k={args.n_clusters}")
        best_k = args.n_clusters
    else:
        print("Running stability-based k selection ...")
        k_candidates = list(range(args.k_min, min(args.k_max, dist_matrix.shape[0]) + 1))
        print(f"Evaluating stability for k in {k_candidates} ...")

        stability_rows = []
        for k in k_candidates:
            print(f"\nClustering full data for k={k} ...")
            labels_full = cluster_with_k(dist_matrix, k)

            aris = stability_for_k(
                dist_matrix,
                labels_full,
                k=k,
                subsample_fraction=args.subsample_fraction,
                n_repeats=args.n_repeats,
                random_state=args.random_state,
            )

            if len(aris) == 0:
                print(f"  Skipped k={k} (not enough samples for subset size).")
                continue

            mean_ari = float(aris.mean())
            median_ari = float(np.median(aris))
            std_ari = float(aris.std())

            stability_rows.append({
                "k": k,
                "mean_ari": mean_ari,
                "median_ari": median_ari,
                "std_ari": std_ari,
                "n_repeats_effective": len(aris),
            })

            print(
                f"  Stability for k={k}: mean ARI={mean_ari:.4f}, "
                f"median ARI={median_ari:.4f}, std={std_ari:.4f} "
                f"(over {len(aris)} runs)."
            )

        if not stability_rows:
            raise RuntimeError("No stability results computed; check your parameters.")

        stability_df = pd.DataFrame(stability_rows).sort_values("k")
        stability_out = f"{output_dir}/{args.prefix}_{args.ecosystem}_config_stability.csv"
        stability_df.to_csv(stability_out, index=False)
        print(f"\nWrote stability summary to: {stability_out}")

        # Choose k with highest median ARI (tie-breaker: mean ARI)
        best_row = stability_df.sort_values(
            ["median_ari", "mean_ari"], ascending=[False, False]
        ).iloc[0]
        best_k = int(best_row["k"])
        print(
            f"\nSelected best k={best_k} based on highest median ARI "
            f"(median={best_row['median_ari']:.4f}, mean={best_row['mean_ari']:.4f})."
        )

    print(f"\nClustering projects into {best_k} configuration-profile clusters ...")
    labels = cluster_with_k(dist_matrix, best_k)

    proj_clusters, cluster_summary = summarize_clusters(projects, features, X, labels)

    proj_out = f"{output_dir}/{args.prefix}_{args.ecosystem}_config_project_clusters.csv"
    proj_clusters.to_csv(proj_out, index=False)
    print(f"Wrote project → configuration-cluster assignments to: {proj_out}")

    summary_out = f"{output_dir}/{args.prefix}_{args.ecosystem}_config_cluster_summary.csv"
    cluster_summary.to_csv(summary_out, index=False)
    print(f"Wrote configuration-cluster summary to: {summary_out}")

    embedding_out = f"{output_dir}/{args.prefix}_{args.ecosystem}_config_embedding.png"
    plot_pca_embedding(X, labels, projects, embedding_out, random_state=args.random_state)

    print("Done.")


if __name__ == "__main__":
    main()
