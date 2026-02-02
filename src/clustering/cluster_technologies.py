"""
Cluster projects based on how they configure a specific technology.

This script takes a technology (e.g., "docker-compose", "nodejs", "npm") and clusters
projects based on their configuration patterns for that technology, identifying
common configuration archetypes.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Add parent directory to path for mapping import
sys.path.insert(0, str(Path(__file__).parent.parent))
from mapping import get_technology

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    IGRAPH_AVAILABLE = False


def load_project_configs(data_dir: str, technology: str) -> pd.DataFrame:
    """
    Load all project configuration files and extract data for a specific technology.

    Returns a DataFrame with columns:
    - project: project name
    - config_options: dict of option->value mappings for this technology
    - num_options: number of configuration options
    - num_files: number of files for this technology
    """
    data_path = Path(data_dir)
    projects_data = []

    print(f"Loading projects with technology: {technology}")

    # Support multiple file naming patterns
    json_files = list(data_path.glob("*_commit.json")) + list(data_path.glob("*_last_commit.json"))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            project_name = data.get("project_name", json_file.stem.replace("_last_commit", "").replace("_commit", ""))

            # Handle different JSON structures
            if "config_data" in data:
                # New format: config_data at top level
                network_data = data["config_data"]
                concepts = network_data.get("concepts", [])
            elif "latest_commit_data" in data:
                # Old format: latest_commit_data.network_data
                latest_commit = data.get("latest_commit_data", {})
                network_data = latest_commit.get("network_data", {})
                concepts = network_data.get("concepts", [])
            else:
                continue

            # Check if technology is directly listed in concepts
            technology_found = technology in concepts

            # If not directly listed, check config files with generic concepts using mapping
            if not technology_found:
                config_files = network_data.get("config_file_data", [])
                for config_file in config_files:
                    if config_file.get("concept") in ["yaml", "json", "xml", "toml", "configparser"]:
                        mapped_technology = get_technology(config_file["file_path"])
                        if mapped_technology == technology:
                            technology_found = True
                            break

            # Skip if technology not found
            if not technology_found:
                continue

            # Extract configuration options for this technology
            config_files = network_data.get("config_file_data", [])
            # Store all option-value pairs (handling multiple files with same option)
            option_values = defaultdict(lambda: {"values": set(), "types": set()})
            num_files = 0

            for config_file in config_files:
                # Check if this config file is for our technology
                # Either directly matches or maps to it via mapping.py
                is_target_tech = False
                concept = config_file.get("concept")

                if concept == technology:
                    is_target_tech = True
                elif concept in ["yaml", "json", "xml", "toml", "configparser"]:
                    mapped_technology = get_technology(config_file["file_path"])
                    if mapped_technology == technology:
                        is_target_tech = True

                if is_target_tech:
                    num_files += 1
                    pairs = config_file.get("pairs", [])

                    for pair in pairs:
                        option = pair.get("option", "")
                        value = pair.get("value", "")
                        option_type = pair.get("type", "")

                        # Collect all values for each option across all files
                        if option and value:
                            option_values[option]["values"].add(value)
                            if option_type:
                                option_values[option]["types"].add(option_type)

            # Convert sets to lists for easier processing
            config_options = {
                opt: {
                    "values": list(data["values"]),
                    "types": list(data["types"])
                }
                for opt, data in option_values.items()
            }

            if config_options:
                projects_data.append({
                    "project": project_name,
                    "config_options": config_options,
                    "num_options": len(config_options),
                    "num_files": num_files
                })

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    df = pd.DataFrame(projects_data)
    print(f"Found {len(df)} projects using {technology}")

    return df


def build_configuration_matrix(df: pd.DataFrame, min_option_frequency: int = 2,
                               feature_mode: str = "option_value") -> tuple:
    """
    Build a project � configuration option binary matrix.

    min_option_frequency: drop features that appear in fewer than this number of projects
    feature_mode: "option_value" uses option=value pairs as features,
                  "option_only" uses option names only (ignoring values)
    """
    feature_counts = Counter()

    for config_options in df["config_options"]:
        for option, data in config_options.items():
            if feature_mode == "option_only":
                feature_counts[option] += 1
            else:
                for value in data["values"]:
                    feature_counts[f"{option}={value}"] += 1

    kept_features = sorted([
        f for f, count in feature_counts.items()
        if count >= min_option_frequency
    ])

    feature_label = "options" if feature_mode == "option_only" else "option=value pairs"
    print(f"Configuration matrix: {len(df)} projects x {len(kept_features)} {feature_label} "
          f"(min frequency >= {min_option_frequency})")

    if len(kept_features) == 0:
        raise ValueError(
            f"No {feature_label} found with frequency >= {min_option_frequency}. "
            f"Try lowering --min-option-frequency"
        )

    feature_index = {f: i for i, f in enumerate(kept_features)}
    X = np.zeros((len(df), len(kept_features)), dtype=int)

    for row_idx, config_options in enumerate(df["config_options"]):
        for option, data in config_options.items():
            if feature_mode == "option_only":
                if option in feature_index:
                    X[row_idx, feature_index[option]] = 1
            else:
                for value in data["values"]:
                    pair_key = f"{option}={value}"
                    if pair_key in feature_index:
                        X[row_idx, feature_index[pair_key]] = 1

    return X, kept_features


def compute_jaccard_distance(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Jaccard distance matrix."""
    if X.shape[1] == 0:
        raise ValueError("No configuration options available for clustering")

    dist = pairwise_distances(X, metric="jaccard")
    return dist


def cluster_with_k(dist_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run agglomerative clustering with precomputed distance matrix."""
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average"
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
    labels = model.fit_predict(dist_matrix)
    return labels


def cluster_with_hdbscan(dist_matrix: np.ndarray, min_cluster_size: int = 5, min_samples: int = 3) -> np.ndarray:
    """Run HDBSCAN clustering."""
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is not installed. Install with: pip install hdbscan")

    n_samples = dist_matrix.shape[0]
    min_cluster_size = min(min_cluster_size, n_samples // 2)
    min_samples = min(min_samples, min_cluster_size - 1)

    if min_cluster_size < 2:
        raise ValueError(f"Dataset too small for HDBSCAN: only {n_samples} samples")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(dist_matrix)
    return labels


def evaluate_k_sweep(dist_matrix: np.ndarray, X: np.ndarray, k_range: range) -> pd.DataFrame:
    """
    Evaluate different k values for agglomerative clustering using multiple metrics.

    Returns a DataFrame with columns: k, silhouette_score, n_clusters
    """
    results = []

    for k in k_range:
        if k < 2 or k > dist_matrix.shape[0] - 1:
            continue

        labels = cluster_with_k(dist_matrix, k)

        # Skip if only one cluster was created
        n_clusters = len(set(labels))
        if n_clusters < 2:
            continue

        # Compute silhouette score on distance matrix
        sil_score = silhouette_score(dist_matrix, labels, metric='precomputed')

        results.append({
            'k': k,
            'silhouette_score': sil_score,
            'n_clusters': n_clusters
        })

    return pd.DataFrame(results)


def evaluate_hdbscan_sweep(dist_matrix: np.ndarray, X: np.ndarray,
                           min_cluster_size_range: range) -> pd.DataFrame:
    """
    Evaluate different min_cluster_size values for HDBSCAN using multiple metrics.

    Returns a DataFrame with columns: min_cluster_size, silhouette_score, n_clusters, n_noise
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is not installed. Install with: pip install hdbscan")

    results = []
    n_samples = dist_matrix.shape[0]

    for min_size in min_cluster_size_range:
        if min_size < 2 or min_size > n_samples // 2:
            continue

        labels = cluster_with_hdbscan(dist_matrix, min_cluster_size=min_size)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Skip if no clusters or all noise
        if n_clusters < 1:
            continue

        # Compute silhouette score (excluding noise points for HDBSCAN)
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 0 and n_clusters > 1:
            sil_score = silhouette_score(
                dist_matrix[non_noise_mask][:, non_noise_mask],
                labels[non_noise_mask],
                metric='precomputed'
            )
        else:
            sil_score = -1

        results.append({
            'min_cluster_size': min_size,
            'silhouette_score': sil_score,
            'n_clusters': n_clusters,
            'n_noise': n_noise
        })

    return pd.DataFrame(results)


def summarize_clusters(df: pd.DataFrame, labels: np.ndarray, kept_pairs: list,
                        X: np.ndarray, top_n: int = 10) -> tuple:
    """
    Attach cluster labels and compute top configuration option=value pairs per cluster.

    Args:
        df: DataFrame with project data
        labels: Cluster labels for each project
        kept_pairs: List of option=value pairs (column names from matrix)
        X: Binary matrix of option=value pair presence
        top_n: Number of top pairs to report per cluster
    """
    df = df.copy()
    df["cluster"] = labels

    cluster_summary = []
    for cluster_id, group in df.groupby("cluster"):
        # Get indices of projects in this cluster
        cluster_mask = labels == cluster_id
        cluster_matrix = X[cluster_mask]

        # Count frequency of each option=value pair in this cluster
        pair_counts = cluster_matrix.sum(axis=0)

        # Get top pairs with their counts
        top_indices = np.argsort(pair_counts)[::-1][:top_n]
        top_pairs = [(kept_pairs[i], int(pair_counts[i])) for i in top_indices if pair_counts[i] > 0]

        total_projects = len(group)

        cluster_summary.append({
            "cluster_id": int(cluster_id),
            "num_projects": int(total_projects),
            "top_options": top_pairs  # Now contains option=value pairs
        })

    return df, cluster_summary


def plot_pca_embedding(X: np.ndarray, labels: np.ndarray, output_path: str,
                       technology: str, cluster_summary: list, method: str,
                       n_projects: int, random_state: int = 42):
    """Plot 2D PCA embedding with cluster labels."""
    if X.shape[1] < 2:
        print("Skipping PCA plot: fewer than 2 dimensions")
        return

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)

    explained_var = pca.explained_variance_ratio_
    total_var = explained_var.sum() * 100

    unique_clusters = np.unique(labels)

    plt.figure(figsize=(8, 6))
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        n_points = mask.sum()

        # Build legend label - cluster ID and count only
        legend_label = f"Cluster {cluster_id} (n={n_points})"

        # Add jitter for overlapping points
        X_cluster = X_2d[mask].copy()
        if n_points > 1:
            seed = abs(int(cluster_id)) + 1000
            X_cluster += np.random.RandomState(seed).normal(0, 0.02, X_cluster.shape)

        plt.scatter(X_cluster[:, 0], X_cluster[:, 1], label=legend_label,
                   alpha=0.6, s=30, edgecolors='white', linewidth=0.5)

    plt.xlabel(f"PCA 1 ({explained_var[0]*100:.1f}% var)")
    plt.ylabel(f"PCA 2 ({explained_var[1]*100:.1f}% var)")
    plt.title(f"{technology} ({method}, n={n_projects})")

    # Adjust legend position based on number of clusters
    n_clusters = len(unique_clusters)
    ncol = min(5, n_clusters)
    # Calculate how many rows the legend will have
    n_rows = (n_clusters + ncol - 1) // ncol
    # Adjust vertical position based on number of rows (more rows = more space below)
    legend_y = -0.15 - (n_rows - 1) * 0.05

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, legend_y),
              ncol=ncol, fontsize='small', frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote PCA plot to: {output_path}")


def plot_heatmap(X: np.ndarray, labels: np.ndarray, option_names: list,
                output_path: str, technology: str):
    """Plot heatmap showing option usage by cluster."""
    unique_clusters = np.unique(labels)
    cluster_option_freq = []

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_projects = X[mask]
        option_freq = (cluster_projects.sum(axis=0) / mask.sum()) * 100
        cluster_option_freq.append(option_freq)

    cluster_matrix = np.array(cluster_option_freq)

    # Show top options (used by at least 20% in some cluster)
    max_usage = cluster_matrix.max(axis=0)
    important_options = max_usage >= 20

    if important_options.sum() == 0:
        top_indices = np.argsort(max_usage)[-min(20, len(option_names)):]
        important_options = np.zeros(len(option_names), dtype=bool)
        important_options[top_indices] = True

    filtered_matrix = cluster_matrix[:, important_options]
    filtered_names = [option_names[i] for i in np.where(important_options)[0]]

    # Escape special characters that matplotlib interprets as LaTeX
    def escape_latex_chars(text):
        """Escape characters that trigger matplotlib's mathtext parser."""
        special_chars = ['$', '_', '^', '{', '}', '\\', '%', '&', '#']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text

    escaped_names = [escape_latex_chars(name) for name in filtered_names]

    plt.figure(figsize=(14, max(6, len(unique_clusters) * 0.5)))
    sns.heatmap(
        filtered_matrix,
        xticklabels=escaped_names,
        yticklabels=[f"Cluster {c}" for c in unique_clusters],
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        cbar_kws={'label': '% of projects'},
        linewidths=0.5
    )
    plt.xlabel("Configuration Option")
    plt.ylabel("Cluster")
    plt.title(f"{technology} Configuration Patterns by Cluster")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote heatmap to: {output_path}")


def run_clustering_for_method(method: str, df: pd.DataFrame, X: np.ndarray,
                             option_names: list, dist_matrix: np.ndarray,
                             tech_safe: str, output_dir: Path, args) -> None:
    """Run clustering for a specific method with automatic parameter sweep."""

    print(f"\n{'='*80}")
    print(f"Running {method.upper()} clustering for {args.technology}")
    print(f"{'='*80}")

    # Always run parameter sweep
    if method == "agglomerative":
        k_min, k_max = map(int, args.k_range.split(","))
        k_range = range(k_min, k_max + 1)

        print(f"\nEvaluating k values from {k_min} to {k_max}...")
        sweep_results = evaluate_k_sweep(dist_matrix, X, k_range)

        sweep_out = output_dir / f"{tech_safe}_k_sweep_agglomerative.csv"
        sweep_results.to_csv(sweep_out, index=False)
        print(f"Wrote sweep results to: {sweep_out}")

        # Find and log best k value
        best_row = sweep_results.loc[sweep_results['silhouette_score'].idxmax()]
        best_k = int(best_row['k'])
        best_score = best_row['silhouette_score']

        print(f"\n{'='*80}")
        print(f"BEST k VALUE: {best_k} (silhouette score: {best_score:.4f})")
        print(f"{'='*80}")

        # Show top 5 by silhouette score
        print("\nTop 5 k values by silhouette score:")
        print(sweep_results.nlargest(5, 'silhouette_score')[['k', 'silhouette_score', 'n_clusters']])

        # Use best k for final clustering
        n_clusters = best_k

    elif method == "hdbscan":
        size_min, size_max = map(int, args.min_size_range.split(","))
        size_range = range(size_min, size_max + 1)

        print(f"\nEvaluating min_cluster_size values from {size_min} to {size_max}...")
        sweep_results = evaluate_hdbscan_sweep(dist_matrix, X, size_range)

        sweep_out = output_dir / f"{tech_safe}_min_size_sweep_hdbscan.csv"
        sweep_results.to_csv(sweep_out, index=False)
        print(f"Wrote sweep results to: {sweep_out}")

        # Find and log best min_cluster_size value
        best_row = sweep_results.loc[sweep_results['silhouette_score'].idxmax()]
        best_size = int(best_row['min_cluster_size'])
        best_score = best_row['silhouette_score']
        best_n_clusters = int(best_row['n_clusters'])
        best_n_noise = int(best_row['n_noise'])

        print(f"\n{'='*80}")
        print(f"BEST min_cluster_size VALUE: {best_size}")
        print(f"  Silhouette score: {best_score:.4f}")
        print(f"  Number of clusters: {best_n_clusters}")
        print(f"  Number of noise points: {best_n_noise}")
        print(f"{'='*80}")

        # Show top 5 by silhouette score
        print("\nTop 5 min_cluster_size values by silhouette score:")
        print(sweep_results.nlargest(5, 'silhouette_score')[['min_cluster_size', 'silhouette_score', 'n_clusters', 'n_noise']])

        # Use best min_cluster_size for final clustering
        min_cluster_size = best_size

    print(f"\n{'='*80}")
    print("Sweep complete! Now running final clustering with best parameters...")
    print(f"{'='*80}\n")

    # Save configuration matrix
    matrix_out = output_dir / f"{tech_safe}_config_matrix_{method}.csv"
    matrix_df = pd.DataFrame(X, columns=option_names)
    matrix_df.insert(0, "project", df["project"].values)
    matrix_df.to_csv(matrix_out, index=False)
    print(f"Wrote configuration matrix to: {matrix_out}")

    # Cluster with best parameters
    if method == "hdbscan":
        print(f"\nRunning HDBSCAN (min_cluster_size={min_cluster_size})...")
        labels = cluster_with_hdbscan(dist_matrix, min_cluster_size=min_cluster_size)
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Found {n_clusters_found} clusters and {n_noise} noise points")
    else:
        print(f"\nRunning agglomerative clustering (k={n_clusters})...")
        labels = cluster_with_k(dist_matrix, n_clusters)
        n_clusters_found = n_clusters
        print(f"Created {n_clusters_found} clusters")

    # Summarize clusters
    df_with_clusters, cluster_summary = summarize_clusters(df, labels, option_names, X)

    # Save project assignments
    assignments_out = output_dir / f"{tech_safe}_cluster_assignments_{method}.csv"
    df_with_clusters[["project", "cluster", "num_options", "num_files"]].to_csv(
        assignments_out, index=False
    )
    print(f"Wrote cluster assignments to: {assignments_out}")

    # Save cluster summary
    summary_rows = []
    for cluster in cluster_summary:
        cluster_id = cluster["cluster_id"]
        num_projects = cluster["num_projects"]
        top_opts_str = "; ".join(f"{opt}:{count}" for opt, count in cluster["top_options"])
        summary_rows.append({
            "cluster_id": cluster_id,
            "num_projects": num_projects,
            "top_options": top_opts_str
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id")
    summary_out = output_dir / f"{tech_safe}_cluster_summary_{method}.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Wrote cluster summary to: {summary_out}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"Clustering Summary for {args.technology} ({method})")
    print(f"{'='*80}")
    for cluster in cluster_summary:
        print(f"\nCluster {cluster['cluster_id']}: {cluster['num_projects']} projects")
        print(f"  Top options: {', '.join([opt for opt, _ in cluster['top_options'][:5]])}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    pca_out = output_dir / f"{tech_safe}_pca_{method}.png"
    n_projects = len(df)
    plot_pca_embedding(X, labels, pca_out, args.technology, cluster_summary,
                      method, n_projects, args.random_state)

    heatmap_out = output_dir / f"{tech_safe}_heatmap_{method}.png"
    plot_heatmap(X, labels, option_names, heatmap_out, args.technology)

    print(f"\n{'='*80}")
    print(f"{method.upper()} clustering complete!")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster projects by how they configure a specific technology"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Name of the directory of a company (e.g., 'netflix', 'disney')."
    )
    parser.add_argument(
        "--technology",
        required=True,
        help="Technology to analyze (e.g., 'docker-compose', 'nodejs', 'npm')"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing project JSON files. "
             "If not provided, defaults to data/{input}/latest_commit"
    )
    parser.add_argument(
        "--min-option-frequency",
        type=int,
        default=2,
        help="Minimum number of projects an option must appear in (default: 2)"
    )
    parser.add_argument(
        "--feature-mode",
        choices=["option_value", "option_only"],
        default="option_value",
        help="Feature type for clustering: 'option_value' uses option=value pairs, "
             "'option_only' uses option names only, ignoring values (default: option_value)"
    )
    parser.add_argument(
        "--method",
        choices=["agglomerative", "hdbscan", "all"],
        default="agglomerative",
        help="Clustering method (default: agglomerative). Use 'all' to run all methods."
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=5,
        help="Number of clusters for agglomerative clustering (default: 5)"
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="For HDBSCAN: minimum cluster size (default: 2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--k-range",
        type=str,
        default="2,20",
        help="Range of k values to sweep for agglomerative (format: 'min,max', default: '2,20')"
    )
    parser.add_argument(
        "--min-size-range",
        type=str,
        default="3,20",
        help="Range of min_cluster_size values to sweep for HDBSCAN (format: 'min,max', default: '3,20')"
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save clustering results. "
             "If not provided, defaults to data/{input}/clustering/technologies"
    )

    args = parser.parse_args()

    # Set default paths based on input if not provided
    if args.data_dir is None:
        args.data_dir = f"../../data/{args.input}/latest_commit"
    if args.output_dir is None:
        args.output_dir = f"../../data/{args.input}/clustering/technologies"

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load project configurations
    df = load_project_configs(args.data_dir, args.technology)

    if len(df) == 0:
        print(f"No projects found using technology: {args.technology}")
        return

    # Build configuration matrix
    X, option_names = build_configuration_matrix(df, args.min_option_frequency,
                                                  args.feature_mode)

    # Compute distance matrix
    dist_matrix = compute_jaccard_distance(X)
    print("Computed Jaccard distance matrix")

    tech_safe = args.technology.replace("/", "_").replace(" ", "_")

    # Handle "all" method - run all clustering methods
    if args.method == "all":
        methods = ["agglomerative", "hdbscan"]
        for method in methods:
            if method == "hdbscan" and not HDBSCAN_AVAILABLE:
                print(f"\nSkipping {method}: hdbscan not installed")
                continue
            run_clustering_for_method(method, df, X, option_names, dist_matrix,
                                     tech_safe, output_dir, args)

        print(f"\n{'='*80}")
        print(f"All clustering methods complete! Results saved to: {output_dir}")
        print(f"{'='*80}")
        return

    # Run single method
    run_clustering_for_method(args.method, df, X, option_names, dist_matrix,
                             tech_safe, output_dir, args)

    print(f"\n{'='*80}")
    print(f"Clustering complete! Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
