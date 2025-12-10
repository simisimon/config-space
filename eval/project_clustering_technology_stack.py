import argparse
import ast
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx.algorithms.community as nx_community
import seaborn as sns

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


def load_projects(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "technologies" not in df.columns or "project" not in df.columns:
        raise ValueError("CSV must contain 'project' and 'technologies' columns.")
    # Parse stringified lists into Python lists
    df["tech_list"] = df["technologies"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else []
    )
    return df


def build_technology_matrix(df: pd.DataFrame, min_frequency: int = 5, exclude_github: bool = False):
    """
    Build a project × technology binary matrix.

    min_frequency:
        drop technologies that appear in fewer than this number of projects
        to reduce noise and dimensionality.
    exclude_github:
        if True, exclude GitHub-centric technologies (GitHub Actions, Dependabot, etc.)
    """
    # Define GitHub-centric technologies to exclude
    github_techs = {
        'github-action',
        'github',
        'github config',
        'github issues',
        'github funding',
        'dependabot',
    }

    all_techs = Counter()
    for techs in df["tech_list"]:
        # Filter out GitHub-centric techs if requested
        if exclude_github:
            techs = [t for t in techs if t not in github_techs]
        all_techs.update(set(techs))  # set() → avoid double-counting per project

    kept_techs = sorted([t for t, c in all_techs.items() if c >= min_frequency])
    tech_index = {t: i for i, t in enumerate(kept_techs)}

    X = np.zeros((len(df), len(kept_techs)), dtype=int)
    for row_idx, techs in enumerate(df["tech_list"]):
        # Filter out GitHub-centric techs if requested
        if exclude_github:
            techs = [t for t in techs if t not in github_techs]
        for t in set(techs):
            if t in tech_index:
                X[row_idx, tech_index[t]] = 1

    return X, kept_techs


def compute_jaccard_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute an n×n Jaccard distance matrix between projects.
    """
    if X.shape[1] == 0:
        raise ValueError(
            "No technologies left after filtering; try lowering --min-tech-frequency."
        )
    dist = pairwise_distances(X, metric="jaccard")
    return dist


def cluster_with_k(dist_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Run agglomerative clustering with a precomputed distance matrix.
    """
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="precomputed", linkage="average"
        )
    except TypeError:
        # Newer sklearn versions use 'metric' instead of 'affinity'
        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
    labels = model.fit_predict(dist_matrix)
    return labels


def cluster_with_hdbscan(dist_matrix: np.ndarray, min_cluster_size: int = 5, min_samples: int = 3) -> np.ndarray:
    """
    Run HDBSCAN clustering with a precomputed distance matrix.

    min_cluster_size: minimum number of samples in a cluster
    min_samples: minimum number of samples in a neighborhood for a point to be considered core

    Returns labels where -1 indicates noise/outliers.
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan is not installed. Install with: pip install hdbscan")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_method='eom'  # Excess of Mass
    )
    labels = clusterer.fit_predict(dist_matrix)
    return labels


def cluster_with_louvain(dist_matrix: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    """
    Run Louvain community detection by converting distance matrix to similarity graph.

    resolution: higher values lead to more communities

    Returns cluster labels.
    """
    if not IGRAPH_AVAILABLE:
        raise ImportError("igraph is not installed. Install with: pip install igraph")

    # Convert distance to similarity (1 - distance)
    # Use a threshold to create edges only for similar projects
    similarity = 1 - dist_matrix

    # Create weighted graph from similarity matrix
    # Keep only edges above a threshold to avoid fully connected graph
    threshold = np.percentile(similarity[np.triu_indices_from(similarity, k=1)], 50)

    edges = []
    weights = []
    n = similarity.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] > threshold:
                edges.append((i, j))
                weights.append(similarity[i, j])

    # Create graph
    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es['weight'] = weights

    # Run Louvain community detection
    communities = g.community_multilevel(weights='weight', return_levels=False, resolution=resolution)

    labels = np.array(communities.membership)
    return labels


def evaluate_resolution_sweep(dist_matrix: np.ndarray,
                              X: np.ndarray,
                              resolution_values: list,
                              n_stability_runs: int = 10) -> pd.DataFrame:
    """
    Sweep over resolution values for Louvain clustering and evaluate metrics.

    Returns DataFrame with columns: resolution, n_clusters, silhouette_score,
                                     modularity, stability_mean, stability_std
    """
    if not IGRAPH_AVAILABLE:
        raise ImportError("igraph is not installed. Install with: pip install igraph")

    results = []

    for res in resolution_values:
        print(f"\nEvaluating resolution={res}...")

        # Get clustering
        labels = cluster_with_louvain(dist_matrix, resolution=res)
        n_clusters = len(set(labels))

        # Silhouette score (using embedding space)
        # Filter out clusters with only 1 member for silhouette calculation
        if n_clusters > 1 and min(np.bincount(labels)) > 1:
            sil_score = silhouette_score(X, labels, metric='jaccard')
        else:
            sil_score = -1.0

        # Modularity (need to reconstruct graph)
        similarity = 1 - dist_matrix
        threshold = np.percentile(similarity[np.triu_indices_from(similarity, k=1)], 50)

        # Create NetworkX graph for modularity calculation
        import networkx as nx
        G = nx.Graph()
        n = similarity.shape[0]
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] > threshold:
                    G.add_edge(i, j, weight=similarity[i, j])

        # Calculate modularity
        communities = {}
        for node, comm in enumerate(labels):
            communities.setdefault(comm, []).append(node)

        modularity = nx_community.modularity(G, communities.values(), weight='weight')

        # Stability via perturbation
        stability_scores = []
        for run in range(n_stability_runs):
            # Run with different random initialization
            labels_perturbed = cluster_with_louvain(dist_matrix, resolution=res)
            ari = adjusted_rand_score(labels, labels_perturbed)
            stability_scores.append(ari)

        stability_mean = np.mean(stability_scores)
        stability_std = np.std(stability_scores)

        results.append({
            'resolution': res,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'modularity': modularity,
            'stability_mean': stability_mean,
            'stability_std': stability_std
        })

        print(f"  Clusters: {n_clusters}, Silhouette: {sil_score:.3f}, "
              f"Modularity: {modularity:.3f}, Stability: {stability_mean:.3f}±{stability_std:.3f}")

    return pd.DataFrame(results)


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


def summarize_clusters(df: pd.DataFrame, labels, top_n: int = 10, exclude_github: bool = False):
    """
    Attach cluster labels to the dataframe and compute per-cluster top technologies.

    exclude_github: if True, exclude GitHub-centric technologies from summary
    """
    # Define GitHub-centric technologies to exclude
    github_techs = {
        'github-action',
        'github',
        'github config',
        'github issues',
        'github funding',
        'dependabot',
    }

    df = df.copy()
    df["ecosystem"] = labels

    cluster_summary = []
    for cluster_id, group in df.groupby("ecosystem"):
        tech_counter = Counter()
        for techs in group["tech_list"]:
            # Filter out GitHub-centric techs if requested
            if exclude_github:
                techs = [t for t in techs if t not in github_techs]
            tech_counter.update(set(techs))

        total_projects = len(group)
        top_techs = tech_counter.most_common(top_n)
        cluster_summary.append(
            {
                "cluster_id": int(cluster_id),
                "num_projects": int(total_projects),
                "top_technologies": top_techs,
            }
        )

    return df, cluster_summary


def plot_tsne_embedding(X: np.ndarray,
                        labels: np.ndarray,
                        output_path: str,
                        method: str = "",
                        cluster_summary=None,
                        random_state: int = 42):
    """
    2D t-SNE embedding of projects colored by ecosystem.
    t-SNE better preserves local structure than PCA.
    """
    if X.shape[1] < 2:
        print("Skipping t-SNE plot: fewer than 2 technology dimensions.")
        return

    print("Computing t-SNE embedding (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, X.shape[0]-1))
    X_2d = tsne.fit_transform(X)

    unique_clusters = np.unique(labels)

    plt.figure(figsize=(10, 8))
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        n_points = mask.sum()

        legend_label = f"ecosystem {cluster_id} (n={n_points})"
        if cluster_summary is not None:
            cluster_info = next((c for c in cluster_summary if c["cluster_id"] == cluster_id), None)
            if cluster_info:
                top_techs = cluster_info["top_technologies"][:3]
                if top_techs:
                    tech_label = ", ".join([tech for tech, count in top_techs])
                    legend_label = f"ecosystem {cluster_id} (n={n_points}, {tech_label})"

        X_cluster = X_2d[mask]
        plt.scatter(
            X_cluster[:, 0],
            X_cluster[:, 1],
            label=legend_label,
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5,
        )

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title(f"Technology Ecosystems - t-SNE (Method: {method})")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize="small", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote t-SNE embedding plot to: {output_path}")


def plot_heatmap(X: np.ndarray,
                 labels: np.ndarray,
                 tech_names: list,
                 output_path: str,
                 method: str = ""):
    """
    Heatmap showing technology usage patterns by ecosystem.
    Shows which technologies define each cluster.
    """
    # Calculate technology frequency per cluster
    unique_clusters = np.unique(labels)
    cluster_tech_freq = []

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_projects = X[mask]
        # Calculate percentage of projects in cluster using each tech
        tech_freq = (cluster_projects.sum(axis=0) / mask.sum()) * 100
        cluster_tech_freq.append(tech_freq)

    cluster_tech_matrix = np.array(cluster_tech_freq)

    # Only show top technologies (those used by at least 20% in some cluster)
    max_usage = cluster_tech_matrix.max(axis=0)
    important_techs = max_usage >= 20

    if important_techs.sum() == 0:
        # If no tech meets threshold, show top 20
        top_indices = np.argsort(max_usage)[-20:]
        important_techs = np.zeros(len(tech_names), dtype=bool)
        important_techs[top_indices] = True

    filtered_matrix = cluster_tech_matrix[:, important_techs]
    filtered_names = [tech_names[i] for i in np.where(important_techs)[0]]

    # Create heatmap
    plt.figure(figsize=(12, max(6, len(unique_clusters) * 0.5)))
    sns.heatmap(
        filtered_matrix,
        xticklabels=filtered_names,
        yticklabels=[f"Ecosystem {c}" for c in unique_clusters],
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        cbar_kws={'label': '% of projects'},
        linewidths=0.5
    )
    plt.xlabel("Technology")
    plt.ylabel("Ecosystem")
    plt.title(f"Technology Usage by Ecosystem (Method: {method})")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote heatmap to: {output_path}")


def plot_cluster_size_distribution(labels: np.ndarray,
                                   output_path: str,
                                   method: str = ""):
    """
    Bar chart showing the size distribution of clusters.
    """
    unique_clusters, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique_clusters, counts, edgecolor='black', linewidth=0.5)

    # Color bars
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel("Ecosystem ID")
    plt.ylabel("Number of Projects")
    plt.title(f"Ecosystem Size Distribution (Method: {method})")
    plt.xticks(unique_clusters)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for cluster_id, count in zip(unique_clusters, counts):
        plt.text(cluster_id, count, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote size distribution plot to: {output_path}")


def plot_embedding(X: np.ndarray,
                   labels: np.ndarray,
                   output_path: str,
                   method: str = "",
                   cluster_summary=None,
                   random_state: int = 42):
    """
    2D PCA embedding of projects colored by ecosystem.
    """
    if X.shape[1] < 2:
        # Not enough dimensions for PCA to be meaningful
        print("Skipping embedding plot: fewer than 2 technology dimensions.")
        return

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X)

    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_
    total_var = explained_var.sum() * 100

    unique_clusters = np.unique(labels)

    plt.figure(figsize=(10, 8))
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        n_points = mask.sum()

        # Build legend label with technologies in brackets
        legend_label = f"ecosystem {cluster_id} (n={n_points})"
        if cluster_summary is not None:
            cluster_info = next((c for c in cluster_summary if c["cluster_id"] == cluster_id), None)
            if cluster_info:
                top_techs = cluster_info["top_technologies"][:3]  # Top 3 technologies
                if top_techs:  # Only add tech label if there are technologies
                    tech_label = ", ".join([tech for tech, count in top_techs])
                    legend_label = f"ecosystem {cluster_id} (n={n_points}, {tech_label})"

        # Add jitter to make overlapping points visible
        X_cluster = X_2d[mask].copy()
        if n_points > 1:
            # Add small random jitter to spread overlapping points
            jitter_amount = 0.02  # Adjust this to control spread
            # Use abs to handle negative cluster_id (e.g., -1 for noise in HDBSCAN)
            seed = abs(int(cluster_id)) + 1000  # Add offset to avoid seed=0 issues
            X_cluster += np.random.RandomState(seed).normal(0, jitter_amount, X_cluster.shape)

        plt.scatter(
            X_cluster[:, 0],
            X_cluster[:, 1],
            label=legend_label,
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5,
        )

    plt.xlabel(f"PCA component 1 ({explained_var[0]*100:.1f}% var)")
    plt.ylabel(f"PCA component 2 ({explained_var[1]*100:.1f}% var)")
    plt.title(f"Technology Ecosystems (Method: {method}, Projects: {X.shape[0]})")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize="small", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Wrote PCA embedding plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster projects into configuration ecosystems "
                    "with stability-based k selection."
    )
    parser.add_argument(
        "--csv_path",
        default="../data/technology_composition/project_technologies_filtered.csv",
        help="Path to project_technologies_filtered.csv "
             "(must have 'project' and 'technologies' columns).",
    )
    parser.add_argument(
        "--min-tech-frequency",
        type=int,
        default=2,
        help="Minimum number of projects a technology must appear in "
             "to be kept (default: 5).",
    )
    parser.add_argument(
        "--exclude-github",
        action="store_true",
        help="Exclude GitHub-centric technologies (github-action, dependabot, renovate, etc.) "
             "from the clustering analysis.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=3,
        help="Minimum number of clusters to consider (default: 4).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=30,
        help="Maximum number of clusters to consider (default: 10).",
    )
    parser.add_argument(
        "--subsample-fraction",
        type=float,
        default=0.8,
        help="Fraction of projects to sample for each stability run "
             "(default: 0.8).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="Number of subsampling runs per k to estimate stability "
             "(default: 20).",
    )
    parser.add_argument(
        "--output-prefix",
        default="ecosystems",
        help="Prefix for output files (default: 'ecosystems').",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--method",
        choices=["agglomerative", "hdbscan", "louvain", "all"],
        default="agglomerative",
        help="Clustering method to use. Use 'all' to run all available methods (default: agglomerative).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="For HDBSCAN: minimum samples in a cluster (default: 5).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="For HDBSCAN: minimum samples in neighborhood (default: 3).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="For Louvain: resolution parameter, higher = more communities (default: 1.0).",
    )
    parser.add_argument(
        "--sweep-resolution",
        action="store_true",
        help="For Louvain: sweep over multiple resolution values to find optimal (overrides --resolution).",
    )
    parser.add_argument(
        "--resolution-values",
        type=str,
        default="0.8,0.9,1.0,1.1,1.2,1.3",
        help="For Louvain sweep: comma-separated resolution values (default: 0.5,0.7,1.0,1.3,1.5,2.0,2.5).",
    )
    parser.add_argument(
        "--n-stability-runs",
        type=int,
        default=10,
        help="For Louvain sweep: number of stability runs per resolution (default: 10).",
    )

    parser.add_argument(
        "--assignments",
        default="ecosystems_project_assignments.csv",
        help="Path to project assignments CSV (default: ecosystems_project_assignments.csv)",
    )
    parser.add_argument(
        "--tech-matrix",
        default="ecosystems_tech_matrix.csv",
        help="Path to project × technology matrix CSV (default: ecosystems_tech_matrix.csv)",
    )
    parser.add_argument(
        "--output",
        default="ecosystems_embedding.png",
        help="Output image file (default: ecosystems_embedding.png)",
    )

    args = parser.parse_args()

    df = load_projects(args.csv_path)
    print(f"Loaded {len(df)} projects from {args.csv_path}")

    X, tech_names = build_technology_matrix(df, min_frequency=args.min_tech_frequency, exclude_github=args.exclude_github)

    github_status = " (excluding GitHub-centric technologies)" if args.exclude_github else ""
    print(
        f"Technology matrix: {X.shape[0]} projects × {X.shape[1]} technologies "
        f"(min frequency ≥ {args.min_tech_frequency}){github_status}"
    )

    dist_matrix = compute_jaccard_distance(X)
    print("Computed Jaccard distance matrix.")

    # Determine which methods to run
    if args.method == "all":
        methods_to_run = ["agglomerative", "louvain"]
        # Add HDBSCAN if available
        if HDBSCAN_AVAILABLE:
            methods_to_run.append("hdbscan")
        else:
            print("\nNote: HDBSCAN not available (install with: pip install hdbscan)")
        # Check if Louvain is available
        if not IGRAPH_AVAILABLE:
            print("\nNote: Louvain not available (install with: pip install igraph)")
            methods_to_run.remove("louvain")
    else:
        methods_to_run = [args.method]

    # Run clustering for each method
    for current_method in methods_to_run:
        print(f"\n{'='*80}")
        print(f"Running clustering method: {current_method.upper()}")
        print(f"{'='*80}")

        # Save project × technology matrix for this method
        tech_mat_out = f"../data/project_clustering_technology_stack/ecosystems_tech_matrix_{current_method}.csv"
        tech_df = pd.DataFrame(X, columns=tech_names)
        tech_df.insert(0, "project", df["project"].values)
        tech_df.to_csv(tech_mat_out, index=False)

        # Select clustering method
        if current_method == "hdbscan":
            print(f"\nRunning HDBSCAN clustering (min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples})...")
            best_labels = cluster_with_hdbscan(dist_matrix, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
            n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
            n_noise = list(best_labels).count(-1)
            print(f"Found {n_clusters} clusters and {n_noise} noise points")

        elif current_method == "louvain":
            if args.sweep_resolution:
                # Perform resolution parameter sweep
                resolution_values = [float(x.strip()) for x in args.resolution_values.split(',')]
                print(f"\nSweeping Louvain resolution parameter over values: {resolution_values}")

                sweep_results = evaluate_resolution_sweep(
                    dist_matrix,
                    X,
                    resolution_values,
                    n_stability_runs=args.n_stability_runs
                )

                # Save sweep results
                sweep_out = f"../data/project_clustering_technology_stack/ecosystems_resolution_sweep_{current_method}.csv"
                sweep_results.to_csv(sweep_out, index=False)
                print(f"\nWrote resolution sweep results to: {sweep_out}")

                # Select best resolution based on combined criteria
                # Normalize each metric to [0, 1] for comparison
                sil_range = sweep_results['silhouette_score'].max() - sweep_results['silhouette_score'].min()
                mod_range = sweep_results['modularity'].max() - sweep_results['modularity'].min()
                stab_range = sweep_results['stability_mean'].max() - sweep_results['stability_mean'].min()

                # Only normalize if there's variation, otherwise set to 0.5
                if sil_range > 1e-6:
                    sweep_results['silhouette_norm'] = (sweep_results['silhouette_score'] - sweep_results['silhouette_score'].min()) / sil_range
                else:
                    sweep_results['silhouette_norm'] = 0.5

                if mod_range > 1e-6:
                    sweep_results['modularity_norm'] = (sweep_results['modularity'] - sweep_results['modularity'].min()) / mod_range
                else:
                    sweep_results['modularity_norm'] = 0.5

                if stab_range > 1e-6:
                    sweep_results['stability_norm'] = (sweep_results['stability_mean'] - sweep_results['stability_mean'].min()) / stab_range
                else:
                    sweep_results['stability_norm'] = 0.5

                # Combined score: equal weight to silhouette, modularity, and stability
                sweep_results['combined_score'] = (sweep_results['silhouette_norm'] +
                                                   sweep_results['modularity_norm'] +
                                                   sweep_results['stability_norm']) / 3.0

                # If all scores are equal, just pick the first resolution
                if sweep_results['combined_score'].isna().all():
                    print("\nWarning: All combined scores are NaN, using first resolution value")
                    best_resolution = resolution_values[0]
                    best_row = sweep_results.iloc[0]
                else:
                    best_row = sweep_results.loc[sweep_results['combined_score'].idxmax()]
                    best_resolution = best_row['resolution']

                print(f"\nSelected best resolution={best_resolution} based on combined score:")
                print(f"  Clusters: {best_row['n_clusters']:.0f}")
                print(f"  Silhouette: {best_row['silhouette_score']:.3f}")
                print(f"  Modularity: {best_row['modularity']:.3f}")
                print(f"  Stability: {best_row['stability_mean']:.3f}±{best_row['stability_std']:.3f}")
                print(f"  Combined score: {best_row['combined_score']:.3f}")

                # Cluster with best resolution
                best_labels = cluster_with_louvain(dist_matrix, resolution=best_resolution)
                n_clusters = len(set(best_labels))
            else:
                # Use single resolution value
                print(f"\nRunning Louvain community detection (resolution={args.resolution})...")
                best_labels = cluster_with_louvain(dist_matrix, resolution=args.resolution)
                n_clusters = len(set(best_labels))
                print(f"Found {n_clusters} communities")

        else:  # agglomerative
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

                stability_rows.append(
                    {
                        "k": k,
                        "mean_ari": mean_ari,
                        "median_ari": median_ari,
                        "std_ari": std_ari,
                        "n_repeats_effective": len(aris),
                    }
                )

                print(
                    f"  Stability for k={k}: mean ARI={mean_ari:.4f}, "
                    f"median ARI={median_ari:.4f}, std={std_ari:.4f} "
                    f"(over {len(aris)} runs)."
                )

            if not stability_rows:
                raise RuntimeError("No stability results computed; check your parameters.")

            stability_df = pd.DataFrame(stability_rows).sort_values("k")
            stability_out = f"../data/project_clustering_technology_stack/ecosystems_stability_{current_method}.csv"
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

            print(f"\nClustering full data with best k={best_k} ...")
            best_labels = cluster_with_k(dist_matrix, best_k)

        # Process results for this method
        df_with_clusters, cluster_summary = summarize_clusters(df, best_labels, exclude_github=args.exclude_github)

        proj_out = f"../data/project_clustering_technology_stack/ecosystems_project_assignments_{current_method}.csv"
        df_with_clusters[["project", "ecosystem"]].to_csv(proj_out, index=False)
        print(f"Wrote project-level ecosystem assignments to: {proj_out}")

        summary_rows = []
        for cluster in cluster_summary:
            cluster_id = cluster["cluster_id"]
            num_projects = cluster["num_projects"]
            top_tech_str = "; ".join(
                f"{tech}:{count}" for tech, count in cluster["top_technologies"]
            )

            # Warn if a cluster has no technologies (all were filtered out)
            if not top_tech_str and num_projects > 0:
                print(f"Warning: Ecosystem {cluster_id} has {num_projects} projects but no technologies "
                      f"(all may have been filtered out by --exclude-github)")

            summary_rows.append(
                {
                    "cluster_id": cluster_id,
                    "num_projects": num_projects,
                    "top_technologies": top_tech_str,
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id")
        summary_out = f"../data/project_clustering_technology_stack/ecosystems_cluster_summary_{current_method}.csv"
        summary_df.to_csv(summary_out, index=False)
        print(f"Wrote cluster summary to: {summary_out}")

        # Generate multiple visualizations
        print("\nGenerating visualizations...")

        # 1. PCA embedding
        embedding_out = f"../data/project_clustering_technology_stack/ecosystems_embedding_{current_method}.png"
        plot_embedding(
            X,
            best_labels,
            embedding_out,
            method=current_method,
            cluster_summary=cluster_summary,
            random_state=args.random_state
        )

        # 2. t-SNE embedding (better for local structure)
        tsne_out = f"../data/project_clustering_technology_stack/ecosystems_tsne_{current_method}.png"
        plot_tsne_embedding(
            X,
            best_labels,
            tsne_out,
            method=current_method,
            cluster_summary=cluster_summary,
            random_state=args.random_state
        )

        # 3. Technology usage heatmap
        heatmap_out = f"../data/project_clustering_technology_stack/ecosystems_heatmap_{current_method}.png"
        plot_heatmap(
            X,
            best_labels,
            tech_names,
            heatmap_out,
            method=current_method
        )

        # 4. Cluster size distribution
        size_out = f"../data/project_clustering_technology_stack/ecosystems_sizes_{current_method}.png"
        plot_cluster_size_distribution(
            best_labels,
            size_out,
            method=current_method
        )

        print(f"\nCompleted {current_method} clustering!")

    print(f"\n{'='*80}")
    print("All clustering methods completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
