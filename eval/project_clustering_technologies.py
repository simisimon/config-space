import argparse
import ast
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


def load_projects(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "technologies" not in df.columns or "project" not in df.columns:
        raise ValueError("CSV must contain 'project' and 'technologies' columns.")
    # Parse stringified lists into Python lists
    df["tech_list"] = df["technologies"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else []
    )
    return df


def build_technology_matrix(df: pd.DataFrame, min_frequency: int = 5):
    """
    Build a project × technology binary matrix.

    min_frequency:
        drop technologies that appear in fewer than this number of projects
        to reduce noise and dimensionality.
    """
    all_techs = Counter()
    for techs in df["tech_list"]:
        all_techs.update(set(techs))  # set() → avoid double-counting per project

    kept_techs = sorted([t for t, c in all_techs.items() if c >= min_frequency])
    tech_index = {t: i for i, t in enumerate(kept_techs)}

    X = np.zeros((len(df), len(kept_techs)), dtype=int)
    for row_idx, techs in enumerate(df["tech_list"]):
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


def summarize_clusters(df: pd.DataFrame, labels, top_n: int = 10):
    """
    Attach cluster labels to the dataframe and compute per-cluster top technologies.
    """
    df = df.copy()
    df["ecosystem"] = labels

    cluster_summary = []
    for cluster_id, group in df.groupby("ecosystem"):
        tech_counter = Counter()
        for techs in group["tech_list"]:
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


def plot_embedding(X: np.ndarray,
                   labels: np.ndarray,
                   output_path: str,
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

    unique_clusters = np.unique(labels)

    plt.figure(figsize=(8, 6))
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=f"ecosystem {cluster_id}",
            alpha=0.7,
            s=20,
        )

    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("Configuration ecosystems (PCA embedding)")
    plt.legend(loc="best", fontsize="small", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
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
        "--k-min",
        type=int,
        default=2,
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

    X, tech_names = build_technology_matrix(df, min_frequency=args.min_tech_frequency)
    print(
        f"Technology matrix: {X.shape[0]} projects × {X.shape[1]} technologies "
        f"(min frequency ≥ {args.min_tech_frequency})."
    )

    dist_matrix = compute_jaccard_distance(X)
    print("Computed Jaccard distance matrix.")

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
    stability_out = "../data/project_clustering_technologies/ecosystems_stability.csv"
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
    df_with_clusters, cluster_summary = summarize_clusters(df, best_labels)

    proj_out =  "../data/project_clustering_technologies/ecosystems_project_assignments.csv"
    df_with_clusters[["project", "ecosystem"]].to_csv(proj_out, index=False)
    print(f"Wrote project-level ecosystem assignments to: {proj_out}")

    summary_rows = []
    for cluster in cluster_summary:
        cluster_id = cluster["cluster_id"]
        num_projects = cluster["num_projects"]
        top_tech_str = "; ".join(
            f"{tech}:{count}" for tech, count in cluster["top_technologies"]
        )
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "num_projects": num_projects,
                "top_technologies": top_tech_str,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id")
    summary_out = "../data/project_clustering_technologies/ecosystems_cluster_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Wrote cluster summary to: {summary_out}")

    # Plot data
    embedding_out = "../data/project_clustering_technologies/ecosystems_embedding.png"
    plot_embedding(X, best_labels, embedding_out, random_state=args.random_state)


if __name__ == "__main__":
    main()
