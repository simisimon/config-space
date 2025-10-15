import ast
import os
import json
from math import ceil
from typing import List, Dict
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def load_csv_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    def parse_list(s):
        items = ast.literal_eval(s)
        items = [str(x).strip().lower() for x in items if str(x).strip()]
        return sorted(set(items))  # dedup
    df["technologies"] = df["technologies"].apply(parse_list)
    return df


def find_optimal_clusters(X, min_k=2, max_k=15, random_state=42):
    """
    Returns best_k chosen by silhouette (when defined) with elbow fallback.
    - Caps max_k at n_samples - 1.
    """
    n_samples = X.shape[0]
    if n_samples < 3:
        print("Not enough samples to select k. Using k=2.")
        return 2

    max_k_effective = max(min(max_k, n_samples - 1), min_k)
    ks = list(range(min_k, max_k_effective + 1))

    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        n_unique = len(np.unique(labels))
        if 2 <= n_unique <= n_samples - 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(np.nan)

    # Elbow
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Method: Inertia vs #Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Silhouette
    plt.figure(figsize=(6, 4))
    plt.plot(ks, silhouettes, marker="o")
    plt.title("Silhouette Score vs #Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.show()
    plt.close()

    if np.isfinite(silhouettes).any():
        best_k = ks[int(np.nanargmax(silhouettes))]
        print(f"Suggested number of clusters (by silhouette): {best_k}")
        return best_k

    drops = np.diff(inertias) / np.array(inertias[:-1])
    thresh = -0.1
    elbow_idx = np.argmax(drops > thresh) if (drops > thresh).any() else 0
    best_k = ks[elbow_idx + 1] if elbow_idx + 1 < len(ks) else ks[-1]
    print(f"No valid silhouette; using elbow heuristic: {best_k}")
    return best_k


def cluster_projects_by_technologies(df, top_tech_per_cluster=10, random_state=42):
    """
    KMeans over one-hot tech features.
    Saves PNG + CSV to ../data/results/clustering/.
    """
    out_dir = "../data/results/clustering"
    os.makedirs(out_dir, exist_ok=True)

    projects = df["project"].tolist()
    mlb = MultiLabelBinarizer()
    X_onehot = mlb.fit_transform(df["technologies"])
    tech_vocab = mlb.classes_
    onehot_df = pd.DataFrame(X_onehot, columns=tech_vocab, index=projects)

    n_clusters = find_optimal_clusters(X_onehot, min_k=2, max_k=15)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(X_onehot)
    df_clusters = pd.DataFrame({"project": projects, "cluster": labels})

    print("\n=== KMEANS CLUSTERS (One-Hot) ===")
    for c in range(n_clusters):
        members = df_clusters[df_clusters.cluster == c]["project"].tolist()
        print(f"\nCluster {c} | {len(members)} projects")
        print("Members:", ", ".join(members) if members else "(none)")
        mean_presence = onehot_df.loc[members].mean(axis=0) if members else pd.Series(dtype=float)
        top_feats = mean_presence.sort_values(ascending=False).head(top_tech_per_cluster)
        print("Top technologies (tech -> share of cluster):")
        for t, share in top_feats.items():
            print(f"  - {t:20s} -> {share:.2f}")

    # PCA scatter
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_onehot)

    plt.figure(figsize=(8, 6))
    for c in range(n_clusters):
        mask = (labels == c)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {c}")
    for i, name in enumerate(projects):
        plt.annotate(name, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)

    plt.title("KMeans Clusters (Tech One-Hot, PCA)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(out_dir, "projects_technology_cluster.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")

    csv_path = os.path.join(out_dir, "projects_clustered.csv")
    df_clusters.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    return df_clusters


def load_project_data(project_files: List[str]) -> List[dict]:
    """Load per-project JSONs and extract latest_commit_data."""
    projects_data = []
    for project_file in project_files:
        if not os.path.exists(project_file):
            continue
        with open(project_file, 'r', encoding="utf-8") as f:
            project_data = json.load(f)
        latest_commit = project_data.get('latest_commit_data', {})
        if latest_commit:
            projects_data.append({
                'project_name': project_data.get('project_name', os.path.basename(project_file).replace("_last_commit.json", "")),
                'latest_commit': latest_commit
            })
    return projects_data


def extract_features_from_projects(project_files: List[str]) -> pd.DataFrame:
    """
    Build a tidy DF with:
      project_name, concepts, option_value_pairs (list[str])
    """
    project_data = load_project_data(project_files)
    features = []

    for project in project_data:
        project_name = project['project_name']
        commit_data = project['latest_commit'].get('network_data', {})

        # tech concepts (kept for potential future use)
        concepts = set(map(lambda s: str(s).strip().lower(), commit_data.get('concepts', [])))

        # option=value pairs
        option_value_pairs = []
        for file_data in commit_data.get('config_file_data', []):
            for pair in file_data.get('pairs', []):
                opt = str(pair.get('option', '')).strip().lower()
                val = str(pair.get('value', '')).strip().lower()
                if opt and val:
                    option_value_pairs.append(f"{opt}={val}")

        features.append({
            'project_name': project_name,
            'concepts': sorted(concepts),
            'option_value_pairs': option_value_pairs,
        })

    return pd.DataFrame(features)


def cluster_projects_by_option_value_pairs(project_names: List[str], cluster_num: int,
                                           max_features: int = 500, random_state: int = 42):
    """
    TF-IDF over 'option=value' tokens -> choose k -> KMeans -> report/plot/CSV.
    Outputs to ../data/results/clustering/.
    """
    out_dir = "../data/results/clustering"
    os.makedirs(out_dir, exist_ok=True)

    # Resolve files
    project_files = [
        f"../data/projects_last_commit/{name}_last_commit.json"
        for name in project_names
    ]

    # Extract features
    df_features = extract_features_from_projects(project_files)

    # Build docs, filter out empties to avoid zero rows
    kept = df_features[df_features["option_value_pairs"].map(lambda lst: len(lst) > 0)].copy()
    if kept.empty or len(kept) < 2:
        print(f"[Cluster {cluster_num}] < 2 projects with option=value data; skipping.")
        # still emit a minimal CSV for traceability
        pd.DataFrame({"project": project_names, "cluster": [0]*len(project_names)}).to_csv(
            os.path.join(out_dir, f"project_cluster_{cluster_num}_assignments.csv"), index=False
        )
        return

    clean_names = kept["project_name"].tolist()
    project_texts = [" ".join(pairs) for pairs in kept["option_value_pairs"]]

    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=max_features,
        token_pattern=r"[^ ]+",
        lowercase=False,
        stop_words=None
    )
    tfidf_matrix = tfidf.fit_transform(project_texts)
    X = tfidf_matrix.toarray()        # small max_features; dense OK
    feature_names = tfidf.get_feature_names_out()

    # Choose k and fit
    n_clusters = find_optimal_clusters(X, min_k=2, max_k=15, random_state=random_state)
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(X)

    # Build results consistently (use kept names)
    clusters = defaultdict(list)
    for proj, lab in zip(clean_names, labels):
        clusters[int(lab)].append(proj)

    option_results = {
        'type': 'option_value',
        'features': X,
        'similarity_matrix': cosine_similarity(tfidf_matrix),
        'feature_names': feature_names,
        'kmeans': {
            'labels': labels,
            'projects': clean_names,
            'clusters': dict(clusters)
        }
    }

    # Overlaps & report
    option_overlaps = get_option_value_overlaps(kept, option_results)
    fig = visualize_clusters(option_results)
    fig_path = os.path.join(out_dir, f"project_cluster_{cluster_num}.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    report = generate_option_value_report(kept, option_results, option_overlaps)
    report_path = os.path.join(out_dir, f"project_cluster_{cluster_num}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    # Also save assignments CSV
    assign_path = os.path.join(out_dir, f"project_cluster_{cluster_num}_assignments.csv")
    pd.DataFrame({"project": clean_names, "cluster": labels}).to_csv(assign_path, index=False)
    print(f"Saved: {assign_path}")


def visualize_clusters(clustering_results: Dict):
    """Visualize clustering results (KMeans only)"""
    features = clustering_results['features']

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_features = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        pca_features[:, 0],
        pca_features[:, 1],
        c=clustering_results['kmeans']['labels'],
        cmap='viridis',
        s=100
    )
    ax.set_title('KMeans - Option=Value TF-IDF (PCA)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

    # Labels
    for i, project in enumerate(clustering_results['kmeans']['projects']):
        ax.annotate(
            project,
            (pca_features[i, 0], pca_features[i, 1]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )

    plt.tight_layout()
    return fig


def get_option_value_overlaps(features_df: pd.DataFrame, clustering_results: Dict,
                              min_support: float = 0.5, top_n: int = 10) -> Dict:
    """
    Compute frequent option=value pairs per cluster.
    - min_support: fraction of projects in the cluster (default 0.5).
    """
    overlaps = {}
    method_overlaps = {}
    clusters = clustering_results['kmeans']['clusters']

    for cluster_id, projects in clusters.items():
        cluster_projects = features_df[features_df['project_name'].isin(projects)]
        all_pairs = []
        for _, project in cluster_projects.iterrows():
            all_pairs.extend(project['option_value_pairs'])

        pair_counts = Counter(all_pairs)
        # >= ceil(min_support * size) for clarity
        threshold = max(1, ceil(min_support * max(1, len(projects))))
        frequent_pairs = {pair: cnt for pair, cnt in pair_counts.items() if cnt >= threshold}

        method_overlaps[cluster_id] = {
            'projects': projects,
            'frequent_option_values': dict(sorted(frequent_pairs.items(), key=lambda x: (-x[1], x[0]))[:top_n]),
            'all_option_values': dict(pair_counts.most_common(top_n)),
            'size': len(projects)
        }

    overlaps['kmeans'] = method_overlaps
    return overlaps


def generate_option_value_report(df_features: pd.DataFrame, clustering_results: Dict, overlaps: Dict) -> str:
    """Generate a human-readable option=value clustering report."""
    report = []
    report.append("Option-Value Configuration Clustering Report")
    report.append("=" * 50 + "\n")
    report.append("KMEANS Option-Value Clustering Results:")
    report.append("-" * 45)

    clusters = clustering_results["kmeans"]['clusters']
    method_overlaps = overlaps["kmeans"]

    for cluster_id, projects in clusters.items():
        overlap_info = method_overlaps.get(cluster_id, {})
        report.append(f"\nCluster {cluster_id} ({len(projects)} projects):")
        report.append(f"  Projects: {', '.join(projects)}")

        frequent_pairs = overlap_info.get('frequent_option_values', {})
        if frequent_pairs:
            report.append("  Frequent Option-Value Pairs:")
            for pair, count in frequent_pairs.items():
                report.append(f"    - {pair} (appears in {count} projects)")

        all_pairs = overlap_info.get('all_option_values', {})
        if all_pairs:
            report.append("  Top Option-Value Pairs:")
            shown = 0
            for pair, count in all_pairs.items():
                report.append(f"    - {pair} ({count} times)")
                shown += 1
                if shown >= 3:
                    break

    report.append("")
    return "\n".join(report)


if __name__ == "__main__":
    data_file = "../data/results/project_technologies.csv"
    df = load_csv_data(data_file)

    # Cluster by technologies
    df_cluster = cluster_projects_by_technologies(df=df)

    # For each tech-cluster, cluster by option=value pairs
    clustered_projects = [
        (cluster, group["project"].tolist())
        for cluster, group in df_cluster.groupby("cluster")
    ]

    for cluster_num, projects in clustered_projects:
        cluster_projects_by_option_value_pairs(
            project_names=projects,
            cluster_num=cluster_num,
            max_features=500,
            random_state=42
        )
