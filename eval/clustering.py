import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    def parse_list(s):
        items = ast.literal_eval(s)
        items = [str(x).strip().lower() for x in items if str(x).strip()]
        return sorted(set(items))  # dedup
    df["technologies"] = df["technologies"].apply(parse_list)
    return df

def find_optimal_clusters(X, min_k=2, max_k=15, random_state=42):
    """
    Returns a best_k chosen by silhouette (when defined),
    and also plots elbow + silhouette curves.

    Rules:
    - max_k is capped at n_samples - 1
    - silhouette computed only if 2 <= n_unique_labels <= n_samples - 1
    """
    n_samples = X.shape[0]
    if n_samples < 3:
        # Cannot cluster meaningfully
        print("Not enough samples to select k. Using k=2.")
        return 2

    # Cap max_k so we never ask for k >= n_samples
    max_k_effective = max(min(max_k, n_samples - 1), min_k)
    ks = list(range(min_k, max_k_effective + 1))

    inertias = []
    silhouettes = []

    for k in ks:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)

        # Compute silhouette only if it's defined
        n_unique = len(np.unique(labels))
        if 2 <= n_unique <= n_samples - 1:
            silhouettes.append(silhouette_score(X, labels))
        else:
            silhouettes.append(np.nan)

    # Plot inertia (Elbow)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Method: Inertia vs #Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.show()

    # Plot silhouette
    plt.figure(figsize=(6, 4))
    plt.plot(ks, silhouettes, marker="o")
    plt.title("Silhouette Score vs #Clusters")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.show()

    # Choose best k by silhouette if any finite values exist; otherwise elbow fallback
    if np.isfinite(silhouettes).any():
        best_k = ks[int(np.nanargmax(silhouettes))]
        print(f"Suggested number of clusters (by silhouette): {best_k}")
    else:
        # Fallback: pick elbow-like k as the smallest k where relative inertia drop < threshold
        # Simple heuristic
        drops = np.diff(inertias) / np.array(inertias[:-1])
        thresh = -0.1  # less than 10% improvement -> stop
        elbow_idx = np.argmax(drops > thresh) if (drops > thresh).any() else 0
        best_k = ks[elbow_idx + 1] if elbow_idx + 1 < len(ks) else ks[-1]
        print(f"No valid silhouette; using elbow heuristic: {best_k}")

    return best_k

def cluster_project_technologies(df, output_file, top_tech_per_cluster=10, random_state=42):
    projects = df["project"].tolist()

    # One-Hot Encoding of technologies
    mlb = MultiLabelBinarizer()
    X_onehot = mlb.fit_transform(df["technologies"])
    tech_vocab = mlb.classes_
    onehot_df = pd.DataFrame(X_onehot, columns=tech_vocab, index=projects)

    # Find optimal number of clusters (optional)
    n_clusters = find_optimal_clusters(X_onehot, min_k=2, max_k=15)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(X_onehot)
    df_clusters = pd.DataFrame({"project": projects, "cluster": labels})

    # Print cluster summaries
    print("\n=== KMEANS CLUSTERS (One-Hot) ===")
    for c in range(n_clusters):
        members = df_clusters[df_clusters.cluster == c]["project"].tolist()
        print(f"\nCluster {c} | {len(members)} projects")
        print("Members:", ", ".join(members) if members else "(none)")
        mean_presence = onehot_df.loc[members].mean(axis=0) if members else pd.Series()
        top_feats = mean_presence.sort_values(ascending=False).head(top_tech_per_cluster)
        print("Top technologies (tech -> share of cluster):")
        for t, share in top_feats.items():
            print(f"  - {t:20s} -> {share:.2f}")

    # Plot visualization using PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_onehot)

    plt.figure(figsize=(8, 6))
    for c in range(n_clusters):
        mask = (labels == c)
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f"Cluster {c}")
    for i, name in enumerate(projects):
        plt.annotate(name, (X_2d[i, 0], X_2d[i, 1]), fontsize=8, alpha=0.7)
    plt.title("KMeans Clusters (One-Hot, PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../data/results/project_technology_clusters.png")

    df_clusters.to_csv(output_file, index=False)

if __name__ == "__main__":
    data_file = "../data/results/project_technologies.csv"
    technology_cluster_output_file = "../data/results/project_technology_clusters.csv"
    df = load_data(data_file)

    # Cluster projects by their technology stacks
    df_cluster = cluster_project_technologies(df=df, output_file=technology_cluster_output_file)

    # Cluster proejects with similar technology stacks by their option-value configurations
    # TODO