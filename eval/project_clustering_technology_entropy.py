import pandas as pd
import numpy as np
from math import log

def calculate_entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def main():
    assign = pd.read_csv("../data/project_clustering_technologies/ecosystems_project_assignments.csv")
    tech = pd.read_csv("../data/project_clustering_technologies/ecosystems_tech_matrix.csv")

    df = pd.merge(assign, tech, on="project")

    tech_cols = [c for c in df.columns if c not in ("project", "ecosystem")]

    ecosystem_entropies = []

    for ecosys, group in df.groupby("ecosystem"):
        tech_usage = group[tech_cols].sum()
        H = calculate_entropy(tech_usage)
        H_norm = H / log(len(tech_cols))  # optional normalization
        ecosystem_entropies.append((ecosys, H, H_norm))

    entropy_df = pd.DataFrame(ecosystem_entropies,
                            columns=["ecosystem", "entropy", "entropy_normalized"])

    print(entropy_df)

    entropy_df = entropy_df.sort_values(by="entropy", ascending=False)
    entropy_df.to_csv("../data/project_clustering_technologies/ecosystems_entropy.csv", index=False)


if __name__ == "__main__":
    main()