# Configuration-Based Project Clustering

## Overview

`cluster_technology_stack_config.py` clusters projects within a technology ecosystem based on their configuration option-value pairs. This enables discovery of configuration profile patterns among projects that share the same technology stack.

## Approach

### 1. Feature Extraction

Each project is represented as a set of configuration features extracted from JSON files. Features are constructed as:

- **With values**: `concept::option::value` (e.g., `maven::compiler.source::17`)
- **Without values**: `concept::option` (e.g., `maven::compiler.source`)

Certain value types (PASSWORD, UNKNOWN, COMMAND, EMAIL, IMAGE) are always treated as valueless to avoid noise and sensitive data.

### 2. Feature Matrix Construction

A binary project × feature matrix is built where entry (i, j) = 1 if project i uses feature j. Features appearing in fewer than `--min-feature-frequency` projects are filtered out.

### 3. Distance Metric

Projects are compared using **Jaccard distance** on their binary feature vectors:

```
d(A, B) = 1 - |A ∩ B| / |A ∪ B|
```

### 4. Clustering Algorithm

**Agglomerative hierarchical clustering** with average linkage on the precomputed Jaccard distance matrix.

### 5. Cluster Count Selection

If `--n-clusters` is not specified, the script performs **stability-based k selection**:

1. For each candidate k in [k_min, k_max]:
   - Cluster the full dataset
   - Repeatedly subsample (default 80%) and recluster
   - Compute Adjusted Rand Index (ARI) between subsample and full-data labels
2. Select k with highest median ARI across subsampling runs

## Usage

```bash
# Basic usage with automatic k selection
python cluster_configs.py --ecosystem 6

# Specify number of clusters
python cluster_configs.py --ecosystem 6 --n-clusters 5

# Filter to specific technologies
python cluster_configs.py --ecosystem 6 --concepts maven,docker

# Cluster by option presence only (ignore values)
python cluster_configs.py --ecosystem 6 --no-values
```

## Output Files

| File | Description |
|------|-------------|
| `{ecosystem}_config_matrix.csv` | Binary project × feature matrix |
| `{ecosystem}_config_stability.csv` | Stability metrics per k (if auto-selecting) |
| `{ecosystem}_config_project_clusters.csv` | Project → cluster assignments |
| `{ecosystem}_config_cluster_summary.csv` | Cluster summaries with top features |
| `{ecosystem}_config_embedding.png` | 2D PCA visualization |
