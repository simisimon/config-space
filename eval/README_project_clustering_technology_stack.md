# Project Clustering by Technology Stack

This document describes the clustering approaches available for grouping projects into technology ecosystems based on their technology stack composition.

## Overview

The `project_clustering_technology_stack.py` script clusters projects based on their technology usage patterns. Projects are represented as binary vectors (presence/absence of technologies), and Jaccard distance measures the dissimilarity between projects. The script supports multiple clustering methods, automatic parameter selection, and generates comprehensive visualizations for analysis.

## Available Clustering Methods

### 1. Agglomerative Clustering (Default)

**Type**: Hierarchical clustering with stability-based k selection

**How it works**:
- Builds a hierarchy of clusters using average linkage
- Evaluates multiple values of k (number of clusters)
- Uses subsampling + Adjusted Rand Index (ARI) to measure stability
- Selects k with the highest median stability score

**Parameters**:
- `--k-min`: Minimum number of clusters to evaluate (default: 3)
- `--k-max`: Maximum number of clusters to evaluate (default: 30)
- `--subsample-fraction`: Fraction of data to subsample for stability (default: 0.8)
- `--n-repeats`: Number of stability runs per k (default: 20)

**Pros**:
- Systematic approach to selecting the number of clusters
- Well-established and interpretable
- Works reliably with various data sizes

**Cons**:
- Requires specifying a range for k
- Computationally expensive due to stability analysis
- May not handle noise/outliers well

**Use when**: You want a systematic, data-driven approach to determine the number of clusters with strong statistical validation.

**Example**:
```bash
python project_clustering_technologies.py \
    --method agglomerative \
    --k-min 5 \
    --k-max 20 \
    --n-repeats 30
```

### 2. HDBSCAN (Hierarchical Density-Based Spatial Clustering)

**Type**: Density-based clustering with automatic cluster discovery

**How it works**:
- Identifies dense regions in the data as clusters
- Automatically determines the number of clusters
- Labels low-density points as noise/outliers (cluster -1)
- Uses hierarchical approach for varying density clusters

**Parameters**:
- `--min-cluster-size`: Minimum samples required in a cluster (default: 5)
- `--min-samples`: Minimum samples in neighborhood for core point (default: 3)

**Pros**:
- Automatically discovers number of clusters
- Handles outliers explicitly (noise points)
- Can find clusters of varying densities and shapes
- No need for stability analysis

**Cons**:
- Requires `hdbscan` library installation
- Sensitive to parameter tuning
- May produce many small clusters or mark many points as noise

**Use when**: You expect natural density-based groupings in your data and want automatic cluster discovery with explicit outlier handling.

**Example**:
```bash
# Install dependency first
pip install hdbscan

# Run clustering
python project_clustering_technologies.py \
    --method hdbscan \
    --min-cluster-size 10 \
    --min-samples 5
```

### 3. Louvain Community Detection

**Type**: Graph-based community detection

**How it works**:
- Converts distance matrix to weighted similarity graph
- Uses threshold to create edges only for similar projects
- Applies Louvain algorithm to maximize modularity
- Automatically discovers communities (clusters)

**Parameters**:
- `--resolution`: Controls granularity; higher = more communities (default: 1.0)

**Pros**:
- Natural fit for "who uses similar tech" relationships
- Automatically determines number of clusters
- Fast and efficient for large networks
- No explicit outliers (every node assigned)

**Cons**:
- Requires `igraph` library installation
- Results depend on similarity threshold heuristic
- May produce imbalanced cluster sizes
- Less statistically rigorous than stability-based methods

**Use when**: You view projects as a network of technology-sharing relationships and want to discover natural communities.

**Example**:
```bash
# Install dependency first
pip install igraph

# Run clustering
python project_clustering_technologies.py \
    --method louvain \
    --resolution 1.5
```

## Common Parameters

All methods share these parameters:

- `--csv_path`: Input CSV with 'project' and 'technologies' columns
  - Default: `../data/technology_composition/project_technologies_filtered.csv`

- `--min-tech-frequency`: Minimum projects a technology must appear in to be included
  - Default: 2
  - Higher values reduce noise and dimensionality

- `--random-state`: Random seed for reproducibility
  - Default: 42

## Output Files

All methods produce the same output structure:

1. **`ecosystems_tech_matrix.csv`**: Binary project × technology matrix
   - Rows: projects
   - Columns: technologies (filtered by min-tech-frequency)
   - Values: 1 if project uses technology, 0 otherwise

2. **`ecosystems_project_assignments.csv`**: Cluster assignments
   - Columns: `project`, `ecosystem`
   - Note: HDBSCAN may have ecosystem = -1 for outliers

3. **`ecosystems_cluster_summary.csv`**: Summary of each cluster
   - Columns: `cluster_id`, `num_projects`, `top_technologies`
   - Top technologies listed with counts

4. **`ecosystems_embedding.png`**: 2D PCA visualization
   - Projects colored by cluster assignment
   - Legend shows top 3 technologies per cluster

5. **`ecosystems_stability.csv`**: Stability metrics (agglomerative only)
   - Per-k stability scores used for selection

## Complete Usage Examples

### Example 1: Quick Test with Sample Data

```bash
python project_clustering_technologies.py \
    --csv_path ../data/technology_composition/project_technologies_sample.csv \
    --min-tech-frequency 1 \
    --method agglomerative \
    --k-min 2 \
    --k-max 5 \
    --n-repeats 10
```

### Example 2: Full Analysis with Agglomerative

```bash
python project_clustering_technologies.py \
    --csv_path ../data/technology_composition/project_technologies_filtered.csv \
    --min-tech-frequency 5 \
    --method agglomerative \
    --k-min 5 \
    --k-max 30 \
    --subsample-fraction 0.8 \
    --n-repeats 20 \
    --random-state 42
```

### Example 3: HDBSCAN with Outlier Detection

```bash
python project_clustering_technologies.py \
    --method hdbscan \
    --min-cluster-size 10 \
    --min-samples 5 \
    --min-tech-frequency 3
```

### Example 4: Louvain with High Resolution

```bash
python project_clustering_technologies.py \
    --method louvain \
    --resolution 2.0 \
    --min-tech-frequency 5
```

### Example 5: Compare All Methods

```bash
# Test all three methods on the same data
for method in agglomerative hdbscan louvain; do
    echo "Testing $method..."
    python project_clustering_technologies.py \
        --csv_path ../data/technology_composition/project_technologies_sample.csv \
        --method $method \
        --min-tech-frequency 1
done
```

## Installation

### Core Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Optional Dependencies (for HDBSCAN and Louvain)

```bash
pip install -r clustering_requirements.txt
```

Or install individually:
```bash
pip install hdbscan  # For HDBSCAN method
pip install igraph   # For Louvain method
```

## Choosing the Right Method

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| Need statistically validated results | Agglomerative | Stability-based selection provides rigorous justification |
| Expect outlier projects | HDBSCAN | Explicitly identifies and handles outliers |
| Large dataset (>10k projects) | Louvain | Fast and scalable for large graphs |
| Uncertain about cluster count | HDBSCAN or Louvain | Both automatically determine number of clusters |
| Varying cluster densities | HDBSCAN | Handles multi-density clusters well |
| Network/community perspective | Louvain | Natural fit for relationship-based clustering |
| Maximum interpretability | Agglomerative | Well-established with clear stability metrics |

## Testing

A test script is provided to verify all methods work:

```bash
python test_clustering_methods.py
```

This runs all three methods on sample data and reports success/failure.

## Algorithm Details

### Distance Metric: Jaccard Distance

All methods use Jaccard distance to measure project dissimilarity:

```
Jaccard(A, B) = 1 - |A ∩ B| / |A ∪ B|
```

Where A and B are the sets of technologies used by two projects.

- Distance = 0: Projects use identical technologies
- Distance = 1: Projects share no technologies
- 0 < Distance < 1: Partial technology overlap

### Stability Analysis (Agglomerative)

For each candidate k:
1. Cluster full dataset → get labels_full
2. Repeat n times:
   - Subsample 80% of projects
   - Recluster subsample → get labels_sub
   - Compute ARI between labels_full (restricted to subsample) and labels_sub
3. Select k with highest median ARI

Higher ARI = more stable clustering at that k.

### HDBSCAN Core Algorithm

1. Build minimum spanning tree of mutual reachability distances
2. Convert to hierarchy of clusters using single linkage
3. Extract stable clusters using excess of mass criterion
4. Assign points to most persistent cluster or mark as noise

### Louvain Core Algorithm

1. Initialize: each node in own community
2. For each node, move to neighbor community that maximizes modularity gain
3. Create new network where communities become nodes
4. Repeat until no improvement
5. Return community assignments

## Troubleshooting

### "No module named 'hdbscan'"

Install the library: `pip install hdbscan`

### "No module named 'igraph'"

Install the library: `pip install igraph`

### Too many noise points with HDBSCAN

- Decrease `--min-cluster-size`
- Decrease `--min-samples`
- Increase `--min-tech-frequency` to reduce noise in data

### Too many small clusters with Louvain

- Decrease `--resolution` parameter
- Increase `--min-tech-frequency` to focus on core technologies

### Agglomerative taking too long

- Reduce `--k-max` range
- Reduce `--n-repeats` (sacrifices stability measurement)
- Increase `--min-tech-frequency` to reduce matrix size

## References

- **Agglomerative Clustering**: scikit-learn documentation
- **HDBSCAN**: McInnes, L., Healy, J., & Astels, S. (2017). "hdbscan: Hierarchical density based clustering." JOSS.
- **Louvain**: Blondel, V. D., et al. (2008). "Fast unfolding of communities in large networks." J. Stat. Mech.
- **Jaccard Distance**: Jaccard, P. (1912). "The distribution of the flora in the alpine zone."