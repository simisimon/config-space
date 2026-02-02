# Clustering Projects by Technology Configuration Patterns

This script (`project_clustering_technology.py`) identifies configuration archetypes for a specific technology by clustering projects based on how they configure that technology. Unlike clustering by technology stacks, this analysis focuses on configuration usage patterns within a single technology (e.g., different ways projects configure Docker, npm, or ESLint).

## Overview

The script analyzes configuration option-value pairs across projects that use a specific technology and identifies common configuration patterns. This reveals:
- **Configuration archetypes**: Common ways developers configure a technology
- **Best practices**: Popular configuration choices in the community
- **Configuration diversity**: Range of configuration strategies for a technology
- **Outliers**: Unusual or innovative configuration approaches

## Use Cases

1. **Technology Standards**: Identify common configuration patterns for a technology
2. **Best Practice Discovery**: Find prevalent configuration choices
3. **Configuration Simplification**: Understand which options are commonly used together
4. **Tool Development**: Design better defaults or configuration templates
5. **Documentation**: Identify which configuration combinations to document
6. **Migration Planning**: Understand configuration diversity before tool upgrades

## How It Works

### Pipeline Overview

```
Input: Project JSON files + Technology name
       
1. Extract configuration option=value pairs for the technology
       
2. Build binary matrix: projects × configuration pairs
       
3. Compute Jaccard distance between projects
       
4. Run clustering algorithm with automatic parameter sweep
       
5. Identify configuration archetypes
       
Output: Cluster assignments, summaries, visualizations
```

### Detailed Methodology

**Step 1: Configuration Extraction**
```
For each project JSON file:
  - Check if project uses the specified technology
  - Extract all configuration files for that technology
  - Parse option=value pairs from configuration data
  - Store: {option: {values: [...], types: [...]}}
```

**Step 2: Feature Matrix Construction**
```
Features: Each unique option=value pair becomes a feature
Example for npm:
  - "scripts=test" (project has test script)
  - "dependencies=react" (project depends on react)
  - "private=true" (package is private)

Matrix: Binary (1 if project has this option=value pair, 0 otherwise)
Filter: Keep only pairs appearing in e min_option_frequency projects
```

**Step 3: Distance Computation**
```
Jaccard distance between projects A and B:
  distance(A, B) = 1 - |A ) B| / |A * B|

Where A and B are sets of option=value pairs
```

**Step 4: Clustering with Automatic Parameter Selection**
Both methods perform automatic parameter sweeps to find optimal settings.

## Clustering Methods

### 1. Agglomerative Hierarchical Clustering (default)

**Type**: Hierarchical clustering with silhouette-based k selection

**How it works**:
- Evaluates multiple k values (number of clusters) in specified range
- For each k: computes silhouette score measuring cluster quality
- Automatically selects k with highest silhouette score
- Uses average linkage with precomputed Jaccard distance

**Parameters**:
- `--k-range`: Range of k values to evaluate (default: "2,20")
  Format: "min,max" (e.g., "3,15")

**Automatic Selection**:
- Computes silhouette score for each k in range
- Silhouette score: measures how well-separated clusters are (-1 to 1, higher is better)
- Selects k maximizing silhouette score
- Reports top 5 k values with their scores

**Pros**:
- Systematic parameter selection based on cluster quality
- Deterministic results
- Interpretable silhouette scores
- Every project assigned to a cluster

**Cons**:
- Requires specifying k range
- May not handle outliers well
- Assumes roughly equal cluster sizes

**Use when**: You want well-separated, interpretable configuration archetypes with automatic quality-based k selection.

### 2. HDBSCAN (Hierarchical Density-Based Spatial Clustering)

**Type**: Density-based clustering with automatic cluster discovery

**How it works**:
- Evaluates multiple min_cluster_size values in specified range
- For each value: identifies dense regions as clusters, marks sparse regions as noise
- Automatically selects min_cluster_size with highest silhouette score (excluding noise)
- Can identify outlier configurations (labeled -1)

**Parameters**:
- `--min-size-range`: Range of min_cluster_size values to evaluate (default: "3,20")
  Format: "min,max" (e.g., "5,15")

**Automatic Selection**:
- Computes silhouette score and noise count for each min_cluster_size
- Selects value maximizing silhouette score on non-noise points
- Reports: number of clusters, number of noise points, scores
- Shows top 5 parameter values

**Pros**:
- Identifies outlier/unusual configurations explicitly
- Handles varying cluster densities
- Automatic cluster number determination
- No assumption about cluster shapes

**Cons**:
- Requires `hdbscan` package: `pip install hdbscan`
- May produce many small clusters
- Some projects labeled as noise (not always desirable)
- Non-deterministic on ties

**Use when**: You expect natural groupings with outliers, want to identify unusual configuration patterns, or have varying configuration pattern densities.

## Installation

### Core Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Optional (for HDBSCAN)
```bash
pip install hdbscan
```

## Basic Usage

### Analyze a specific technology
```bash
python project_clustering_technology.py \
    --technology docker-compose \
    --data-dir ../data/projects_last_commit \
    --min-option-frequency 2
```

### Run with automatic parameter sweep (default behavior)
```bash
# Agglomerative: sweeps k from 2 to 20
python project_clustering_technology.py \
    --technology npm \
    --k-range "2,20"

# HDBSCAN: sweeps min_cluster_size from 3 to 20
python project_clustering_technology.py \
    --technology eslint \
    --method hdbscan \
    --min-size-range "3,20"
```

### Run both methods for comparison
```bash
python project_clustering_technology.py \
    --technology docker \
    --method all
```

This runs both agglomerative and HDBSCAN (if installed) with automatic parameter selection for each.

### Customize parameter sweep ranges
```bash
# Narrow k range for faster computation
python project_clustering_technology.py \
    --technology webpack \
    --k-range "3,10"

# Wider range for more thorough search
python project_clustering_technology.py \
    --technology prettier \
    --k-range "2,30"
```

### Adjust option frequency threshold
```bash
# Higher threshold: focus on common options
python project_clustering_technology.py \
    --technology babel \
    --min-option-frequency 5

# Lower threshold: include rare options (more features, more noise)
python project_clustering_technology.py \
    --technology typescript \
    --min-option-frequency 1
```

## Command-Line Arguments

### Required
- `--technology`: Technology to analyze (must match technology name in JSON data)
  Examples: "docker-compose", "npm", "eslint", "webpack", "typescript"

### Input/Output
- `--data-dir`: Directory with project JSON files (default: `../data/projects_last_commit`)
- `--output-dir`: Directory for results (default: `../data/project_clustering_technology`)

### Data Preprocessing
- `--min-option-frequency`: Minimum projects an option=value pair must appear in (default: 2)
  - Higher: fewer features, focus on common patterns, faster
  - Lower: more features, capture rare patterns, slower

### Clustering Method
- `--method`: Algorithm to use: `agglomerative`, `hdbscan`, or `all` (default: `agglomerative`)

### Agglomerative Parameters
- `--k-range`: Range of k values for sweep (default: "2,20")
  Format: "min,max"
  - Larger range: more thorough search, slower
  - Smaller range: faster, may miss optimal k

### HDBSCAN Parameters
- `--min-size-range`: Range of min_cluster_size values for sweep (default: "3,20")
  Format: "min,max"
  - Lower values: more clusters, less noise
  - Higher values: fewer clusters, more noise

### Other
- `--random-state`: Random seed for reproducibility (default: 42)

### Deprecated/Ignored Parameters
These are ignored since automatic sweeps are always performed:
- `--n-clusters`: Use `--k-range` instead
- `--min-cluster-size`: Use `--min-size-range` instead

## Output Files

All outputs saved to `../data/project_clustering_technology/` with technology-specific naming:

### Data Files

1. **`{technology}_config_matrix_{method}.csv`**
   Binary project � configuration pairs matrix
   - Rows: projects
   - Columns: option=value pairs (e.g., "scripts=test", "private=true")
   - Values: 1 if project has pair, 0 otherwise

2. **`{technology}_cluster_assignments_{method}.csv`**
   Project-to-cluster assignments
   - Columns: `project`, `cluster`, `num_options`, `num_files`
   - `cluster`: cluster ID (HDBSCAN: -1 for noise/outliers)
   - `num_options`: number of configuration options for this technology
   - `num_files`: number of configuration files for this technology

3. **`{technology}_cluster_summary_{method}.csv`**
   Per-cluster statistics
   - Columns: `cluster_id`, `num_projects`, `top_options`
   - `top_options`: Most common option=value pairs with counts (format: "opt=val:count; opt=val:count; ...")

4. **`{technology}_k_sweep_agglomerative.csv`** (Agglomerative only)
   Results from automatic k sweep
   - Columns: `k`, `silhouette_score`, `n_clusters`
   - Shows quality metric for each k value evaluated
   - Best k is the one with highest silhouette_score

5. **`{technology}_min_size_sweep_hdbscan.csv`** (HDBSCAN only)
   Results from automatic min_cluster_size sweep
   - Columns: `min_cluster_size`, `silhouette_score`, `n_clusters`, `n_noise`
   - Shows clusters found and noise points for each parameter value
   - Best value is the one with highest silhouette_score

### Visualizations

1. **`{technology}_pca_{method}.png`**
   2D PCA projection of projects colored by cluster
   - Shows explained variance percentage
   - Legend shows cluster ID and size
   - Jittered points to reveal overlaps
   - Title includes technology, method, and number of projects

2. **`{technology}_heatmap_{method}.png`**
   Configuration option usage heatmap by cluster
   - Rows: clusters
   - Columns: option=value pairs
   - Values: % of projects in cluster using each option
   - Shows only important options (e20% usage in some cluster, or top 20)
   - Identifies defining configuration patterns for each archetype

## Interpreting Results

### Understanding Clusters

Each cluster represents a **configuration archetype** - a common way projects configure the technology.

**Example: npm clustering might find:**
- **Cluster 0**: Library packages (private=false, scripts focused on build/publish)
- **Cluster 1**: Application packages (private=true, scripts for start/dev/test)
- **Cluster 2**: Monorepo workspaces (workspaces defined, many dev dependencies)
- **Cluster 3**: TypeScript projects (typescript dependency, build scripts)

### Cluster Summary CSV

```
cluster_id,num_projects,top_options
0,45,"scripts=test:42; scripts=build:40; private=false:38; ..."
1,32,"private=true:30; scripts=start:28; scripts=dev:25; ..."
```

**Reading**:
- Cluster 0 has 45 projects
- 42 of them have "scripts=test"
- 40 have "scripts=build"
- etc.

### Sweep Results CSV (Agglomerative)

```
k,silhouette_score,n_clusters
2,0.245,2
3,0.312,3
4,0.387,4  <- Best (highest silhouette)
5,0.356,5
```

**Interpretation**:
- k=4 has highest silhouette score (0.387)
- This k produces the most well-separated clusters
- Automatically selected for final clustering

### Sweep Results CSV (HDBSCAN)

```
min_cluster_size,silhouette_score,n_clusters,n_noise
3,0.298,8,15
5,0.342,5,22  <- Best (highest silhouette)
10,0.315,3,45
```

**Interpretation**:
- min_cluster_size=5 has highest silhouette score (0.342)
- Produces 5 clusters with 22 noise points
- Automatically selected for final clustering

### PCA Plot

- **Proximity**: Projects close together have similar configurations
- **Clusters**: Color-coded groups show identified archetypes
- **Separation**: Well-separated clusters indicate distinct configuration patterns
- **Overlap**: Fuzzy boundaries suggest gradual transitions between patterns

### Heatmap

- **Hot (bright) cells**: Defining configurations for an archetype
  - Example: Cluster 0 has 90% usage of "private=false" � likely library packages
- **Cold (dark) cells**: Rare or absent configurations
- **Row patterns**: Configuration signature of each archetype
- **Column patterns**: Options spanning multiple archetypes (common practices)

## Example Workflows

### Workflow 1: Quick Configuration Analysis
```bash
# Analyze Docker Compose configurations
python project_clustering_technology.py \
    --technology docker-compose \
    --method all \
    --min-option-frequency 3

# Check sweep CSVs to see if automatic selection was reasonable
# Review cluster summaries to understand configuration archetypes
# Examine heatmap to see defining options per cluster
```

### Workflow 2: Fine-Tuning for Specific Technology
```bash
# Step 1: Wide parameter sweep
python project_clustering_technology.py \
    --technology eslint \
    --k-range "2,30" \
    --min-option-frequency 2

# Step 2: Check sweep CSV for silhouette scores
# Look for plateau or peak in scores

# Step 3: If needed, adjust range and rerun
python project_clustering_technology.py \
    --technology eslint \
    --k-range "4,12" \
    --min-option-frequency 3

# Step 4: Analyze final cluster summaries and visualizations
```

### Workflow 3: Comparing Methods
```bash
# Run both methods with same data preprocessing
python project_clustering_technology.py \
    --technology webpack \
    --method all \
    --min-option-frequency 5

# Compare:
# - Number of clusters found
# - Silhouette scores
# - Cluster size distributions
# - Top options per cluster
# - Visual separation in PCA plots
```

### Workflow 4: Identifying Outlier Configurations
```bash
# Use HDBSCAN to find unusual configurations
python project_clustering_technology.py \
    --technology typescript \
    --method hdbscan \
    --min-size-range "5,25"

# Check assignments CSV for cluster=-1 (noise points)
# These are projects with unusual configuration patterns
# Investigate them separately for innovative approaches
```

### Workflow 5: Technology Comparison
```bash
# Analyze configuration diversity across technologies
for tech in npm docker-compose eslint webpack; do
    echo "Analyzing $tech..."
    python project_clustering_technology.py \
        --technology $tech \
        --method agglomerative \
        --min-option-frequency 5
done

# Compare:
# - Number of archetypes (k values selected)
# - Silhouette scores (configuration diversity)
# - Cluster size distributions (centralization vs diversity)
```

## Tips for Parameter Selection

### min-option-frequency

**Impact**: Controls feature space size and noise level

- **Low (1-2)**:
  - Pros: Captures rare/innovative configurations
  - Cons: Many features, more noise, slower computation
  - Use: Small datasets, want comprehensive analysis

- **Medium (3-5)**:
  - Pros: Balanced feature set, good noise filtering
  - Cons: May miss niche patterns
  - Use: Most analyses, recommended starting point

- **High (>5)**:
  - Pros: Focus on widespread patterns, fast computation
  - Cons: Loses diversity, may miss interesting variations
  - Use: Large datasets, want only common patterns

### k-range (Agglomerative)

**Impact**: Search space for number of configuration archetypes

- **Wide (e.g., 2,30)**:
  - Pros: Thorough search, less likely to miss optimal
  - Cons: Slower computation
  - Use: Initial exploration, unclear expected archetypes

- **Narrow (e.g., 3,10)**:
  - Pros: Faster computation
  - Cons: May miss optimal if outside range
  - Use: Refinement after initial exploration

**Strategy**: Start wide, check sweep CSV, narrow range if needed

### min-size-range (HDBSCAN)

**Impact**: Trade-off between cluster granularity and noise

- **Lower values (e.g., 2,10)**:
  - Effect: More clusters, less noise
  - Use: Want fine-grained archetypes, few outliers

- **Higher values (e.g., 10,25)**:
  - Effect: Fewer clusters, more noise
  - Use: Want only major archetypes, explicit outlier detection

**Strategy**: Check sweep CSV for noise counts, adjust if too many/few noise points

## Troubleshooting

### No Projects Found
**Problem**: "No projects found using technology: X"
**Solutions**:
- Check technology name spelling (must match JSON data exactly)
- Verify JSON files in `--data-dir` have the expected structure
- Check if technology appears in `concepts` field of JSON files

### No Option=Value Pairs After Filtering
**Problem**: "No option=value pairs found with frequency >= N"
**Solutions**:
- Lower `--min-option-frequency` (e.g., from 5 to 2)
- Check if projects actually have configuration options (not just using technology)
- Verify JSON files have `config_file_data` with `pairs`

### Import Error: hdbscan
**Problem**: `ModuleNotFoundError: No module named 'hdbscan'`
**Solution**:
```bash
pip install hdbscan
```
Or use `--method agglomerative` instead

### Too Few Clusters
**Problem**: Sweep selects k=2 or k=3, seems too simple
**Solutions**:
- Data may genuinely have few configuration patterns
- Increase `--min-option-frequency` to focus on meaningful differences
- Check if option=value pairs are too general
- Review sweep CSV to see if scores are flat (indistinguishable patterns)

### Too Many Clusters
**Problem**: Sweep selects k=15+, seems over-segmented
**Solutions**:
- May indicate high configuration diversity (legitimate)
- Increase `--min-option-frequency` to reduce noise
- Check sweep CSV for silhouette scores - if low (<0.2), clusters may not be meaningful
- Review cluster summaries - many similar clusters suggest over-segmentation

### All Noise Points (HDBSCAN)
**Problem**: Most/all projects labeled as cluster=-1
**Solutions**:
- Lower min_cluster_size in `--min-size-range` (e.g., "2,10")
- Configurations may be too diverse for density-based clustering
- Try agglomerative method instead
- Increase `--min-option-frequency` to focus on clear patterns

### Poor PCA Visualization
**Problem**: Low explained variance (<20%)
**Interpretation**:
- Configuration space is high-dimensional
- PCA cannot capture complexity in 2D
- This is expected for diverse configurations
- Clusters are still meaningful, just hard to visualize

**Solutions**:
- Focus on heatmap instead (shows full pattern)
- Check silhouette scores (meaningful despite visualization)
- Accept that configuration diversity is genuinely complex

### Sweep Takes Too Long
**Problem**: Parameter sweep is slow
**Solutions**:
- Reduce k-range or min-size-range (e.g., "5,15" instead of "2,30")
- Increase `--min-option-frequency` (reduces matrix size)
- Use `--method agglomerative` (faster than HDBSCAN)
- Process fewer projects (filter input data)

## Understanding Automatic Parameter Selection

### Why Automatic Selection?

Manual parameter selection is difficult because:
- Optimal k or min_cluster_size depends on data characteristics
- Trial-and-error is time-consuming
- Different technologies have different optimal values
- Silhouette score provides objective quality measure

### How It Works

**For each parameter value:**
1. Run clustering
2. Compute silhouette score
3. Record results

**Selection:**
- Choose parameter with **highest silhouette score**
- Report top 5 for user inspection
- Use selected value for final clustering

### Validating Automatic Selection

**Check the sweep CSV:**
```python
import pandas as pd

# For agglomerative
sweep = pd.read_csv("docker-compose_k_sweep_agglomerative.csv")
print(sweep.nlargest(5, 'silhouette_score'))

# For HDBSCAN
sweep = pd.read_csv("eslint_min_size_sweep_hdbscan.csv")
print(sweep.nlargest(5, 'silhouette_score'))
```

**Good indicators:**
- Clear peak in silhouette score
- Reasonable number of clusters (not too few/many)
- Gradual score changes (smooth curve)

**Warning signs:**
- Flat scores (all similar) � data may not cluster well
- Score at boundary of range � expand range
- Very low scores (<0.1) � clusters not well-separated

## Comparison with Technology Stack Clustering

This script is different from `project_clustering_technology_stack.py`:

| Aspect | This Script (Configuration) | Stack Script |
|--------|----------------------------|--------------|
| **Focus** | How projects configure ONE technology | What technologies projects use |
| **Input** | Configuration option=value pairs | Technology presence/absence |
| **Question** | "How do projects configure Docker?" | "What technology stacks exist?" |
| **Clusters** | Configuration archetypes | Technology ecosystems |
| **Example** | npm: libraries vs apps vs monorepos | Frontend: React vs Vue vs Angular |
| **Use Case** | Understand configuration best practices | Understand technology combinations |

**Use both when**: You want comprehensive understanding - which technologies are used together (stack) AND how they're configured (this script).

## Advanced Usage

### Batch Analysis of Multiple Technologies
```bash
#!/bin/bash
# analyze_all_technologies.sh

technologies=("npm" "docker-compose" "eslint" "webpack" "typescript" "babel")

for tech in "${technologies[@]}"; do
    echo "========================================="
    echo "Analyzing: $tech"
    echo "========================================="

    python project_clustering_technology.py \
        --technology "$tech" \
        --method all \
        --min-option-frequency 3 \
        --k-range "2,20" \
        --min-size-range "3,20"

    echo ""
done

echo "All technologies analyzed!"
```

### Extracting Insights from Results
```python
import pandas as pd
import glob

# Load all cluster summaries
summaries = []
for csv_file in glob.glob("../data/project_clustering_technology/*_cluster_summary_*.csv"):
    df = pd.read_csv(csv_file)
    tech = csv_file.split("/")[-1].split("_cluster")[0]
    method = csv_file.split("_")[-1].replace(".csv", "")
    df['technology'] = tech
    df['method'] = method
    summaries.append(df)

all_summaries = pd.concat(summaries, ignore_index=True)

# Analyze configuration diversity
diversity = all_summaries.groupby(['technology', 'method']).agg({
    'cluster_id': 'count',  # number of archetypes
    'num_projects': 'sum'   # total projects
}).reset_index()

print("Configuration Diversity by Technology:")
print(diversity)
```

## Citation

If you use this clustering approach in research, describe the methodology as:

> We identified configuration archetypes for [technology name] by clustering projects based on their configuration option-value pairs. Projects were represented as binary feature vectors indicating the presence/absence of specific configuration settings, and pairwise Jaccard distances were computed to measure configuration similarity. [For Agglomerative: We performed an automatic parameter sweep evaluating k values from X to Y, selecting the number of clusters that maximized the silhouette score.] [For HDBSCAN: We performed an automatic parameter sweep evaluating min_cluster_size values from X to Y, selecting the parameter that maximized the silhouette score while identifying outlier configurations as noise points.] The resulting clusters represent common configuration patterns, with each archetype defined by its most frequent option-value pairs.

## References

- **Agglomerative Clustering**: scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
- **HDBSCAN**: McInnes, L., Healy, J., & Astels, S. (2017). "hdbscan: Hierarchical density based clustering." Journal of Open Source Software, 2(11), 205.
- **Jaccard Distance**: Jaccard, P. (1912). "The distribution of the flora in the alpine zone."
- **Silhouette Coefficient**: Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." Journal of Computational and Applied Mathematics, 20, 53-65.
