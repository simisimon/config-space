# Social Metrics for Configuration Work Distribution

This directory contains scripts for computing social metrics related to configuration work distribution across contributors.

## Summary of Computed Metrics

### 1. Gini Index (`compute_gini_index.py`)

Measures inequality in configuration commit distribution using the Gini coefficient.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `gini_all` | Gini coefficient across all contributors (including those with 0 config commits) | `G = (2 × Σ(i × xᵢ)) / (n × Σ(xᵢ)) - (n + 1) / n` where values are sorted ascending |
| `gini_active` | Gini coefficient among only active contributors (config_commits > 0) | Same formula, filtered to non-zero contributors |

Range: 0 (perfect equality) to 1 (perfect inequality).

### 2. Dominance (`compute_dominance.py`)

Measures concentration of configuration work in the top contributor.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `dominance` | Top-1 share of config commits | `max(commits) / sum(commits)` |

Range: 0 to 1, where higher values indicate more concentrated work.

### 3. Technology Distribution (`compute_technology_distribution.py`)

Analyzes how technology knowledge is distributed across contributors.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `technology_distribution` | Distribution of technologies per contributor | Count technologies touched per contributor by parsing config files and mapping to technologies |
| `technology_popularity` | Contributors per technology | Count unique contributors who touched each technology |

### 4. Participation Rate (`compute_participation_rate.py`)

Measures the fraction of contributors who work on configuration.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `participation_rate` | Fraction of contributors with config commits above threshold | `count(commits > threshold) / total_contributors` |

Range: 0 to 1.

### 5. Contributor Shares (`compute_contributor_shares.py`)

Calculates percentage share of config commits per contributor.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `share` | Percentage of total config commits per contributor | `contributor_commits / total_commits × 100` |
| `top1_share` | Cumulative share of top 1 contributor | Sum of top 1 share |
| `top3_share` | Cumulative share of top 3 contributors | Sum of top 3 shares |
| `top5_share` | Cumulative share of top 5 contributors | Sum of top 5 shares |

### 6. Contributor-Technology Frequency (`compute_contributor_technology_frequency.py`)

Computes interaction counts between contributors and technologies.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `frequency_matrix` | Matrix of contributor-technology interactions | Sum file touch counts per contributor-technology pair from config file data |

### 7. Core Contributors (`compute_core_contributors.py`)

Identifies whether config contributors are core contributors based on the Pareto principle.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `is_core` | Whether contributor is in the core group | Smallest subset of contributors accounting for 80% of total commits |
| `pct_config_are_core` | Percentage of config contributors that are core | `count(config_contributors ∩ core) / count(config_contributors) × 100` |
| `pct_config_commits_by_core` | Percentage of config commits by core contributors | `sum(config_commits by core) / total_config_commits × 100` |

Core contributors are identified by sorting contributors by commits descending and selecting those needed to reach the 80% cumulative threshold.

### 8. Commit Correlation (`compute_commit_correlation.py`)

Computes correlation between total commits and config commits.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `spearman_rho` | Spearman rank correlation coefficient | Rank-based correlation between total_commits and config_commits |
| `pearson_r` | Pearson correlation coefficient | Linear correlation for comparison |
| `permutation_pvalue` | Permutation test p-value | Significance via 10,000 permutations of the null distribution |
| `ci_lower_95`, `ci_upper_95` | 95% confidence interval | Bootstrap confidence interval (10,000 samples) |

### 9. Technology Risk (`compute_technology_risk.py`)

Identifies technologies at risk due to limited contributor coverage.

| Metric | Description | Computation |
|--------|-------------|-------------|
| `orphaned_technologies` | Technologies with only 1 contributor (bus factor = 1) | Technologies where `count(unique_contributors) == 1` |
| `endangered_technologies` | Technologies with 2-3 contributors where one has >80% of commits | Technologies where `2 <= contributors <= 3` and `max_share > 0.80` |
| `orphaned_rate` | Proportion of orphaned technologies | `orphaned_count / total_technologies` |
| `endangered_rate` | Proportion of endangered technologies | `endangered_count / total_technologies` |
| `at_risk_rate` | Combined at-risk proportion | `(orphaned + endangered) / total_technologies` |
