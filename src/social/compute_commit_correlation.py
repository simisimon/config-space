#!/usr/bin/env python3
"""
Commit Correlation Calculator

Computes Spearman rank correlation between total commits and config commits
to determine whether contributors with more commits tend to have more config commits.

Usage:
    # Single file
    python compute_commit_correlation.py --input project_contributors_merged.csv

    # Batch processing (all projects)
    python compute_commit_correlation.py --all --input ../../data/projects_contributors_merged
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Default number of permutations for significance test
DEFAULT_N_PERMUTATIONS = 10000


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detect the relevant columns in the DataFrame.

    Returns:
        Tuple of (config_commits_col, non_config_commits_col, contributor_col)
    """
    config_commits_col = None
    non_config_commits_col = None
    contributor_col = None

    # Config commits column
    for candidate in ['Config Commits', 'config_commits', 'configuration_commits']:
        if candidate in df.columns:
            config_commits_col = candidate
            break

    # Non-config commits column
    for candidate in ['Non-Config Commits', 'non_config_commits', 'NonConfig Commits']:
        if candidate in df.columns:
            non_config_commits_col = candidate
            break

    # Contributor column
    for candidate in ['Contributor', 'contributor', 'Author', 'author', 'Name', 'name']:
        if candidate in df.columns:
            contributor_col = candidate
            break

    return config_commits_col, non_config_commits_col, contributor_col


def permutation_test(x: np.ndarray, y: np.ndarray,
                     n_permutations: int = DEFAULT_N_PERMUTATIONS,
                     random_seed: int = 42) -> dict:
    """
    Perform a permutation test for Spearman correlation significance.

    This is a non-parametric test that makes no assumptions about the
    underlying distribution of the data.

    Args:
        x: First variable (e.g., total commits)
        y: Second variable (e.g., config commits)
        n_permutations: Number of permutations to perform
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with permutation test results
    """
    rng = np.random.default_rng(random_seed)

    # Observed correlation
    observed_rho = stats.spearmanr(x, y).statistic

    # Generate null distribution by permuting y
    null_distribution = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_permuted = rng.permutation(y)
        null_distribution[i] = stats.spearmanr(x, y_permuted).statistic

    # Two-tailed p-value: proportion of permuted correlations with
    # absolute value >= observed absolute value
    p_value_two_tailed = np.mean(np.abs(null_distribution) >= np.abs(observed_rho))

    # One-tailed p-value (for positive correlation):
    # proportion of permuted correlations >= observed
    p_value_one_tailed = np.mean(null_distribution >= observed_rho)

    return {
        'observed_rho': observed_rho,
        'p_value_two_tailed': p_value_two_tailed,
        'p_value_one_tailed': p_value_one_tailed,
        'null_mean': np.mean(null_distribution),
        'null_std': np.std(null_distribution),
        'n_permutations': n_permutations,
    }


def bootstrap_confidence_interval(x: np.ndarray, y: np.ndarray,
                                   n_bootstrap: int = 10000,
                                   confidence_level: float = 0.95,
                                   random_seed: int = 42) -> dict:
    """
    Compute bootstrap confidence interval for Spearman correlation.

    Args:
        x: First variable
        y: Second variable
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with bootstrap CI results
    """
    rng = np.random.default_rng(random_seed)
    n = len(x)

    bootstrap_rhos = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        bootstrap_rhos[i] = stats.spearmanr(x[indices], y[indices]).statistic

    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_rhos, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_rhos, 100 * (1 - alpha / 2))

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'bootstrap_mean': np.mean(bootstrap_rhos),
        'bootstrap_std': np.std(bootstrap_rhos),
        'n_bootstrap': n_bootstrap,
    }


def compute_correlation(df: pd.DataFrame,
                        config_commits_col: str,
                        non_config_commits_col: str,
                        run_permutation_test: bool = True,
                        n_permutations: int = DEFAULT_N_PERMUTATIONS) -> dict:
    """
    Compute Spearman rank correlation between total commits and config commits.

    Args:
        df: DataFrame with contributor data
        config_commits_col: Name of column containing config commit counts
        non_config_commits_col: Name of column containing non-config commit counts
        run_permutation_test: Whether to run the permutation significance test
        n_permutations: Number of permutations for the significance test

    Returns:
        Dictionary with correlation results
    """
    config_commits = pd.to_numeric(df[config_commits_col], errors='coerce').fillna(0)
    non_config_commits = pd.to_numeric(df[non_config_commits_col], errors='coerce').fillna(0)
    total_commits = config_commits + non_config_commits

    # Filter out contributors with zero total commits
    mask = (total_commits > 0) & (config_commits > 0)
    config_commits = config_commits[mask]
    total_commits = total_commits[mask]

    if len(config_commits) < 3:
        return {
            'n_contributors': len(config_commits),
            'spearman_rho': np.nan,
            'spearman_pvalue': np.nan,
            'pearson_r': np.nan,
            'pearson_pvalue': np.nan,
            'error': 'Insufficient data (need at least 3 contributors)'
        }

    # Compute Spearman rank correlation
    spearman_result = stats.spearmanr(total_commits, config_commits)

    # Compute Pearson correlation for comparison
    pearson_result = stats.pearsonr(total_commits, config_commits)

    result = {
        'n_contributors': len(config_commits),
        'spearman_rho': spearman_result.statistic,
        'spearman_pvalue': spearman_result.pvalue,
        'pearson_r': pearson_result.statistic,
        'pearson_pvalue': pearson_result.pvalue,
        'total_commits_sum': total_commits.sum(),
        'config_commits_sum': config_commits.sum(),
        'config_ratio': config_commits.sum() / total_commits.sum() if total_commits.sum() > 0 else 0,
        'total_commits': total_commits.values,
        'config_commits': config_commits.values,
    }

    # Run permutation test for more robust significance assessment
    if run_permutation_test:
        perm_result = permutation_test(
            total_commits.values, config_commits.values,
            n_permutations=n_permutations
        )
        result['permutation_pvalue_two_tailed'] = perm_result['p_value_two_tailed']
        result['permutation_pvalue_one_tailed'] = perm_result['p_value_one_tailed']
        result['permutation_null_mean'] = perm_result['null_mean']
        result['permutation_null_std'] = perm_result['null_std']

        # Bootstrap confidence interval
        bootstrap_result = bootstrap_confidence_interval(
            total_commits.values, config_commits.values
        )
        result['ci_lower_95'] = bootstrap_result['ci_lower']
        result['ci_upper_95'] = bootstrap_result['ci_upper']

    return result


def aggregate_significance_test(correlations: list) -> dict:
    """
    Test whether the correlations across all projects are significantly
    different from zero using multiple statistical tests.

    Args:
        correlations: List of Spearman correlation values from different projects

    Returns:
        Dictionary with aggregate significance test results
    """
    correlations = np.array(correlations)
    n = len(correlations)

    if n < 3:
        return {
            'error': 'Need at least 3 projects for aggregate test',
            'n_projects': n
        }

    # 1. One-sample t-test: H0: mean correlation = 0
    ttest_result = stats.ttest_1samp(correlations, 0)

    # 2. Wilcoxon signed-rank test: non-parametric alternative
    # H0: median correlation = 0
    # Note: Wilcoxon test requires at least 10 samples for reliable results
    if n >= 10:
        wilcoxon_result = stats.wilcoxon(correlations, alternative='two-sided')
        wilcoxon_stat = wilcoxon_result.statistic
        wilcoxon_pvalue = wilcoxon_result.pvalue
    else:
        wilcoxon_stat = np.nan
        wilcoxon_pvalue = np.nan

    # 3. Sign test: simplest non-parametric test
    # Count how many correlations are positive vs negative
    n_positive = np.sum(correlations > 0)
    n_negative = np.sum(correlations < 0)
    # Binomial test: H0: P(positive) = 0.5
    sign_test_result = stats.binomtest(n_positive, n_positive + n_negative, p=0.5)

    # 4. Fisher's method to combine p-values from individual tests
    # (only if we have individual p-values)

    # Effect size: Cohen's d for the mean correlation
    cohens_d = np.mean(correlations) / np.std(correlations, ddof=1) if np.std(correlations) > 0 else np.nan

    return {
        'n_projects': n,
        'mean_correlation': np.mean(correlations),
        'median_correlation': np.median(correlations),
        'std_correlation': np.std(correlations, ddof=1),
        'se_correlation': np.std(correlations, ddof=1) / np.sqrt(n),
        # One-sample t-test
        'ttest_statistic': ttest_result.statistic,
        'ttest_pvalue': ttest_result.pvalue,
        'ttest_df': n - 1,
        # Wilcoxon signed-rank test
        'wilcoxon_statistic': wilcoxon_stat,
        'wilcoxon_pvalue': wilcoxon_pvalue,
        # Sign test
        'n_positive': n_positive,
        'n_negative': n_negative,
        'sign_test_pvalue': sign_test_result.pvalue,
        # Effect size
        'cohens_d': cohens_d,
        # 95% CI for mean correlation (using t-distribution)
        'mean_ci_lower': np.mean(correlations) - stats.t.ppf(0.975, n-1) * np.std(correlations, ddof=1) / np.sqrt(n),
        'mean_ci_upper': np.mean(correlations) + stats.t.ppf(0.975, n-1) * np.std(correlations, ddof=1) / np.sqrt(n),
    }


def process_single_project(input_file: Path,
                           run_permutation_test: bool = True,
                           n_permutations: int = DEFAULT_N_PERMUTATIONS) -> Optional[dict]:
    """
    Process a single project file and compute correlation.

    Args:
        input_file: Path to CSV file
        run_permutation_test: Whether to run the permutation significance test
        n_permutations: Number of permutations for the significance test

    Returns:
        Dictionary with project results, or None on error
    """
    try:
        df = pd.read_csv(input_file)

        config_col, non_config_col, contributor_col = detect_columns(df)

        if not config_col or not non_config_col:
            print(f"Warning: Required columns not found in {input_file.name}", file=sys.stderr)
            return None

        result = compute_correlation(
            df, config_col, non_config_col,
            run_permutation_test=run_permutation_test,
            n_permutations=n_permutations
        )

        project_name = input_file.stem.replace('_contributors_merged', '')
        result['project_name'] = project_name

        return result

    except Exception as e:
        print(f"Warning: Failed to process {input_file.name}: {e}", file=sys.stderr)
        return None


def process_all_projects(input_dir: Path, limit: Optional[int] = None,
                         run_permutation_test: bool = True,
                         n_permutations: int = DEFAULT_N_PERMUTATIONS) -> list:
    """
    Process all *_contributors_merged.csv files in a directory.

    Args:
        input_dir: Directory containing CSV files
        limit: Maximum number of projects to process (None for all)
        run_permutation_test: Whether to run the permutation significance test
        n_permutations: Number of permutations for the significance test

    Returns:
        List of dictionaries with project results
    """
    csv_files = sorted(input_dir.glob('*_contributors_merged.csv'))
    if limit is not None and limit > 0:
        csv_files = csv_files[:limit]

    if not csv_files:
        print(f"Error: No *_contributors_merged.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(csv_files)} projects...")
    if run_permutation_test:
        print(f"  (Running permutation test with {n_permutations} permutations)")
    results = []

    for idx, csv_file in enumerate(csv_files, 1):
        result = process_single_project(
            csv_file,
            run_permutation_test=run_permutation_test,
            n_permutations=n_permutations
        )
        if result and not np.isnan(result['spearman_rho']):
            results.append(result)
            perm_info = ""
            if run_permutation_test and 'permutation_pvalue_two_tailed' in result:
                perm_p = result['permutation_pvalue_two_tailed']
                perm_str = f"{perm_p:.3f}" if perm_p >= 0.001 else "<0.001"
                perm_info = f", perm_p={perm_str}"
            print(f"  [{idx}/{len(csv_files)}] {result['project_name']}: "
                  f"rho={result['spearman_rho']:.3f}, p={result['spearman_pvalue']:.2e}{perm_info}, "
                  f"n={result['n_contributors']}")
        else:
            print(f"  [{idx}/{len(csv_files)}] {csv_file.stem}: "
                  f"{'Skipped (insufficient data)' if result else 'Failed'}")

    return results


def plot_scatter(result: dict, output_path: Path):
    """
    Create a scatter plot showing total commits vs config commits.

    Args:
        result: Dictionary with correlation results
        output_path: Path to save the plot
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_commits = result['total_commits']
    config_commits = result['config_commits']
    project_name = result['project_name']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(total_commits, config_commits, alpha=0.6, edgecolors='black', linewidth=0.5)

    # Add regression line
    z = np.polyfit(total_commits, config_commits, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(total_commits), max(total_commits), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2,
            label=f'Linear fit (slope={z[0]:.3f})')

    ax.set_xlabel('Total Commits', fontsize=12)
    ax.set_ylabel('Config Commits', fontsize=12)
    ax.set_title(f'Total Commits vs Config Commits: {project_name}\n'
                 f'Spearman rho={result["spearman_rho"]:.3f} (p={result["spearman_pvalue"]:.2e}), '
                 f'n={result["n_contributors"]}',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Set axis limits starting from 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Scatter plot saved: {output_path}")
    plt.close()


def plot_correlation_distribution(results: list, output_dir: Path):
    """
    Create plots showing the distribution of correlations across projects.

    Args:
        results: List of project result dictionaries
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    correlations = [r['spearman_rho'] for r in results]
    pvalues = [r['spearman_pvalue'] for r in results]
    n_contributors = [r['n_contributors'] for r in results]

    # Histogram of correlation coefficients
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histogram of Spearman rho
    ax1 = axes[0, 0]
    ax1.hist(correlations, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=np.median(correlations), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(correlations):.3f}')
    ax1.axvline(x=np.mean(correlations), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(correlations):.3f}')
    ax1.set_xlabel('Spearman Correlation (rho)', fontsize=11)
    ax1.set_ylabel('Number of Projects', fontsize=11)
    ax1.set_title('Distribution of Spearman Correlations\n(Total Commits vs Config Commits)',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Scatter of rho vs p-value (log scale)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(correlations, pvalues, c=n_contributors, cmap='viridis',
                          alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='p=0.05')
    ax2.axhline(y=0.01, color='orange', linestyle='--', linewidth=1, label='p=0.01')
    ax2.set_xlabel('Spearman Correlation (rho)', fontsize=11)
    ax2.set_ylabel('p-value (log scale)', fontsize=11)
    ax2.set_yscale('log')
    ax2.set_title('Correlation vs Statistical Significance', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='N Contributors')

    # 3. Scatter of n_contributors vs rho
    ax3 = axes[1, 0]
    ax3.scatter(n_contributors, correlations, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Number of Contributors', fontsize=11)
    ax3.set_ylabel('Spearman Correlation (rho)', fontsize=11)
    ax3.set_title('Correlation vs Project Size', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Summary statistics box
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Calculate summary statistics
    significant_005 = sum(1 for p in pvalues if p < 0.05)
    significant_001 = sum(1 for p in pvalues if p < 0.01)
    positive_corr = sum(1 for r in correlations if r > 0)
    strong_positive = sum(1 for r in correlations if r > 0.5)

    summary_text = f"""
    SUMMARY STATISTICS
    {'='*40}

    Total projects analyzed: {len(results)}

    Spearman Correlation (rho):
      Mean:   {np.mean(correlations):.3f}
      Median: {np.median(correlations):.3f}
      Std:    {np.std(correlations):.3f}
      Min:    {min(correlations):.3f}
      Max:    {max(correlations):.3f}

    Statistical Significance:
      p < 0.05: {significant_005} ({100*significant_005/len(results):.1f}%)
      p < 0.01: {significant_001} ({100*significant_001/len(results):.1f}%)

    Correlation Direction:
      Positive (rho > 0): {positive_corr} ({100*positive_corr/len(results):.1f}%)
      Strong positive (rho > 0.5): {strong_positive} ({100*strong_positive/len(results):.1f}%)

    Contributors per Project:
      Mean:   {np.mean(n_contributors):.1f}
      Median: {np.median(n_contributors):.1f}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'commit_correlation_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Distribution plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute Spearman correlation between total commits and config commits',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Name of the input directory (e.g., "netflix") under ../../data/, '
             'or a direct path to a CSV file or directory'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all *_contributors_merged.csv files in the input directory'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of projects to process (only applies with --all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for results (default: <input_parent>/social/commit_correlation.csv)'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: <input_parent>/social)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help=f'Number of permutations for significance test (default: {DEFAULT_N_PERMUTATIONS})'
    )
    parser.add_argument(
        '--skip-permutation-test',
        action='store_true',
        help='Skip the permutation test (faster but less robust)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # Resolve input: if it's just a name (not an existing path), treat as directory name under ../../data/
    data_root = Path(__file__).parent.parent.parent / 'data'
    if not input_path.exists() and not input_path.is_absolute() and (data_root / args.input).is_dir():
        base_dir = data_root / args.input
        input_path = base_dir / 'contributors_merged'
    elif input_path.is_dir():
        base_dir = input_path.parent
    else:
        base_dir = input_path.parent.parent

    _social_dir = base_dir / 'social'

    if input_path.is_dir():
        args.all = True

    if args.output is None:
        args.output = str(_social_dir / 'commit_correlation.csv')
    if args.plot_dir is None:
        args.plot_dir = str(_social_dir)

    output_dir = Path(args.plot_dir)

    # Batch processing mode
    if args.all:
        if not input_path.is_dir():
            print(f"Error: --all requires input to be a directory, got: {input_path}", file=sys.stderr)
            sys.exit(1)

        results = process_all_projects(
            input_path, args.limit,
            run_permutation_test=not args.skip_permutation_test,
            n_permutations=args.n_permutations
        )

        if len(results) == 0:
            print("Error: No projects successfully processed", file=sys.stderr)
            sys.exit(1)

        # Save results to CSV
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        summary_data = []
        for r in results:
            row = {
                'project_name': r['project_name'],
                'n_contributors': r['n_contributors'],
                'spearman_rho': r['spearman_rho'],
                'spearman_pvalue': r['spearman_pvalue'],
                'permutation_pvalue': r.get('permutation_pvalue_two_tailed', np.nan),
                'ci_lower_95': r.get('ci_lower_95', np.nan),
                'ci_upper_95': r.get('ci_upper_95', np.nan),
                'pearson_r': r['pearson_r'],
                'pearson_pvalue': r['pearson_pvalue'],
                'total_commits_sum': r['total_commits_sum'],
                'config_commits_sum': r['config_commits_sum'],
                'config_ratio': r['config_ratio'],
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('spearman_rho', ascending=False)
        summary_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

        # Print summary statistics
        correlations = summary_df['spearman_rho'].values
        pvalues = summary_df['spearman_pvalue'].values
        perm_pvalues = summary_df['permutation_pvalue'].dropna().values

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total projects: {len(results)}")
        print(f"\nSpearman Correlation (rho):")
        print(f"  Mean:   {np.mean(correlations):.3f}")
        print(f"  Median: {np.median(correlations):.3f}")
        print(f"  Std:    {np.std(correlations):.3f}")
        print(f"  Min:    {min(correlations):.3f}")
        print(f"  Max:    {max(correlations):.3f}")
        print(f"\nStatistical Significance (parametric p-value):")
        sig_005 = sum(1 for p in pvalues if p < 0.05)
        sig_001 = sum(1 for p in pvalues if p < 0.01)
        print(f"  p < 0.05: {sig_005} ({100*sig_005/len(results):.1f}%)")
        print(f"  p < 0.01: {sig_001} ({100*sig_001/len(results):.1f}%)")
        if len(perm_pvalues) > 0:
            print(f"\nStatistical Significance (permutation test):")
            perm_sig_005 = sum(1 for p in perm_pvalues if p < 0.05)
            perm_sig_001 = sum(1 for p in perm_pvalues if p < 0.01)
            print(f"  p < 0.05: {perm_sig_005} ({100*perm_sig_005/len(perm_pvalues):.1f}%)")
            print(f"  p < 0.01: {perm_sig_001} ({100*perm_sig_001/len(perm_pvalues):.1f}%)")
        pos_corr = sum(1 for r in correlations if r > 0)
        print(f"\nPositive correlation (rho > 0): {pos_corr} ({100*pos_corr/len(results):.1f}%)")
        print("=" * 60)

        # Run aggregate significance test
        print("\n" + "=" * 60)
        print("AGGREGATE SIGNIFICANCE TEST")
        print("=" * 60)
        agg_test = aggregate_significance_test(correlations)

        if 'error' in agg_test:
            print(f"Error: {agg_test['error']}")
        else:
            print(f"Testing H0: Mean correlation = 0 across all projects")
            print(f"\nDescriptive Statistics:")
            print(f"  N projects:         {agg_test['n_projects']}")
            print(f"  Mean rho:           {agg_test['mean_correlation']:.4f}")
            print(f"  Median rho:         {agg_test['median_correlation']:.4f}")
            print(f"  Std:                {agg_test['std_correlation']:.4f}")
            print(f"  SE:                 {agg_test['se_correlation']:.4f}")
            print(f"  95% CI:             [{agg_test['mean_ci_lower']:.4f}, {agg_test['mean_ci_upper']:.4f}]")
            print(f"\nOne-sample t-test:")
            print(f"  t-statistic:        {agg_test['ttest_statistic']:.4f}")
            print(f"  p-value:            {agg_test['ttest_pvalue']:.2e}")
            print(f"  df:                 {agg_test['ttest_df']}")
            if not np.isnan(agg_test['wilcoxon_pvalue']):
                print(f"\nWilcoxon signed-rank test:")
                print(f"  W-statistic:        {agg_test['wilcoxon_statistic']:.4f}")
                print(f"  p-value:            {agg_test['wilcoxon_pvalue']:.2e}")
            else:
                print(f"\nWilcoxon signed-rank test: N/A (need >= 10 projects)")
            print(f"\nSign test:")
            print(f"  Positive:           {agg_test['n_positive']}")
            print(f"  Negative:           {agg_test['n_negative']}")
            print(f"  p-value:            {agg_test['sign_test_pvalue']:.2e}")
            print(f"\nEffect size:")
            print(f"  Cohen's d:          {agg_test['cohens_d']:.4f}")

            # Interpretation
            print(f"\nInterpretation:")
            if agg_test['ttest_pvalue'] < 0.001:
                print(f"  The mean correlation is highly significantly different from zero (p < 0.001).")
            elif agg_test['ttest_pvalue'] < 0.01:
                print(f"  The mean correlation is significantly different from zero (p < 0.01).")
            elif agg_test['ttest_pvalue'] < 0.05:
                print(f"  The mean correlation is significantly different from zero (p < 0.05).")
            else:
                print(f"  The mean correlation is NOT significantly different from zero (p >= 0.05).")

            if agg_test['cohens_d'] > 0.8:
                effect_interp = "large"
            elif agg_test['cohens_d'] > 0.5:
                effect_interp = "medium"
            elif agg_test['cohens_d'] > 0.2:
                effect_interp = "small"
            else:
                effect_interp = "negligible"
            print(f"  Effect size is {effect_interp} (Cohen's d = {agg_test['cohens_d']:.2f}).")
        print("=" * 60)

        # Generate distribution plot
        print("\nGenerating plots...")
        plot_correlation_distribution(results, output_dir)

        sys.exit(0)

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = process_single_project(
        input_path,
        run_permutation_test=not args.skip_permutation_test,
        n_permutations=args.n_permutations
    )

    if not result:
        print("Error: Failed to process file", file=sys.stderr)
        sys.exit(1)

    if 'error' in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print("\n" + "=" * 60)
    print(f"COMMIT CORRELATION: {result['project_name']}")
    print("=" * 60)
    print(f"Number of contributors: {result['n_contributors']}")
    print(f"\nSpearman Rank Correlation:")
    print(f"  rho:     {result['spearman_rho']:.4f}")
    print(f"  p-value (parametric): {result['spearman_pvalue']:.2e}")
    if 'permutation_pvalue_two_tailed' in result:
        perm_p_two = result['permutation_pvalue_two_tailed']
        perm_p_one = result['permutation_pvalue_one_tailed']
        perm_str_two = f"{perm_p_two:.3f}" if perm_p_two >= 0.001 else "<0.001"
        perm_str_one = f"{perm_p_one:.3f}" if perm_p_one >= 0.001 else "<0.001"
        print(f"  p-value (permutation, two-tailed): {perm_str_two}")
        print(f"  p-value (permutation, one-tailed): {perm_str_one}")
    if 'ci_lower_95' in result:
        print(f"  95% CI (bootstrap): [{result['ci_lower_95']:.4f}, {result['ci_upper_95']:.4f}]")
    print(f"\nPearson Correlation (for comparison):")
    print(f"  r:       {result['pearson_r']:.4f}")
    print(f"  p-value: {result['pearson_pvalue']:.2e}")
    print(f"\nCommit Statistics:")
    print(f"  Total commits:  {result['total_commits_sum']:.0f}")
    print(f"  Config commits: {result['config_commits_sum']:.0f}")
    print(f"  Config ratio:   {result['config_ratio']*100:.1f}%")
    print("=" * 60)

    # Interpret result using permutation test p-value if available
    rho = result['spearman_rho']
    p = result.get('permutation_pvalue_two_tailed', result['spearman_pvalue'])
    p_source = "permutation test" if 'permutation_pvalue_two_tailed' in result else "parametric test"

    if p < 0.05:
        if rho > 0.7:
            interpretation = "Strong positive correlation"
        elif rho > 0.4:
            interpretation = "Moderate positive correlation"
        elif rho > 0:
            interpretation = "Weak positive correlation"
        elif rho > -0.4:
            interpretation = "Weak negative correlation"
        elif rho > -0.7:
            interpretation = "Moderate negative correlation"
        else:
            interpretation = "Strong negative correlation"
        print(f"\nInterpretation: {interpretation}")
        print(f"  Statistically significant at p<0.05 ({p_source})")
    else:
        print(f"\nInterpretation: No statistically significant correlation")
        print(f"  p={p:.3f} ({p_source})")

    # Generate scatter plot
    print("\nGenerating plot...")
    output_plot = output_dir / f'{result["project_name"]}_commit_correlation.png'
    plot_scatter(result, output_plot)


if __name__ == '__main__':
    main()
