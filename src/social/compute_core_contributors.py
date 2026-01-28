#!/usr/bin/env python3
"""
Core Contributor Analysis for Configuration Contributors

Determines whether config contributors are core contributors.

Definition: A core contributor is a developer belonging to the smallest subset
of contributors whose commits account for roughly 80% of all commits in the
project, reflecting the Pareto principle that a minority of contributors
performs the majority of development work.

Usage:
    # Single file
    python compute_core_contributors.py --input contributors.csv

    # Batch processing
    python compute_core_contributors.py --all --input ../../data/projects_contributors_merged
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """
    Auto-detect a column from a list of candidates.

    Args:
        df: DataFrame to search
        candidates: List of candidate column names

    Returns:
        Column name if found, None otherwise
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def identify_core_contributors(
    df: pd.DataFrame,
    total_commits_col: str,
    contributor_col: str,
    threshold: float = 0.80
) -> Tuple[set, pd.DataFrame]:
    """
    Identify core contributors based on the Pareto principle.

    Core contributors are the smallest subset of contributors whose commits
    account for the specified threshold (default 80%) of all commits.

    Args:
        df: DataFrame with contributor data
        total_commits_col: Column name for total commits
        contributor_col: Column name for contributor identifier
        threshold: Cumulative commit threshold (default 0.80 for 80%)

    Returns:
        Tuple of:
        - Set of core contributor names
        - DataFrame with contributor details including core status
    """
    # Calculate total commits per contributor
    commits = pd.to_numeric(df[total_commits_col], errors='coerce').fillna(0)

    result_df = pd.DataFrame({
        'contributor': df[contributor_col],
        'total_commits': commits
    })

    # Sort by commits descending
    result_df = result_df.sort_values('total_commits', ascending=False).reset_index(drop=True)

    # Calculate cumulative percentage
    total_all_commits = result_df['total_commits'].sum()
    if total_all_commits == 0:
        result_df['commit_share'] = 0.0
        result_df['cumulative_share'] = 0.0
        result_df['is_core'] = False
        return set(), result_df

    result_df['commit_share'] = result_df['total_commits'] / total_all_commits
    result_df['cumulative_share'] = result_df['commit_share'].cumsum()

    # Identify core contributors (those needed to reach threshold)
    # Include the contributor that crosses the threshold
    core_mask = result_df['cumulative_share'].shift(1, fill_value=0) < threshold
    result_df['is_core'] = core_mask

    core_contributors = set(result_df[result_df['is_core']]['contributor'])

    return core_contributors, result_df


def analyze_config_contributors(
    df: pd.DataFrame,
    core_contributors: set,
    contributor_col: str,
    config_commits_col: str
) -> pd.DataFrame:
    """
    Analyze which config contributors are core contributors.

    Args:
        df: Original DataFrame with contributor data
        core_contributors: Set of core contributor names
        contributor_col: Column name for contributor identifier
        config_commits_col: Column name for config commits

    Returns:
        DataFrame with config contributor analysis
    """
    config_commits = pd.to_numeric(df[config_commits_col], errors='coerce').fillna(0)

    # Filter to contributors with config commits
    config_mask = config_commits > 0

    result_df = pd.DataFrame({
        'contributor': df[contributor_col],
        'config_commits': config_commits,
        'has_config_commits': config_mask,
        'is_core': df[contributor_col].isin(core_contributors)
    })

    # Filter to only config contributors
    config_contributors_df = result_df[result_df['has_config_commits']].copy()
    config_contributors_df = config_contributors_df.sort_values('config_commits', ascending=False)

    return config_contributors_df


def process_single_project(
    input_file: Path,
    threshold: float = 0.80
) -> Optional[dict]:
    """
    Process a single project file.

    Args:
        input_file: Path to CSV file
        threshold: Core contributor threshold

    Returns:
        Dictionary with analysis results, or None on error
    """
    try:
        df = pd.read_csv(input_file)

        # Detect columns
        contributor_col = detect_column(df, ['Contributor', 'contributor', 'Author', 'author'])
        config_commits_col = detect_column(df, ['Config Commits', 'config_commits'])
        non_config_commits_col = detect_column(df, ['Non-Config Commits', 'non_config_commits'])

        if not contributor_col:
            print(f"Warning: No contributor column found in {input_file.name}", file=sys.stderr)
            return None

        if not config_commits_col:
            print(f"Warning: No config commits column found in {input_file.name}", file=sys.stderr)
            return None

        # Calculate total commits
        config_commits = pd.to_numeric(df[config_commits_col], errors='coerce').fillna(0)

        if non_config_commits_col:
            non_config_commits = pd.to_numeric(df[non_config_commits_col], errors='coerce').fillna(0)
            total_commits = config_commits + non_config_commits
        else:
            # If no non-config column, assume config commits are total
            total_commits = config_commits

        df['_total_commits'] = total_commits

        # Identify core contributors
        core_contributors, core_df = identify_core_contributors(
            df, '_total_commits', contributor_col, threshold
        )

        # Analyze config contributors
        config_analysis_df = analyze_config_contributors(
            df, core_contributors, contributor_col, config_commits_col
        )

        # Calculate statistics
        total_contributors = len(df)
        num_core = len(core_contributors)
        num_config_contributors = len(config_analysis_df)
        num_config_core = config_analysis_df['is_core'].sum()

        # Config commits by core vs non-core
        config_commits_by_core = config_analysis_df[config_analysis_df['is_core']]['config_commits'].sum()
        config_commits_by_non_core = config_analysis_df[~config_analysis_df['is_core']]['config_commits'].sum()
        total_config_commits = config_commits_by_core + config_commits_by_non_core

        project_name = input_file.stem.replace('_contributors_merged', '')

        return {
            'project_name': project_name,
            'total_contributors': total_contributors,
            'num_core_contributors': num_core,
            'num_config_contributors': num_config_contributors,
            'num_config_core': int(num_config_core),
            'num_config_non_core': num_config_contributors - int(num_config_core),
            'pct_config_are_core': (num_config_core / num_config_contributors * 100) if num_config_contributors > 0 else 0,
            'total_config_commits': total_config_commits,
            'config_commits_by_core': config_commits_by_core,
            'config_commits_by_non_core': config_commits_by_non_core,
            'pct_config_commits_by_core': (config_commits_by_core / total_config_commits * 100) if total_config_commits > 0 else 0,
            'core_df': core_df,
            'config_analysis_df': config_analysis_df,
            'threshold': threshold
        }

    except Exception as e:
        print(f"Warning: Failed to process {input_file.name}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


def process_all_projects(
    input_dir: Path,
    threshold: float = 0.80,
    limit: Optional[int] = None
) -> list:
    """
    Process all *_contributors_merged.csv files in a directory.

    Args:
        input_dir: Directory containing CSV files
        threshold: Core contributor threshold
        limit: Maximum number of projects to process

    Returns:
        List of result dictionaries
    """
    csv_files = sorted(input_dir.glob('*_contributors_merged.csv'))

    if limit is not None and limit > 0:
        csv_files = csv_files[:limit]

    if not csv_files:
        print(f"Error: No *_contributors_merged.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(csv_files)} projects...")
    results = []

    for idx, csv_file in enumerate(csv_files, 1):
        result = process_single_project(csv_file, threshold)
        if result:
            results.append(result)
            print(f"  [{idx}/{len(csv_files)}] {result['project_name']}: "
                  f"{result['num_config_core']}/{result['num_config_contributors']} config contributors are core "
                  f"({result['pct_config_are_core']:.1f}%)")
        else:
            print(f"  [{idx}/{len(csv_files)}] {csv_file.stem}: Failed")

    return results


def plot_single_project(result: dict, output_dir: Path):
    """
    Create visualizations for a single project.

    Args:
        result: Dictionary with analysis results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    project_name = result['project_name']

    # Plot 1: Pie chart of config contributors (core vs non-core)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Number of config contributors
    labels = ['Core', 'Non-Core']
    sizes = [result['num_config_core'], result['num_config_non_core']]
    colors = ['#2ecc71', '#e74c3c']

    if sum(sizes) > 0:
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, explode=(0.05, 0))
        axes[0].set_title(f'Config Contributors\n({result["num_config_contributors"]} total)')
    else:
        axes[0].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[0].set_title('Config Contributors')

    # Right: Config commits by core vs non-core
    sizes = [result['config_commits_by_core'], result['config_commits_by_non_core']]

    if sum(sizes) > 0:
        axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90, explode=(0.05, 0))
        axes[1].set_title(f'Config Commits\n({result["total_config_commits"]:.0f} total)')
    else:
        axes[1].text(0.5, 0.5, 'No data', ha='center', va='center')
        axes[1].set_title('Config Commits')

    plt.suptitle(f'Core Contributor Analysis: {project_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'{project_name}_core_contributors.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()


def plot_all_projects(results: list, output_dir: Path):
    """
    Create summary visualizations for all projects.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    project_names = [r['project_name'] for r in results]
    pct_config_are_core = [r['pct_config_are_core'] for r in results]
    pct_config_commits_by_core = [r['pct_config_commits_by_core'] for r in results]

    # Plot 1: Histogram of % config contributors that are core
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pct_config_are_core, bins=20, edgecolor='black', alpha=0.7, color='#3498db')
    axes[0].axvline(np.mean(pct_config_are_core), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(pct_config_are_core):.1f}%')
    axes[0].axvline(np.median(pct_config_are_core), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(pct_config_are_core):.1f}%')
    axes[0].set_xlabel('% of Config Contributors that are Core', fontsize=12)
    axes[0].set_ylabel('Number of Projects', fontsize=12)
    axes[0].set_title('Distribution: Config Contributors as Core', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].set_xlim(0, 100)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Histogram of % config commits by core
    axes[1].hist(pct_config_commits_by_core, bins=20, edgecolor='black', alpha=0.7, color='#2ecc71')
    axes[1].axvline(np.mean(pct_config_commits_by_core), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(pct_config_commits_by_core):.1f}%')
    axes[1].axvline(np.median(pct_config_commits_by_core), color='orange', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(pct_config_commits_by_core):.1f}%')
    axes[1].set_xlabel('% of Config Commits by Core Contributors', fontsize=12)
    axes[1].set_ylabel('Number of Projects', fontsize=12)
    axes[1].set_title('Distribution: Config Commits by Core', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].set_xlim(0, 100)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'core_contributors_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()

    # Plot 2: Scatter plot comparing the two metrics
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(pct_config_are_core, pct_config_commits_by_core, alpha=0.6, s=50)

    # Add diagonal reference line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='x = y')

    ax.set_xlabel('% of Config Contributors that are Core', fontsize=12)
    ax.set_ylabel('% of Config Commits by Core Contributors', fontsize=12)
    ax.set_title('Config Contributors vs Config Commits by Core', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'core_contributors_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()

    # Plot 3: Stacked bar chart for top projects
    fig, ax = plt.subplots(figsize=(14, max(8, len(results) * 0.3)))

    # Sort by % config commits by core
    sorted_results = sorted(results, key=lambda x: x['pct_config_commits_by_core'], reverse=True)
    if len(sorted_results) > 40:
        sorted_results = sorted_results[:40]

    y_pos = np.arange(len(sorted_results))
    project_names = [r['project_name'] for r in sorted_results]
    core_pcts = [r['pct_config_commits_by_core'] for r in sorted_results]
    non_core_pcts = [100 - r['pct_config_commits_by_core'] for r in sorted_results]

    ax.barh(y_pos, core_pcts, color='#2ecc71', label='Core Contributors', height=0.8)
    ax.barh(y_pos, non_core_pcts, left=core_pcts, color='#e74c3c', label='Non-Core Contributors', height=0.8)

    # Add contributor counts
    for idx, r in enumerate(sorted_results):
        ax.text(102, idx, f'({r["num_config_core"]}/{r["num_config_contributors"]})',
                ha='left', va='center', fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(project_names, fontsize=8)
    ax.set_xlabel('% of Config Commits', fontsize=12)
    ax.set_ylabel('Project', fontsize=12)
    ax.set_title('Config Commits: Core vs Non-Core Contributors', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'core_contributors_by_project.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze whether config contributors are core contributors',
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
        '--threshold',
        type=float,
        default=0.80,
        help='Cumulative commit threshold for core contributors (default: 0.80 for 80%%)'
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
        help='Output CSV file for results (default: <input_parent>/social/core_contributors.csv)'
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
        args.output = str(_social_dir / 'core_contributors.csv')
    if args.plot_dir is None:
        args.plot_dir = str(_social_dir)

    output_dir = Path(args.plot_dir)

    # Batch processing mode
    if args.all:
        if not input_path.is_dir():
            print(f"Error: --all requires input to be a directory, got: {input_path}", file=sys.stderr)
            sys.exit(1)

        results = process_all_projects(input_path, args.threshold, args.limit)

        if len(results) == 0:
            print("Error: No projects successfully processed", file=sys.stderr)
            sys.exit(1)

        # Save summary to CSV
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        summary_data = []
        for r in results:
            summary_data.append({
                'project_name': r['project_name'],
                'total_contributors': r['total_contributors'],
                'num_core_contributors': r['num_core_contributors'],
                'num_config_contributors': r['num_config_contributors'],
                'num_config_core': r['num_config_core'],
                'num_config_non_core': r['num_config_non_core'],
                'pct_config_are_core': r['pct_config_are_core'],
                'total_config_commits': r['total_config_commits'],
                'config_commits_by_core': r['config_commits_by_core'],
                'config_commits_by_non_core': r['config_commits_by_non_core'],
                'pct_config_commits_by_core': r['pct_config_commits_by_core'],
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSummary saved to: {output_csv}")

        # Print statistics
        print("\n" + "=" * 70)
        print(f"CORE CONTRIBUTOR ANALYSIS (threshold: {args.threshold:.0%})")
        print("=" * 70)
        print(f"Total projects: {len(results)}")

        print(f"\n% of Config Contributors that are Core:")
        print(f"  Mean:   {summary_df['pct_config_are_core'].mean():.1f}%")
        print(f"  Median: {summary_df['pct_config_are_core'].median():.1f}%")
        print(f"  Std:    {summary_df['pct_config_are_core'].std():.1f}%")

        print(f"\n% of Config Commits by Core Contributors:")
        print(f"  Mean:   {summary_df['pct_config_commits_by_core'].mean():.1f}%")
        print(f"  Median: {summary_df['pct_config_commits_by_core'].median():.1f}%")
        print(f"  Std:    {summary_df['pct_config_commits_by_core'].std():.1f}%")

        # Count projects where majority of config work is by core
        high_core = (summary_df['pct_config_commits_by_core'] >= 80).sum()
        medium_core = ((summary_df['pct_config_commits_by_core'] >= 50) &
                       (summary_df['pct_config_commits_by_core'] < 80)).sum()
        low_core = (summary_df['pct_config_commits_by_core'] < 50).sum()

        print(f"\nConfig Commits by Core (project breakdown):")
        print(f"  High (>=80%):    {high_core} projects ({high_core/len(results)*100:.1f}%)")
        print(f"  Medium (50-80%): {medium_core} projects ({medium_core/len(results)*100:.1f}%)")
        print(f"  Low (<50%):      {low_core} projects ({low_core/len(results)*100:.1f}%)")
        print("=" * 70)

        # Generate plots
        print("\nGenerating plots...")
        plot_all_projects(results, output_dir)

        sys.exit(0)

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = process_single_project(input_path, args.threshold)

    if not result:
        print("Error: Failed to process file", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("\n" + "=" * 70)
        print(f"CORE CONTRIBUTOR ANALYSIS: {result['project_name']}")
        print(f"(Core = smallest subset accounting for {result['threshold']:.0%} of commits)")
        print("=" * 70)
        print(f"\nTotal contributors: {result['total_contributors']}")
        print(f"Core contributors: {result['num_core_contributors']} "
              f"({result['num_core_contributors']/result['total_contributors']*100:.1f}%)")
        print(f"\nConfig contributors: {result['num_config_contributors']}")
        print(f"  - Core: {result['num_config_core']} ({result['pct_config_are_core']:.1f}%)")
        print(f"  - Non-Core: {result['num_config_non_core']} "
              f"({100-result['pct_config_are_core']:.1f}%)")
        print(f"\nConfig commits: {result['total_config_commits']:.0f}")
        print(f"  - By Core: {result['config_commits_by_core']:.0f} "
              f"({result['pct_config_commits_by_core']:.1f}%)")
        print(f"  - By Non-Core: {result['config_commits_by_non_core']:.0f} "
              f"({100-result['pct_config_commits_by_core']:.1f}%)")

        # Show top config contributors
        print("\nTop 10 Config Contributors:")
        config_df = result['config_analysis_df'].head(10)
        for _, row in config_df.iterrows():
            core_status = "CORE" if row['is_core'] else "non-core"
            name = row['contributor'][:40] + '...' if len(row['contributor']) > 40 else row['contributor']
            print(f"  {name}: {row['config_commits']:.0f} commits [{core_status}]")

        print("=" * 70)
    else:
        print(f"Config contributors: {result['num_config_core']}/{result['num_config_contributors']} are core "
              f"({result['pct_config_are_core']:.1f}%)")
        print(f"Config commits: {result['pct_config_commits_by_core']:.1f}% by core contributors")

    # Generate plot
    print("\nGenerating plot...")
    plot_single_project(result, output_dir)


if __name__ == '__main__':
    main()
