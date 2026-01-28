#!/usr/bin/env python3
"""
Gini Index Calculator for Project Contributors

Computes the Gini coefficient to measure inequality in configuration commit distribution.
The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).

Usage:
    # Single file
    python compute_gini_index.py --input contributors.csv
    python compute_gini_index.py --input contributors.csv --column "Config Commits"

    # Batch processing
    python compute_gini_index.py --all --input ../data/projects_contributors_merged
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for inequality measurement.

    Formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    where x is sorted in ascending order.

    Args:
        values: Array of numeric values (e.g., commit counts)

    Returns:
        Gini coefficient in [0, 1] where 0 = perfect equality, 1 = perfect inequality
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)
    total = sorted_values.sum()

    if total == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * total) - (n + 1) / n

    return float(gini)


def detect_commit_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect the commit count column in the DataFrame.

    Looks for columns containing keywords like 'config', 'commit', 'commits'.

    Args:
        df: DataFrame to search

    Returns:
        Column name if found, None otherwise
    """
    # Check exact matches first
    candidates = [
        'Config Commits',
        'config_commits',
        'configuration_commits',
        'Commits',
        'commits'
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Check partial matches
    for col in df.columns:
        col_lower = col.lower()
        if 'config' in col_lower and 'commit' in col_lower:
            return col
        if col_lower == 'commits':
            return col

    return None


def process_single_project(input_file: Path, column: Optional[str] = None) -> Optional[dict]:
    """
    Process a single project file and compute Gini coefficients.

    Args:
        input_file: Path to CSV file
        column: Optional column name (auto-detected if not provided)

    Returns:
        Dictionary with project_name, gini_all, gini_active, or None on error
    """
    try:
        df = pd.read_csv(input_file)

        # Detect or use specified column
        if column:
            commit_column = column
            if commit_column not in df.columns:
                return None
        else:
            commit_column = detect_commit_column(df)
            if not commit_column:
                return None

        # Extract commit values
        commits = pd.to_numeric(df[commit_column], errors='coerce').fillna(0).values

        # Compute Gini coefficients
        gini_all = compute_gini_coefficient(commits)
        active_commits = commits[commits > 0]
        gini_active = compute_gini_coefficient(active_commits)

        # Extract project name
        project_name = input_file.stem.replace('_contributors_merged', '')

        return {
            'project_name': project_name,
            'gini_all': round(gini_all, 4),
            'gini_active': round(gini_active, 4)
        }

    except Exception as e:
        print(f"Warning: Failed to process {input_file.name}: {e}", file=sys.stderr)
        return None


def process_all_projects(input_dir: Path, column: Optional[str] = None) -> pd.DataFrame:
    """
    Process all *_contributors_merged.csv files in a directory.

    Args:
        input_dir: Directory containing CSV files
        column: Optional column name (auto-detected if not provided)

    Returns:
        DataFrame with columns: project_name, gini_all, gini_active
    """
    csv_files = sorted(input_dir.glob('*_contributors_merged.csv'))

    if not csv_files:
        print(f"Error: No *_contributors_merged.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(csv_files)} projects...")
    results = []

    for idx, csv_file in enumerate(csv_files, 1):
        result = process_single_project(csv_file, column)
        if result:
            results.append(result)
            print(f"  [{idx}/{len(csv_files)}] {result['project_name']}: "
                  f"Gini(all)={result['gini_all']:.4f}, Gini(active)={result['gini_active']:.4f}")
        else:
            print(f"  [{idx}/{len(csv_files)}] {csv_file.stem}: Failed")

    return pd.DataFrame(results)


def plot_gini_distribution(results_df: pd.DataFrame, output_dir: Path):
    """
    Create individual visualizations of Gini coefficient distributions.

    Args:
        results_df: DataFrame with gini_all and gini_active columns
        output_dir: Directory to save the plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Histogram for all contributors
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(results_df['gini_all'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(results_df['gini_all'].mean(), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {results_df['gini_all'].mean():.4f}")
    ax.axvline(results_df['gini_all'].median(), color='orange', linestyle='--',
               linewidth=2, label=f"Median: {results_df['gini_all'].median():.4f}")
    ax.set_xlabel('Gini Coefficient', fontsize=12)
    ax.set_ylabel('Frequency (Number of Projects)', fontsize=12)
    ax.set_title('Gini Coefficient Distribution (All Contributors)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'gini_all_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 2: Histogram for active contributors
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(results_df['gini_active'], bins=20, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(results_df['gini_active'].mean(), color='red', linestyle='--',
               linewidth=2, label=f"Mean: {results_df['gini_active'].mean():.4f}")
    ax.axvline(results_df['gini_active'].median(), color='orange', linestyle='--',
               linewidth=2, label=f"Median: {results_df['gini_active'].median():.4f}")
    ax.set_xlabel('Gini Coefficient', fontsize=12)
    ax.set_ylabel('Frequency (Number of Projects)', fontsize=12)
    ax.set_title('Gini Coefficient Distribution (Active Contributors Only)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'gini_active_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 3: Box plot comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([results_df['gini_all'], results_df['gini_active']],
               tick_labels=['All Contributors', 'Active Only'])
    ax.set_ylabel('Gini Coefficient', fontsize=12)
    ax.set_title('Gini Coefficient Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'gini_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 4: Scatter plot: Gini (all) vs Gini (active)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results_df['gini_all'], results_df['gini_active'], alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, linewidth=2, label='y=x')
    ax.set_xlabel('Gini (All Contributors)', fontsize=12)
    ax.set_ylabel('Gini (Active Contributors)', fontsize=12)
    ax.set_title('Gini Coefficient: All vs Active Contributors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'gini_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute Gini coefficient for project contributor inequality',
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
        '--column',
        type=str,
        help='Name of the column containing commit counts (auto-detected if not specified)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all *_contributors_merged.csv files in the input directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for batch results (default: <input_parent>/social/gini_results.csv)'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: <input_parent>/social)'
    )
    parser.add_argument(
        '--active-only',
        action='store_true',
        help='Compute Gini only for contributors with non-zero commits (single file mode)'
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
        args.output = str(_social_dir / 'gini_results.csv')
    if args.plot_dir is None:
        args.plot_dir = str(_social_dir)

    # Batch processing mode
    if args.all:
        if not input_path.is_dir():
            print(f"Error: --all requires input to be a directory, got: {input_path}", file=sys.stderr)
            sys.exit(1)

        # Process all projects
        results_df = process_all_projects(input_path, args.column)

        if len(results_df) == 0:
            print("Error: No projects successfully processed", file=sys.stderr)
            sys.exit(1)

        # Save results to CSV
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total projects: {len(results_df)}")
        print(f"\nGini (All Contributors):")
        print(f"  Mean:   {results_df['gini_all'].mean():.4f}")
        print(f"  Median: {results_df['gini_all'].median():.4f}")
        print(f"  Std:    {results_df['gini_all'].std():.4f}")
        print(f"  Min:    {results_df['gini_all'].min():.4f}")
        print(f"  Max:    {results_df['gini_all'].max():.4f}")
        print(f"\nGini (Active Contributors Only):")
        print(f"  Mean:   {results_df['gini_active'].mean():.4f}")
        print(f"  Median: {results_df['gini_active'].median():.4f}")
        print(f"  Std:    {results_df['gini_active'].std():.4f}")
        print(f"  Min:    {results_df['gini_active'].min():.4f}")
        print(f"  Max:    {results_df['gini_active'].max():.4f}")
        print("=" * 60)

        # Create plots
        print("\nGenerating plots...")
        output_plot_dir = Path(args.plot_dir)
        plot_gini_distribution(results_df, output_plot_dir)

        sys.exit(0)

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Detect or use specified column
    if args.column:
        commit_column = args.column
        if commit_column not in df.columns:
            print(f"Error: Column '{commit_column}' not found in CSV", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
    else:
        commit_column = detect_commit_column(df)
        if not commit_column:
            print("Error: Could not auto-detect commit column", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            print("Please specify column name with --column", file=sys.stderr)
            sys.exit(1)

    if args.verbose:
        print(f"Using column: {commit_column}")

    # Extract commit values
    try:
        commits = pd.to_numeric(df[commit_column], errors='coerce').fillna(0).values
    except Exception as e:
        print(f"Error processing commit column: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute Gini coefficient
    gini_all = compute_gini_coefficient(commits)

    # Compute Gini for active contributors only
    active_commits = commits[commits > 0]
    gini_active = compute_gini_coefficient(active_commits)

    # Output results
    num_contributors = len(commits)
    num_active = len(active_commits)
    total_commits = commits.sum()

    if args.verbose:
        print("\n" + "=" * 60)
        print("GINI INDEX RESULTS")
        print("=" * 60)
        print(f"Total contributors: {num_contributors}")
        print(f"Active contributors: {num_active}")
        print(f"Total commits: {int(total_commits)}")
        print(f"\nGini coefficient (all contributors): {gini_all:.4f}")
        print(f"Gini coefficient (active only): {gini_active:.4f}")
        print("=" * 60)
    else:
        # Simple output for programmatic use
        if args.active_only:
            print(f"{gini_active:.4f}")
        else:
            print(f"{gini_all:.4f}")


if __name__ == '__main__':
    main()
