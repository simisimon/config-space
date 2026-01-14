#!/usr/bin/env python3
"""
Configuration Dominance (Top-1 Share) Calculator

Calculates the dominance metric for projects:
    Dominance = (Config commits by top contributor) / (Total config commits)

This measures how concentrated the configuration work is in the hands of the single
most active contributor.

Usage:
    # Single file
    python compute_dominance.py --input contributors.csv

    # Batch processing
    python compute_dominance.py --all --input ../../data/projects_contributors_merged
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_commit_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect the config commit count column in the DataFrame.

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


def compute_dominance(df: pd.DataFrame, commit_column: str) -> dict:
    """
    Compute dominance (top-1 share) for a project.

    Args:
        df: DataFrame with contributor data
        commit_column: Name of column containing commit counts

    Returns:
        Dictionary with dominance metrics
    """
    commits = pd.to_numeric(df[commit_column], errors='coerce').fillna(0).values

    total_commits = commits.sum()

    if total_commits == 0:
        return {
            'total_commits': 0,
            'top_contributor_commits': 0,
            'dominance': 0.0,
            'num_contributors': len(commits)
        }

    top_contributor_commits = commits.max()
    dominance = top_contributor_commits / total_commits

    return {
        'total_commits': int(total_commits),
        'top_contributor_commits': int(top_contributor_commits),
        'dominance': round(dominance, 4),
        'num_contributors': len(commits)
    }


def process_single_project(input_file: Path, column: Optional[str] = None) -> Optional[dict]:
    """
    Process a single project file and compute dominance.

    Args:
        input_file: Path to CSV file
        column: Optional column name (auto-detected if not provided)

    Returns:
        Dictionary with project_name and dominance metrics, or None on error
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

        # Compute dominance
        metrics = compute_dominance(df, commit_column)

        # Extract project name
        project_name = input_file.stem.replace('_contributors_merged', '')

        return {
            'project_name': project_name,
            **metrics
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
        DataFrame with columns: project_name, total_commits, top_contributor_commits, dominance, num_contributors
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
                  f"Dominance={result['dominance']:.2%} "
                  f"({result['top_contributor_commits']}/{result['total_commits']} commits)")
        else:
            print(f"  [{idx}/{len(csv_files)}] {csv_file.stem}: Failed")

    return pd.DataFrame(results)


def plot_dominance(results_df: pd.DataFrame, output_dir: Path):
    """
    Create individual visualizations of dominance distributions.

    Args:
        results_df: DataFrame with dominance data
        output_dir: Directory to save the plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Histogram of dominance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(results_df['dominance'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results_df['dominance'].mean(), color='red',
               linestyle='--', linewidth=2,
               label=f"Mean: {results_df['dominance'].mean():.2%}")
    ax.axvline(results_df['dominance'].median(), color='orange',
               linestyle='--', linewidth=2,
               label=f"Median: {results_df['dominance'].median():.2%}")
    ax.set_xlabel('Dominance (Top-1 Share)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Projects)', fontsize=12)
    ax.set_title('Configuration Dominance Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'dominance_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 2: Box plot of dominance
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([results_df['dominance']],
               tick_labels=['Dominance'])
    ax.set_ylabel('Dominance (Top-1 Share)', fontsize=12)
    ax.set_title('Dominance Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'dominance_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 3: Scatter plot: Total commits vs Top contributor commits
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results_df['total_commits'],
              results_df['top_contributor_commits'],
              alpha=0.5, s=30, c=results_df['dominance'],
              cmap='YlOrRd')
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Dominance')
    # Add reference lines for different dominance levels
    max_val = results_df['total_commits'].max()
    for dom_level, label in [(0.5, '50%'), (0.75, '75%'), (1.0, '100%')]:
        x = np.linspace(1, max_val, 100)
        y = dom_level * x
        ax.plot(x, y, '--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Total Config Commits (log scale)', fontsize=12)
    ax.set_ylabel('Top Contributor Commits (log scale)', fontsize=12)
    ax.set_title('Total Commits vs Top Contributor Commits', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / 'dominance_commits_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 4: Scatter plot: Number of contributors vs Dominance
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(results_df['num_contributors'],
                        results_df['dominance'],
                        alpha=0.5, s=30, c=results_df['dominance'],
                        cmap='YlOrRd')
    plt.colorbar(scatter, ax=ax, label='Dominance')
    ax.set_xlabel('Number of Contributors (log scale)', fontsize=12)
    ax.set_ylabel('Dominance (Top-1 Share)', fontsize=12)
    ax.set_title('Project Size vs Dominance', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'dominance_size_vs_dominance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute configuration dominance (top-1 share) for projects',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to CSV file or directory containing contributor data'
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
        default='../../data/social/dominance_results.csv',
        help='Output CSV file for batch results (default: ../../data/social/dominance_results.csv)'
    )
    parser.add_argument(
        '--plot-dir',
        type=str,
        default='../../data/social',
        help='Output directory for plots (default: ../../data/social)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

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
        print(f"\nDominance (Top-1 Share):")
        print(f"  Mean:   {results_df['dominance'].mean():.2%}")
        print(f"  Median: {results_df['dominance'].median():.2%}")
        print(f"  Std:    {results_df['dominance'].std():.4f}")
        print(f"  Min:    {results_df['dominance'].min():.2%}")
        print(f"  Max:    {results_df['dominance'].max():.2%}")

        # Count projects by dominance level
        high_dominance = (results_df['dominance'] >= 0.75).sum()
        medium_dominance = ((results_df['dominance'] >= 0.5) & (results_df['dominance'] < 0.75)).sum()
        low_dominance = (results_df['dominance'] < 0.5).sum()

        print(f"\nDominance Levels:")
        print(f"  High (≥75%):   {high_dominance} projects ({high_dominance/len(results_df):.1%})")
        print(f"  Medium (50-75%): {medium_dominance} projects ({medium_dominance/len(results_df):.1%})")
        print(f"  Low (<50%):    {low_dominance} projects ({low_dominance/len(results_df):.1%})")

        print(f"\nTotal Config Commits:")
        print(f"  Mean:   {results_df['total_commits'].mean():.1f}")
        print(f"  Median: {results_df['total_commits'].median():.0f}")
        print(f"  Min:    {results_df['total_commits'].min()}")
        print(f"  Max:    {results_df['total_commits'].max()}")
        print("=" * 60)

        # Create plots
        print("\nGenerating plots...")
        output_plot_dir = Path(args.plot_dir)
        plot_dominance(results_df, output_plot_dir)

        sys.exit(0)

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = process_single_project(input_path, args.column)

    if not result:
        print("Error: Failed to process file", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("\n" + "=" * 60)
        print("DOMINANCE RESULTS")
        print("=" * 60)
        print(f"Project: {result['project_name']}")
        print(f"Total config commits: {result['total_commits']}")
        print(f"Top contributor commits: {result['top_contributor_commits']}")
        print(f"Dominance (top-1 share): {result['dominance']:.2%}")
        print(f"Number of contributors: {result['num_contributors']}")
        print("=" * 60)
    else:
        print(f"{result['dominance']:.4f}")


if __name__ == '__main__':
    main()
