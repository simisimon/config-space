#!/usr/bin/env python3
"""
Configuration Participation Rate Calculator

Calculates the configuration participation rate for projects:
    Participation Rate = (Number of contributors with > 1 config commit) / (Total contributors)

Usage:
    # Single file
    python compute_participation_rate.py --input contributors.csv

    # Batch processing
    python compute_participation_rate.py --all --input ../../data/projects_contributors_merged
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


def compute_participation_rate(df: pd.DataFrame, commit_column: str, threshold: int = 1) -> dict:
    """
    Compute configuration participation rate for a project.

    Args:
        df: DataFrame with contributor data
        commit_column: Name of column containing commit counts
        threshold: Minimum commits to be considered "active" (default: 1)

    Returns:
        Dictionary with participation metrics
    """
    commits = pd.to_numeric(df[commit_column], errors='coerce').fillna(0).values

    total_contributors = len(commits)
    active_contributors = np.sum(commits > threshold)
    participation_rate = active_contributors / total_contributors if total_contributors > 0 else 0.0

    return {
        'total_contributors': total_contributors,
        'active_contributors': int(active_contributors),
        'participation_rate': round(participation_rate, 4)
    }


def process_single_project(input_file: Path, column: Optional[str] = None,
                          threshold: int = 1) -> Optional[dict]:
    """
    Process a single project file and compute participation rate.

    Args:
        input_file: Path to CSV file
        column: Optional column name (auto-detected if not provided)
        threshold: Minimum commits to be considered "active"

    Returns:
        Dictionary with project_name and participation metrics, or None on error
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

        # Compute participation rate
        metrics = compute_participation_rate(df, commit_column, threshold)

        # Extract project name
        project_name = input_file.stem.replace('_contributors_merged', '')

        return {
            'project_name': project_name,
            **metrics
        }

    except Exception as e:
        print(f"Warning: Failed to process {input_file.name}: {e}", file=sys.stderr)
        return None


def process_all_projects(input_dir: Path, column: Optional[str] = None,
                        threshold: int = 1) -> pd.DataFrame:
    """
    Process all *_contributors_merged.csv files in a directory.

    Args:
        input_dir: Directory containing CSV files
        column: Optional column name (auto-detected if not provided)
        threshold: Minimum commits to be considered "active"

    Returns:
        DataFrame with columns: project_name, total_contributors, active_contributors, participation_rate
    """
    csv_files = sorted(input_dir.glob('*_contributors_merged.csv'))

    if not csv_files:
        print(f"Error: No *_contributors_merged.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(csv_files)} projects...")
    results = []

    for idx, csv_file in enumerate(csv_files, 1):
        result = process_single_project(csv_file, column, threshold)
        if result:
            results.append(result)
            print(f"  [{idx}/{len(csv_files)}] {result['project_name']}: "
                  f"{result['active_contributors']}/{result['total_contributors']} "
                  f"({result['participation_rate']:.2%})")
        else:
            print(f"  [{idx}/{len(csv_files)}] {csv_file.stem}: Failed")

    return pd.DataFrame(results)


def plot_participation_rate(results_df: pd.DataFrame, output_dir: Path):
    """
    Create individual visualizations of participation rate distributions.

    Args:
        results_df: DataFrame with participation rate data
        output_dir: Directory to save the plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Histogram of participation rates
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(results_df['participation_rate'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results_df['participation_rate'].mean(), color='red',
               linestyle='--', linewidth=2,
               label=f"Mean: {results_df['participation_rate'].mean():.2%}")
    ax.axvline(results_df['participation_rate'].median(), color='orange',
               linestyle='--', linewidth=2,
               label=f"Median: {results_df['participation_rate'].median():.2%}")
    ax.set_xlabel('Participation Rate', fontsize=12)
    ax.set_ylabel('Frequency (Number of Projects)', fontsize=12)
    ax.set_title('Configuration Participation Rate Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'participation_rate_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 2: Box plot of participation rate
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([results_df['participation_rate']],
               tick_labels=['Participation Rate'])
    ax.set_ylabel('Participation Rate', fontsize=12)
    ax.set_title('Participation Rate Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'participation_rate_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 3: Scatter plot: Total contributors vs Active contributors
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(results_df['total_contributors'],
              results_df['active_contributors'],
              alpha=0.5, s=30)
    # Add diagonal line (perfect participation)
    max_val = max(results_df['total_contributors'].max(),
                  results_df['active_contributors'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2,
            label='Perfect Participation (100%)')
    ax.set_xlabel('Total Contributors (log scale)', fontsize=12)
    ax.set_ylabel('Active Contributors (> 1 commit, log scale)', fontsize=12)
    ax.set_title('Total vs Active Contributors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    output_path = output_dir / 'participation_total_vs_active.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 4: Scatter plot: Total contributors vs Participation rate
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(results_df['total_contributors'],
                        results_df['participation_rate'],
                        alpha=0.5, s=30, c=results_df['participation_rate'],
                        cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Participation Rate')
    ax.set_xlabel('Total Contributors (log scale)', fontsize=12)
    ax.set_ylabel('Participation Rate', fontsize=12)
    ax.set_title('Project Size vs Participation Rate', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'participation_size_vs_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute configuration participation rate for projects',
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
        '--threshold',
        type=int,
        default=0,
        help='Minimum commits to be considered active (default: 0, i.e., >= 1 commit)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all *_contributors_merged.csv files in the input directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../data/social/participation_rate_results.csv',
        help='Output CSV file for batch results (default: ../../data/social/participation_rate_results.csv)'
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
        results_df = process_all_projects(input_path, args.column, args.threshold)

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
        print(f"\nParticipation Rate (threshold > {args.threshold}):")
        print(f"  Mean:   {results_df['participation_rate'].mean():.2%}")
        print(f"  Median: {results_df['participation_rate'].median():.2%}")
        print(f"  Std:    {results_df['participation_rate'].std():.4f}")
        print(f"  Min:    {results_df['participation_rate'].min():.2%}")
        print(f"  Max:    {results_df['participation_rate'].max():.2%}")

        print(f"\nTotal Contributors:")
        print(f"  Mean:   {results_df['total_contributors'].mean():.1f}")
        print(f"  Median: {results_df['total_contributors'].median():.0f}")
        print(f"  Min:    {results_df['total_contributors'].min()}")
        print(f"  Max:    {results_df['total_contributors'].max()}")

        print(f"\nActive Contributors:")
        print(f"  Mean:   {results_df['active_contributors'].mean():.1f}")
        print(f"  Median: {results_df['active_contributors'].median():.0f}")
        print(f"  Min:    {results_df['active_contributors'].min()}")
        print(f"  Max:    {results_df['active_contributors'].max()}")
        print("=" * 60)

        # Create plots
        print("\nGenerating plots...")
        output_plot_dir = Path(args.plot_dir)
        plot_participation_rate(results_df, output_plot_dir)

        sys.exit(0)

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = process_single_project(input_path, args.column, args.threshold)

    if not result:
        print("Error: Failed to process file", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("\n" + "=" * 60)
        print("PARTICIPATION RATE RESULTS")
        print("=" * 60)
        print(f"Project: {result['project_name']}")
        print(f"Total contributors: {result['total_contributors']}")
        print(f"Active contributors (> {args.threshold}): {result['active_contributors']}")
        print(f"Participation rate: {result['participation_rate']:.2%}")
        print("=" * 60)
    else:
        print(f"{result['participation_rate']:.4f}")


if __name__ == '__main__':
    main()
