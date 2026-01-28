#!/usr/bin/env python3
"""
Configuration Contributor Share Calculator

Calculates the percentage share of config-commits for each contributor and
visualizes them as stacked bar plots.

Usage:
    # Single file
    python compute_contributor_shares.py --input contributors.csv

    # Batch processing (all projects)
    python compute_contributor_shares.py --all --input ../../data/projects_contributors_merged
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

    for col in df.columns:
        col_lower = col.lower()
        if 'config' in col_lower and 'commit' in col_lower:
            return col
        if col_lower == 'commits':
            return col

    return None


def detect_contributor_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect the contributor name column in the DataFrame.

    Args:
        df: DataFrame to search

    Returns:
        Column name if found, None otherwise
    """
    candidates = [
        'Contributor',
        'contributor',
        'Author',
        'author',
        'Name',
        'name'
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def compute_contributor_shares(df: pd.DataFrame, commit_column: str,
                                contributor_column: str) -> pd.DataFrame:
    """
    Compute the percentage share of config-commits for each contributor.

    Args:
        df: DataFrame with contributor data
        commit_column: Name of column containing commit counts
        contributor_column: Name of column containing contributor names

    Returns:
        DataFrame with contributor shares sorted by share descending
    """
    commits = pd.to_numeric(df[commit_column], errors='coerce').fillna(0)
    total_commits = commits.sum()

    if total_commits == 0:
        return pd.DataFrame(columns=['contributor', 'commits', 'share'])

    result = pd.DataFrame({
        'contributor': df[contributor_column],
        'commits': commits,
        'share': commits / total_commits * 100
    })

    result = result.sort_values('share', ascending=False).reset_index(drop=True)
    return result


def process_single_project(input_file: Path, column: Optional[str] = None,
                           top_n: int = 10) -> Optional[dict]:
    """
    Process a single project file and compute contributor shares.

    Args:
        input_file: Path to CSV file
        column: Optional column name (auto-detected if not provided)
        top_n: Number of top contributors to include (rest grouped as "Others")

    Returns:
        Dictionary with project_name and contributor shares, or None on error
    """
    try:
        df = pd.read_csv(input_file)

        if column:
            commit_column = column
            if commit_column not in df.columns:
                return None
        else:
            commit_column = detect_commit_column(df)
            if not commit_column:
                return None

        contributor_column = detect_contributor_column(df)
        if not contributor_column:
            return None

        shares_df = compute_contributor_shares(df, commit_column, contributor_column)

        if len(shares_df) == 0:
            return None

        # Count contributors with config commits > 0 (before grouping)
        num_config_contributors = (shares_df['commits'] > 0).sum()
        num_total_contributors = len(df)

        # Group contributors beyond top_n as "Others"
        if len(shares_df) > top_n:
            top_contributors = shares_df.head(top_n).copy()
            others_share = shares_df.iloc[top_n:]['share'].sum()
            others_commits = shares_df.iloc[top_n:]['commits'].sum()
            others_row = pd.DataFrame({
                'contributor': ['Others'],
                'commits': [others_commits],
                'share': [others_share]
            })
            shares_df = pd.concat([top_contributors, others_row], ignore_index=True)

        project_name = input_file.stem.replace('_contributors_merged', '')

        return {
            'project_name': project_name,
            'shares': shares_df,
            'total_commits': shares_df['commits'].sum(),
            'num_config_contributors': num_config_contributors,
            'num_total_contributors': num_total_contributors
        }

    except Exception as e:
        print(f"Warning: Failed to process {input_file.name}: {e}", file=sys.stderr)
        return None


def process_all_projects(input_dir: Path, column: Optional[str] = None,
                         top_n: int = 5, limit: Optional[int] = None) -> list:
    """
    Process all *_contributors_merged.csv files in a directory.

    Args:
        input_dir: Directory containing CSV files
        column: Optional column name (auto-detected if not provided)
        top_n: Number of top contributors per project
        limit: Maximum number of projects to process (None for all)

    Returns:
        List of dictionaries with project data
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
        result = process_single_project(csv_file, column, top_n)
        if result:
            results.append(result)
            top_share = result['shares'].iloc[0]['share']
            print(f"  [{idx}/{len(csv_files)}] {result['project_name']}: "
                  f"Top contributor: {top_share:.1f}%")
        else:
            print(f"  [{idx}/{len(csv_files)}] {csv_file.stem}: Failed")

    return results


def plot_single_project(result: dict, output_dir: Path):
    """
    Create a stacked bar plot for a single project.

    Args:
        result: Dictionary with project shares data
        output_dir: Directory to save the plot
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    shares_df = result['shares']
    project_name = result['project_name']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create stacked horizontal bar
    left = 0
    colors = plt.cm.tab20(np.linspace(0, 1, len(shares_df)))

    bars = []
    for idx, row in shares_df.iterrows():
        bar = ax.barh(0, row['share'], left=left, color=colors[idx],
                      edgecolor='white', linewidth=0.5)
        bars.append(bar)
        left += row['share']

    # Add legend
    legend_labels = [f"{row['contributor'][:20]}... ({row['share']:.1f}%)"
                     if len(row['contributor']) > 20
                     else f"{row['contributor']} ({row['share']:.1f}%)"
                     for _, row in shares_df.iterrows()]
    ax.legend([b[0] for b in bars], legend_labels,
              loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of Config Commits (%)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'Configuration Commit Distribution: {project_name}\n'
                 f'Contributors: ({result["num_config_contributors"]}/{result["num_total_contributors"]})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'{project_name}_contributor_shares.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()


def plot_all_projects(results: list, output_dir: Path, top_n_projects: int = 30):
    """
    Create a stacked bar plot comparing all projects.

    Args:
        results: List of project result dictionaries
        output_dir: Directory to save the plot
        top_n_projects: Maximum number of projects to show
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by dominance (top contributor share) and limit
    results = sorted(results, key=lambda x: x['shares'].iloc[0]['share'], reverse=True)
    if len(results) > top_n_projects:
        results = results[:top_n_projects]

    # Collect all unique contributors across projects for consistent coloring
    all_contributors = set()
    for r in results:
        for _, row in r['shares'].iterrows():
            all_contributors.add(row['contributor'])
    all_contributors = sorted(all_contributors)

    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(all_contributors))))
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(all_contributors)}
    # "Others" gets a gray color
    color_map['Others'] = (0.7, 0.7, 0.7, 1.0)

    fig, ax = plt.subplots(figsize=(14, max(8, len(results) * 0.4)))

    y_positions = np.arange(len(results))
    project_names = [r['project_name'] for r in results]

    for idx, result in enumerate(results):
        left = 0
        shares_df = result['shares']

        for _, row in shares_df.iterrows():
            contributor = row['contributor']
            share = row['share']
            color = color_map.get(contributor, (0.5, 0.5, 0.5, 1.0))

            ax.barh(idx, share, left=left, color=color,
                    edgecolor='white', linewidth=0.3, height=0.8)

            # Add percentage label if share is large enough
            if share >= 10:
                ax.text(left + share/2, idx, f'{share:.0f}%',
                        ha='center', va='center', fontsize=7, color='white',
                        fontweight='bold')

            left += share

        # Add contributor count at the end of the bar (config/total)
        ax.text(101, idx, f'({result["num_config_contributors"]}/{result["num_total_contributors"]})',
                ha='left', va='center', fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(project_names, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of Config Commits (%)', fontsize=12)
    ax.set_ylabel('Project', fontsize=12)
    ax.set_title('Configuration Commit Distribution by Contributor',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    output_path = output_dir / 'all_projects_contributor_shares.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()

    # Also create a summary plot showing share distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get top-1, top-3, top-5 shares for each project
    top1_shares = []
    top3_shares = []
    top5_shares = []

    for r in results:
        shares = r['shares']['share'].values
        top1_shares.append(shares[0] if len(shares) >= 1 else 0)
        top3_shares.append(sum(shares[:3]) if len(shares) >= 3 else sum(shares))
        top5_shares.append(sum(shares[:5]) if len(shares) >= 5 else sum(shares))

    x = np.arange(len(results))
    width = 0.25

    ax.bar(x - width, top1_shares, width, label='Top 1', color='#2ecc71')
    ax.bar(x, top3_shares, width, label='Top 3', color='#3498db')
    ax.bar(x + width, top5_shares, width, label='Top 5', color='#9b59b6')

    ax.set_xlabel('Project', fontsize=12)
    ax.set_ylabel('Cumulative Share (%)', fontsize=12)
    ax.set_title('Cumulative Contributor Share by Project', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(project_names, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'cumulative_contributor_shares.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute and visualize configuration commit shares by contributor',
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
        '--top-n',
        type=int,
        default=10,
        help='Number of top contributors to show individually (default: 10)'
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
        help='Output CSV file for results (default: <input_parent>/social/contributor_shares.csv)'
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
        args.output = str(_social_dir / 'contributor_shares.csv')
    if args.plot_dir is None:
        args.plot_dir = str(_social_dir)

    output_dir = Path(args.plot_dir)

    # Batch processing mode
    if args.all:
        if not input_path.is_dir():
            print(f"Error: --all requires input to be a directory, got: {input_path}", file=sys.stderr)
            sys.exit(1)

        results = process_all_projects(input_path, args.column, args.top_n, args.limit)

        if len(results) == 0:
            print("Error: No projects successfully processed", file=sys.stderr)
            sys.exit(1)

        # Save summary to CSV
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        summary_data = []
        for r in results:
            shares = r['shares']['share'].values
            summary_data.append({
                'project_name': r['project_name'],
                'num_config_contributors': r['num_config_contributors'],
                'num_total_contributors': r['num_total_contributors'],
                'total_commits': r['total_commits'],
                'top1_share': shares[0] if len(shares) >= 1 else 0,
                'top3_share': sum(shares[:3]) if len(shares) >= 3 else sum(shares),
                'top5_share': sum(shares[:5]) if len(shares) >= 5 else sum(shares),
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_csv, index=False)
        print(f"\nSummary saved to: {output_csv}")

        # Print statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total projects: {len(results)}")
        print(f"\nTop-1 Share:")
        print(f"  Mean:   {summary_df['top1_share'].mean():.1f}%")
        print(f"  Median: {summary_df['top1_share'].median():.1f}%")
        print(f"\nTop-3 Share:")
        print(f"  Mean:   {summary_df['top3_share'].mean():.1f}%")
        print(f"  Median: {summary_df['top3_share'].median():.1f}%")
        print(f"\nTop-5 Share:")
        print(f"  Mean:   {summary_df['top5_share'].mean():.1f}%")
        print(f"  Median: {summary_df['top5_share'].median():.1f}%")
        print("=" * 60)

        # Generate plots
        print("\nGenerating plots...")
        plot_all_projects(results, output_dir)

        sys.exit(0)

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = process_single_project(input_path, args.column, args.top_n)

    if not result:
        print("Error: Failed to process file", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print("\n" + "=" * 60)
        print(f"CONTRIBUTOR SHARES: {result['project_name']}")
        print("=" * 60)
        print(f"Total config commits: {result['total_commits']:.0f}")
        print(f"Contributors: {result['num_config_contributors']} config / {result['num_total_contributors']} total")
        print("\nShare breakdown:")
        for _, row in result['shares'].iterrows():
            print(f"  {row['contributor'][:40]:40s} {row['share']:6.2f}% ({row['commits']:.0f} commits)")
        print("=" * 60)
    else:
        for _, row in result['shares'].iterrows():
            print(f"{row['share']:.2f}%\t{row['contributor']}")

    # Generate plot
    print("\nGenerating plot...")
    plot_single_project(result, output_dir)


if __name__ == '__main__':
    main()
