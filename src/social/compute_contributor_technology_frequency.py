#!/usr/bin/env python3
"""
Contributor Technology Frequency Calculator

Computes how often each contributor works with each technology based on their
configuration file contributions. Outputs a matrix showing the count of
interactions between contributors and technologies.

Usage:
    # Single file
    python compute_contributor_technology_frequency.py --input project_contributors.csv

    # Batch processing
    python compute_contributor_technology_frequency.py --all --input ../../data/projects_contributors_merged
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import the technology mapping function
sys.path.insert(0, str(Path(__file__).parent.parent))
from mapping import get_technology


def parse_file_list(value, delimiter_regex: str = r'[;,|]') -> List[Tuple[str, int]]:
    """
    Parse a config files column value into a list of (file_path, count) tuples.

    Handles:
    - Python list of tuples: "[('file1.yml', 2), ('file2.json', 3)]"
    - Python list strings: "['file1.yml', 'file2.json']" (count defaults to 1)
    - Delimited strings: "file1.yml;file2.json" (count defaults to 1)
    - Empty values: [] or ""

    Returns:
        List of (file_path, count) tuples
    """
    if pd.isna(value) or value == '' or value == '[]':
        return []

    # Try to parse as Python literal (list)
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if not item:
                        continue
                    # Check if item is a tuple (file_path, count)
                    if isinstance(item, tuple) and len(item) == 2:
                        file_path, count = item
                        result.append((str(file_path), int(count)))
                    # Otherwise treat as just a file path with count=1
                    else:
                        result.append((str(item), 1))
                return result
        except (ValueError, SyntaxError):
            pass

    # Try delimiter-based splitting (count defaults to 1)
    if isinstance(value, str):
        files = re.split(delimiter_regex, value.strip())
        return [(f.strip().strip("'\""), 1) for f in files if f.strip()]

    return []


def compute_contributor_technology_frequency(
    df: pd.DataFrame, config_files_col: str
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Compute how often each contributor works with each technology.

    Args:
        df: DataFrame with contributor data
        config_files_col: Name of the column containing config files

    Returns:
        Tuple of:
        - DataFrame with contributor-technology frequency matrix
        - Dictionary mapping contributor -> {technology -> count}
    """
    contributor_tech_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for idx, row in df.iterrows():
        contributor = row.get('Contributor', row.get('contributor', f'Contributor_{idx}'))

        # Skip contributors with no config files
        if pd.isna(row[config_files_col]) or row[config_files_col] == '' or row[config_files_col] == '[]':
            continue

        file_tuples = parse_file_list(row[config_files_col])

        if not file_tuples:
            continue

        for file_path, count in file_tuples:
            tech = get_technology(file_path)
            if tech:
                contributor_tech_counts[contributor][tech] += count

    # Convert to DataFrame (contributor-technology matrix)
    if not contributor_tech_counts:
        return pd.DataFrame(), {}

    # Get all unique technologies
    all_technologies = set()
    for tech_counts in contributor_tech_counts.values():
        all_technologies.update(tech_counts.keys())

    # Build matrix
    rows = []
    for contributor, tech_counts in sorted(contributor_tech_counts.items()):
        row = {'Contributor': contributor}
        for tech in sorted(all_technologies):
            row[tech] = tech_counts.get(tech, 0)
        rows.append(row)

    frequency_df = pd.DataFrame(rows)
    frequency_df = frequency_df.set_index('Contributor')

    return frequency_df, dict(contributor_tech_counts)


def compute_contributor_shares(df: pd.DataFrame) -> pd.Series:
    """
    Compute the share of config commits for each contributor.

    Args:
        df: DataFrame with contributor data

    Returns:
        Series mapping contributor name to their share (0-100)
    """
    # Try to find config commits column
    config_col = None
    for col in ['Config Commits', 'config_commits', 'Commits', 'commits']:
        if col in df.columns:
            config_col = col
            break

    if not config_col:
        return pd.Series(dtype=float)

    # Get contributor column
    contributor_col = None
    for col in ['Contributor', 'contributor', 'Author', 'author']:
        if col in df.columns:
            contributor_col = col
            break

    if not contributor_col:
        return pd.Series(dtype=float)

    commits = pd.to_numeric(df[config_col], errors='coerce').fillna(0)
    total = commits.sum()

    if total == 0:
        return pd.Series(dtype=float)

    shares = (commits / total * 100)
    return pd.Series(shares.values, index=df[contributor_col])


def process_single_project(
    input_file: Path, config_files_col: str = 'Config Files'
) -> Optional[Tuple[str, pd.DataFrame, Dict[str, Dict[str, int]], pd.Series]]:
    """
    Process a single project file.

    Args:
        input_file: Path to CSV file
        config_files_col: Name of the config files column

    Returns:
        Tuple of (project_name, frequency_df, contributor_tech_counts, contributor_shares) or None on error
    """
    try:
        df = pd.read_csv(input_file)

        if config_files_col not in df.columns:
            print(f"Warning: Column '{config_files_col}' not found in {input_file.name}", file=sys.stderr)
            return None

        frequency_df, contributor_tech_counts = compute_contributor_technology_frequency(
            df, config_files_col
        )

        contributor_shares = compute_contributor_shares(df)

        project_name = input_file.stem.replace('_contributors_merged', '')

        return project_name, frequency_df, contributor_tech_counts, contributor_shares

    except Exception as e:
        print(f"Warning: Failed to process {input_file.name}: {e}", file=sys.stderr)
        return None


def process_all_projects(
    input_dir: Path, config_files_col: str = 'Config Files'
) -> pd.DataFrame:
    """
    Process all *_contributors_merged.csv files in a directory.

    Aggregates results across all projects.

    Args:
        input_dir: Directory containing CSV files
        config_files_col: Name of the config files column

    Returns:
        DataFrame with aggregated contributor-technology frequencies across all projects
    """
    csv_files = sorted(input_dir.glob('*_contributors_merged.csv'))

    if not csv_files:
        print(f"Error: No *_contributors_merged.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(csv_files)} projects...")

    all_results = []

    for idx, csv_file in enumerate(csv_files, 1):
        result = process_single_project(csv_file, config_files_col)
        if result:
            project_name, frequency_df, _, _ = result
            if not frequency_df.empty:
                # Add project column
                frequency_df = frequency_df.reset_index()
                frequency_df['Project'] = project_name
                all_results.append(frequency_df)
                print(f"  [{idx}/{len(csv_files)}] {project_name}: "
                      f"{len(frequency_df)} contributors, "
                      f"{len(frequency_df.columns) - 2} technologies")
            else:
                print(f"  [{idx}/{len(csv_files)}] {project_name}: No data")
        else:
            print(f"  [{idx}/{len(csv_files)}] {csv_file.stem}: Failed")

    if not all_results:
        print("Error: No projects successfully processed", file=sys.stderr)
        sys.exit(1)

    # Concatenate all results
    combined_df = pd.concat(all_results, ignore_index=True)

    return combined_df


def plot_heatmap(frequency_df: pd.DataFrame, project_name: str, output_path: Path,
                 contributor_shares: pd.Series = None,
                 max_contributors: int = 50, max_technologies: int = 30):
    """
    Create a heatmap visualization of contributor-technology frequency.

    Args:
        frequency_df: DataFrame with contributor-technology matrix
        project_name: Name of the project
        output_path: Path to save the plot
        contributor_shares: Series mapping contributor to their config commit share (for sorting)
        max_contributors: Maximum contributors to show (top by share)
        max_technologies: Maximum technologies to show (top by total)
    """
    if frequency_df.empty:
        print("Warning: Empty frequency matrix, skipping heatmap", file=sys.stderr)
        return

    # Work with a copy
    df = frequency_df.copy()

    # Calculate totals for filtering
    technology_totals = df.sum(axis=0)

    # Sort contributors by their config commit share if available
    if contributor_shares is not None and not contributor_shares.empty:
        # Get contributors that exist in both frequency_df and shares
        common_contributors = df.index.intersection(contributor_shares.index)
        shares_for_sorting = contributor_shares.loc[common_contributors].sort_values(ascending=False)
        top_contributors = shares_for_sorting.head(max_contributors).index
    else:
        # Fallback to sorting by total interactions
        contributor_totals = df.sum(axis=1)
        top_contributors = contributor_totals.nlargest(max_contributors).index

    top_technologies = technology_totals.nlargest(max_technologies).index

    df_filtered = df.loc[top_contributors, top_technologies]

    # Sort contributors by share (descending), technologies by total (descending)
    if contributor_shares is not None and not contributor_shares.empty:
        # Sort by share
        sorted_contributors = contributor_shares.loc[df_filtered.index].sort_values(ascending=False).index
        df_filtered = df_filtered.loc[sorted_contributors]
    else:
        df_filtered = df_filtered.loc[df_filtered.sum(axis=1).sort_values(ascending=False).index]

    df_filtered = df_filtered[df_filtered.sum(axis=0).sort_values(ascending=False).index]

    # Create heatmap
    fig_height = max(8, len(df_filtered) * 0.3)
    fig_width = max(10, len(df_filtered.columns) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        df_filtered,
        annot=True if len(df_filtered) <= 20 and len(df_filtered.columns) <= 15 else False,
        fmt='d',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Frequency'}
    )

    ax.set_title(f'Contributor-Technology Frequency: {project_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Technology', fontsize=12)
    ax.set_ylabel('Contributor (sorted by config commit share)', fontsize=12)

    # Update y-tick labels to show share percentage and number of technologies
    new_labels = []
    for contributor in df_filtered.index:
        # Count number of technologies this contributor touches (non-zero columns)
        num_techs = (df_filtered.loc[contributor] > 0).sum()
        # Truncate long names
        display_name = contributor[:25] + '...' if len(contributor) > 25 else contributor
        if contributor_shares is not None and not contributor_shares.empty:
            share = contributor_shares.get(contributor, 0)
            new_labels.append(f'{display_name} ({share:.1f}%, {num_techs} techs)')
        else:
            new_labels.append(f'{display_name} ({num_techs} techs)')
    ax.set_yticklabels(new_labels)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    plt.close()


def print_summary(frequency_df: pd.DataFrame, project_name: str):
    """Print summary statistics for the frequency matrix."""
    if frequency_df.empty:
        print("No data to summarize.")
        return

    print("\n" + "=" * 60)
    print(f"CONTRIBUTOR-TECHNOLOGY FREQUENCY: {project_name}")
    print("=" * 60)

    print(f"\nTotal contributors: {len(frequency_df)}")
    print(f"Total technologies: {len(frequency_df.columns)}")

    # Technology totals (most popular)
    tech_totals = frequency_df.sum(axis=0).sort_values(ascending=False)
    print("\nTop 10 Technologies (by total interactions):")
    for tech, count in tech_totals.head(10).items():
        print(f"  {tech}: {count}")

    # Contributor totals (most active)
    contributor_totals = frequency_df.sum(axis=1).sort_values(ascending=False)
    print("\nTop 10 Contributors (by total interactions):")
    for contributor, count in contributor_totals.head(10).items():
        # Truncate long contributor names
        display_name = contributor[:40] + '...' if len(contributor) > 40 else contributor
        print(f"  {display_name}: {count}")

    # Average interactions
    print(f"\nAverage interactions per contributor: {contributor_totals.mean():.2f}")
    print(f"Average contributors per technology: {tech_totals.mean():.2f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Compute contributor-technology frequency matrix',
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
        '--config-files-column',
        type=str,
        default='Config Files',
        help='Name of the config files column (default: Config Files)'
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
        help='Output CSV file for frequency matrix'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: <input_parent>/social)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating heatmap plot'
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

    if args.output_dir is None:
        args.output_dir = str(_social_dir)

    output_dir = Path(args.output_dir)

    # Batch processing mode
    if args.all:
        if not input_path.is_dir():
            print(f"Error: --all requires input to be a directory, got: {input_path}", file=sys.stderr)
            sys.exit(1)

        combined_df = process_all_projects(input_path, args.config_files_column)

        # Save combined results
        output_csv = Path(args.output) if args.output else output_dir / 'contributor_technology_frequency_all.csv'
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"\nCombined results saved to: {output_csv}")

        # Print aggregated summary
        print("\n" + "=" * 60)
        print("AGGREGATE SUMMARY")
        print("=" * 60)
        print(f"Total projects: {combined_df['Project'].nunique()}")
        print(f"Total contributor-project pairs: {len(combined_df)}")

        # Technology columns (exclude Contributor and Project)
        tech_cols = [c for c in combined_df.columns if c not in ['Contributor', 'Project']]
        tech_totals = combined_df[tech_cols].sum().sort_values(ascending=False)
        print(f"Total technologies: {len(tech_cols)}")
        print("\nTop 10 Technologies (across all projects):")
        for tech, count in tech_totals.head(10).items():
            print(f"  {tech}: {count}")
        print("=" * 60)

        sys.exit(0)

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = process_single_project(input_path, args.config_files_column)

    if not result:
        print("Error: Failed to process file", file=sys.stderr)
        sys.exit(1)

    project_name, frequency_df, contributor_tech_counts, contributor_shares = result

    if frequency_df.empty:
        print("No contributor-technology data found.")
        sys.exit(0)

    # Print summary
    if args.verbose:
        print_summary(frequency_df, project_name)

    # Save CSV
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = output_dir / f'{project_name}_contributor_technology_frequency.csv'

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frequency_df.to_csv(output_csv)
    print(f"Frequency matrix saved to: {output_csv}")

    # Generate heatmap
    if not args.no_plot:
        output_plot = output_dir / f'{project_name}_contributor_technology_heatmap.png'
        plot_heatmap(frequency_df, project_name, output_plot, contributor_shares)


if __name__ == '__main__':
    main()
