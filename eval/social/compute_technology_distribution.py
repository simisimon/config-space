#!/usr/bin/env python3
"""
Technology Knowledge Distribution Calculator

Analyzes the distribution of technology knowledge across contributors in a project.
For each contributor, determines which technologies they touched based on their
config file contributions.

Creates two bar plots:

1. Technology Distribution (per contributor):
   - Y-axis: Number of technologies touched
   - X-axis: Number of contributors
   - Shows how technology knowledge is distributed (e.g., 5 contributors touched 3 technologies)

2. Technology Popularity (per technology):
   - Y-axis: Technology name
   - X-axis: Number of contributors who touched that technology
   - Shows which technologies are most/least commonly worked on

Usage:
    python compute_technology_distribution.py --input project_contributors.csv
    python compute_technology_distribution.py --input project_contributors.csv --output-dir plots/
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

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


def extract_contributor_technologies(df: pd.DataFrame, config_files_col: str) -> Dict[str, set]:
    """
    Extract technologies touched by each contributor who worked on configuration.

    Args:
        df: DataFrame with contributor data
        config_files_col: Name of the config files column

    Returns:
        Dictionary mapping contributor to set of technologies they touched
        (only includes contributors who actually touched config files)
    """
    contributor_technologies = {}

    for idx, row in df.iterrows():
        contributor = row.get('Contributor', row.get('contributor', f'Contributor_{idx}'))

        # Skip contributors with no config files
        if pd.isna(row[config_files_col]) or row[config_files_col] == '' or row[config_files_col] == '[]':
            continue

        file_tuples = parse_file_list(row[config_files_col])

        # Skip if parsing failed
        if not file_tuples:
            continue

        technologies = set()

        for file_path, count in file_tuples:
            tech = get_technology(file_path)
            if tech:
                technologies.add(tech)

        # Only add contributor if they touched at least one recognizable technology
        # (if they only touched unrecognized config files, they'll have 0 technologies)
        contributor_technologies[contributor] = technologies

    return contributor_technologies


def compute_technology_distribution(contributor_technologies: Dict[str, set]) -> Dict[int, int]:
    """
    Compute the distribution of technology knowledge.

    Args:
        contributor_technologies: Dictionary mapping contributor to set of technologies

    Returns:
        Dictionary mapping number of technologies to number of contributors
        (e.g., {1: 5, 2: 3} means 5 contributors touched 1 tech, 3 touched 2 techs)
    """
    distribution = {}

    for contributor, technologies in contributor_technologies.items():
        num_techs = len(technologies)
        distribution[num_techs] = distribution.get(num_techs, 0) + 1

    return distribution


def compute_technology_popularity(contributor_technologies: Dict[str, set]) -> Dict[str, int]:
    """
    Compute how many contributors touched each technology.

    Args:
        contributor_technologies: Dictionary mapping contributor to set of technologies

    Returns:
        Dictionary mapping technology to number of contributors who touched it
    """
    technology_contributors = {}

    for contributor, technologies in contributor_technologies.items():
        for tech in technologies:
            technology_contributors[tech] = technology_contributors.get(tech, 0) + 1

    return technology_contributors


def plot_technology_distribution(distribution: Dict[int, int], project_name: str,
                                 output_path: Path, config_contributors: int):
    """
    Create a bar plot of technology distribution.

    Args:
        distribution: Dictionary mapping num_technologies to num_contributors
        project_name: Name of the project
        output_path: Path to save the plot
        config_contributors: Number of contributors who worked on configuration
    """
    # Sort by number of technologies
    sorted_items = sorted(distribution.items())
    num_techs = [item[0] for item in sorted_items]
    num_contributors = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create horizontal bar plot (swapped axes)
    bars = ax.barh(num_techs, num_contributors, edgecolor='black', alpha=0.7)

    # Color bars based on number of technologies (gradient)
    colors = plt.cm.viridis([i / max(num_techs) if max(num_techs) > 0 else 0 for i in num_techs])
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_ylabel('Number of Technologies', fontsize=12)
    ax.set_xlabel('Number of Contributors', fontsize=12)
    ax.set_title(f'Technology Knowledge Distribution: {project_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (y, x) in enumerate(zip(num_techs, num_contributors)):
        ax.text(x + max(num_contributors) * 0.01, y, str(x),
                ha='left', va='center', fontsize=9)

    # Add summary statistics as text
    total_in_chart = sum(num_contributors)
    avg_techs = sum(k * v for k, v in distribution.items()) / total_in_chart if total_in_chart > 0 else 0

    summary_text = (
        f"Config contributors: {config_contributors}\n"
        f"Avg techs/contributor: {avg_techs:.2f}"
    )

    ax.text(0.98, 0.97, summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_technology_popularity(technology_contributors: Dict[str, int], project_name: str,
                               output_path: Path):
    """
    Create a bar plot showing how many contributors touched each technology.

    Args:
        technology_contributors: Dictionary mapping technology to number of contributors
        project_name: Name of the project
        output_path: Path to save the plot
    """
    # Sort by number of contributors (descending)
    sorted_items = sorted(technology_contributors.items(), key=lambda x: x[1], reverse=True)
    technologies = [item[0] for item in sorted_items]
    num_contributors = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(6, len(technologies) * 0.4)))

    # Create horizontal bar plot
    bars = ax.barh(technologies, num_contributors, edgecolor='black', alpha=0.7)

    # Color bars based on contributor count (gradient)
    max_contrib = max(num_contributors) if num_contributors else 1
    colors = plt.cm.YlOrRd([c / max_contrib for c in num_contributors])
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    ax.set_xlabel('Number of Contributors', fontsize=12)
    ax.set_ylabel('Technology', fontsize=12)
    ax.set_title(f'Technology Popularity: {project_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (tech, count) in enumerate(zip(technologies, num_contributors)):
        ax.text(count + max(num_contributors) * 0.01, i, str(count),
                ha='left', va='center', fontsize=9)

    # Add summary statistics
    total_techs = len(technologies)
    avg_contributors = sum(num_contributors) / total_techs if total_techs > 0 else 0

    summary_text = (
        f"Total technologies: {total_techs}\n"
        f"Avg contributors/tech: {avg_contributors:.2f}"
    )

    ax.text(0.98, 0.02, summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute technology knowledge distribution for a project',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to project contributors CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../data/social',
        help='Output directory for plot (default: ../../data/social)'
    )
    parser.add_argument(
        '--config-files-column',
        type=str,
        default='Config Files',
        help='Name of the config files column (default: Config Files)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Extract project name from filename
    project_name = input_path.stem.replace('_contributors_merged', '')

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Check if config files column exists
    if args.config_files_column not in df.columns:
        print(f"Error: Column '{args.config_files_column}' not found in CSV", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    total_contributors = len(df)

    print(f"Processing project: {project_name}")
    print(f"Total contributors in project: {total_contributors}")

    # Extract technologies for each contributor
    print("Extracting technologies from config files...")
    contributor_technologies = extract_contributor_technologies(df, args.config_files_column)

    config_contributors = len(contributor_technologies)
    non_config_contributors = total_contributors - config_contributors

    print(f"  Config contributors: {config_contributors}")
    print(f"  Non-config contributors: {non_config_contributors}")

    if config_contributors == 0:
        print("\nNo contributors worked on configuration files with recognized technologies.")
        sys.exit(0)

    # Compute distribution
    distribution = compute_technology_distribution(contributor_technologies)

    # Print results
    print("\n" + "=" * 60)
    print("TECHNOLOGY DISTRIBUTION (Config Contributors Only)")
    print("=" * 60)

    for num_techs in sorted(distribution.keys()):
        num_contribs = distribution[num_techs]
        percentage = (num_contribs / config_contributors) * 100
        print(f"Contributors with {num_techs} {'technology' if num_techs == 1 else 'technologies'}: "
              f"{num_contribs} ({percentage:.1f}%)")

    # Calculate statistics
    avg_techs = sum(k * v for k, v in distribution.items()) / config_contributors
    print(f"\nAverage technologies per config contributor: {avg_techs:.2f}")

    # Compute technology popularity (how many contributors per technology)
    technology_popularity = compute_technology_popularity(contributor_technologies)

    print(f"\nTotal unique technologies in project: {len(technology_popularity)}")
    print("\n" + "=" * 60)
    print("TECHNOLOGY POPULARITY (Contributors per Technology)")
    print("=" * 60)

    # Sort technologies by contributor count (descending)
    for tech, contrib_count in sorted(technology_popularity.items(), key=lambda x: x[1], reverse=True):
        percentage = (contrib_count / config_contributors) * 100
        print(f"  {tech}: {contrib_count} contributors ({percentage:.1f}%)")

    print("=" * 60)

    # Create plots
    output_dir = Path(args.output_dir)

    # Plot 1: Technology distribution (contributors by number of technologies)
    output_path = output_dir / f'{project_name}_technology_distribution.png'
    plot_technology_distribution(distribution, project_name, output_path, config_contributors)

    # Plot 2: Technology popularity (contributors per technology)
    output_path_popularity = output_dir / f'{project_name}_technology_popularity.png'
    plot_technology_popularity(technology_popularity, project_name, output_path_popularity)


if __name__ == '__main__':
    main()
