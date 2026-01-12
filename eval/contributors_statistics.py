#!/usr/bin/env python3
"""
Script to calculate contributor statistics from project contributor data.

This script analyzes contributor data from CSV files in data/projects_contributors/
and computes various statistics including total contributors, config-related contributors,
commit counts, and averages.
"""

import os
import csv
import ast
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Increase CSV field size limit to handle large config file lists
csv.field_size_limit(sys.maxsize)


def parse_config_files(config_files_str: str) -> List[str]:
    """Parse the config files string (which is a Python list representation)."""
    try:
        return ast.literal_eval(config_files_str)
    except:
        return []


def read_contributor_file(filepath: Path) -> Dict:
    """Read a single contributor CSV file and extract statistics."""
    stats = {
        'project_name': filepath.stem.replace('_contributors', ''),
        'total_contributors': 0,
        'config_contributors': 0,
        'non_config_contributors': 0,
        'total_config_commits': 0,
        'total_non_config_commits': 0,
        'total_commits': 0,
        'unique_config_files': set(),
        'contributors_data': []
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            contributor = row['Contributor']
            config_commits = int(row['Config Commits'])
            non_config_commits = int(row['Non-Config Commits'])
            avg_config_files = float(row['Avg Config Files Per Commit'])
            config_files = parse_config_files(row['Config Files'])

            stats['total_contributors'] += 1
            stats['total_config_commits'] += config_commits
            stats['total_non_config_commits'] += non_config_commits
            stats['total_commits'] += config_commits + non_config_commits

            if config_commits > 0:
                stats['config_contributors'] += 1
            else:
                stats['non_config_contributors'] += 1

            # Add unique config files
            stats['unique_config_files'].update(config_files)

            # Store individual contributor data
            stats['contributors_data'].append({
                'name': contributor,
                'config_commits': config_commits,
                'non_config_commits': non_config_commits,
                'total_commits': config_commits + non_config_commits,
                'avg_config_files': avg_config_files,
                'config_files': config_files
            })

    # Convert set to count
    stats['unique_config_files_count'] = len(stats['unique_config_files'])
    stats['unique_config_files'] = list(stats['unique_config_files'])

    return stats


def calculate_project_statistics(stats: Dict) -> Dict:
    """Calculate additional statistics for a project."""
    if stats['total_contributors'] == 0:
        return stats

    # Percentage of config contributors
    stats['config_contributors_percentage'] = (
        stats['config_contributors'] / stats['total_contributors'] * 100
    )

    # Average commits per contributor
    stats['avg_commits_per_contributor'] = (
        stats['total_commits'] / stats['total_contributors']
    )

    # Average config commits per contributor (including those with 0)
    stats['avg_config_commits_per_contributor'] = (
        stats['total_config_commits'] / stats['total_contributors']
    )

    # Average config commits per config contributor (only those with >0)
    if stats['config_contributors'] > 0:
        stats['avg_config_commits_per_config_contributor'] = (
            stats['total_config_commits'] / stats['config_contributors']
        )
    else:
        stats['avg_config_commits_per_config_contributor'] = 0

    # Percentage of config commits
    if stats['total_commits'] > 0:
        stats['config_commits_percentage'] = (
            stats['total_config_commits'] / stats['total_commits'] * 100
        )
    else:
        stats['config_commits_percentage'] = 0

    return stats


def calculate_aggregate_statistics(all_stats: List[Dict]) -> Dict:
    """Calculate aggregate statistics across all projects."""
    if not all_stats:
        return {}

    total_contributors_list = [s['total_contributors'] for s in all_stats]
    config_contributors_list = [s['config_contributors'] for s in all_stats]
    config_percentage_list = [s['config_contributors_percentage'] for s in all_stats]
    total_commits_list = [s['total_commits'] for s in all_stats]
    config_commits_list = [s['total_config_commits'] for s in all_stats]
    unique_config_files_list = [s['unique_config_files_count'] for s in all_stats]

    aggregate = {
        'total_projects': len(all_stats),
        'total_contributors_across_all_projects': sum(total_contributors_list),
        'total_config_contributors_across_all_projects': sum(config_contributors_list),
        'total_commits_across_all_projects': sum(total_commits_list),
        'total_config_commits_across_all_projects': sum(config_commits_list),

        # Averages
        'avg_contributors_per_project': statistics.mean(total_contributors_list),
        'median_contributors_per_project': statistics.median(total_contributors_list),
        'avg_config_contributors_per_project': statistics.mean(config_contributors_list),
        'median_config_contributors_per_project': statistics.median(config_contributors_list),
        'avg_config_contributor_percentage': statistics.mean(config_percentage_list),
        'median_config_contributor_percentage': statistics.median(config_percentage_list),
        'avg_unique_config_files_per_project': statistics.mean(unique_config_files_list),
        'median_unique_config_files_per_project': statistics.median(unique_config_files_list),

        # Min/Max
        'min_contributors': min(total_contributors_list),
        'max_contributors': max(total_contributors_list),
        'min_config_contributors': min(config_contributors_list),
        'max_config_contributors': max(config_contributors_list),
    }

    # Standard deviations
    if len(total_contributors_list) > 1:
        aggregate['stdev_contributors_per_project'] = statistics.stdev(total_contributors_list)
        aggregate['stdev_config_contributors_per_project'] = statistics.stdev(config_contributors_list)

    return aggregate


def main():
    """Main function to process all contributor files and generate statistics."""
    # Path to the contributors data directory
    data_dir = Path('../data/projects_contributors')

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return

    # Get all CSV files
    csv_files = sorted(data_dir.glob('*_contributors.csv'))

    if not csv_files:
        print(f"Error: No contributor CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} project contributor files")
    print("Processing...\n")

    # Process each file
    all_stats = []
    for filepath in csv_files:
        stats = read_contributor_file(filepath)
        stats = calculate_project_statistics(stats)
        all_stats.append(stats)

    # Calculate aggregate statistics
    aggregate_stats = calculate_aggregate_statistics(all_stats)

    # Print aggregate statistics
    print("=" * 80)
    print("AGGREGATE STATISTICS ACROSS ALL PROJECTS")
    print("=" * 80)
    print(f"\nTotal Projects Analyzed: {aggregate_stats['total_projects']}")
    print(f"\nContributor Statistics:")
    print(f"  Total contributors (across all projects): {aggregate_stats['total_contributors_across_all_projects']:,}")
    print(f"  Total config contributors (across all projects): {aggregate_stats['total_config_contributors_across_all_projects']:,}")
    print(f"  Average contributors per project: {aggregate_stats['avg_contributors_per_project']:.2f}")
    print(f"  Median contributors per project: {aggregate_stats['median_contributors_per_project']:.2f}")
    print(f"  Min contributors: {aggregate_stats['min_contributors']}")
    print(f"  Max contributors: {aggregate_stats['max_contributors']}")

    if 'stdev_contributors_per_project' in aggregate_stats:
        print(f"  Std dev contributors per project: {aggregate_stats['stdev_contributors_per_project']:.2f}")

    print(f"\nConfig Contributor Statistics:")
    print(f"  Average config contributors per project: {aggregate_stats['avg_config_contributors_per_project']:.2f}")
    print(f"  Median config contributors per project: {aggregate_stats['median_config_contributors_per_project']:.2f}")
    print(f"  Min config contributors: {aggregate_stats['min_config_contributors']}")
    print(f"  Max config contributors: {aggregate_stats['max_config_contributors']}")

    if 'stdev_config_contributors_per_project' in aggregate_stats:
        print(f"  Std dev config contributors per project: {aggregate_stats['stdev_config_contributors_per_project']:.2f}")

    print(f"\nConfig Contributor Percentage:")
    print(f"  Average percentage of config contributors: {aggregate_stats['avg_config_contributor_percentage']:.2f}%")
    print(f"  Median percentage of config contributors: {aggregate_stats['median_config_contributor_percentage']:.2f}%")

    print(f"\nCommit Statistics:")
    print(f"  Total commits (across all projects): {aggregate_stats['total_commits_across_all_projects']:,}")
    print(f"  Total config commits (across all projects): {aggregate_stats['total_config_commits_across_all_projects']:,}")

    print(f"\nConfig File Statistics:")
    print(f"  Average unique config files per project: {aggregate_stats['avg_unique_config_files_per_project']:.2f}")
    print(f"  Median unique config files per project: {aggregate_stats['median_unique_config_files_per_project']:.2f}")

    # Print top 10 projects by total contributors
    print("\n" + "=" * 80)
    print("TOP 10 PROJECTS BY TOTAL CONTRIBUTORS")
    print("=" * 80)
    sorted_by_contributors = sorted(all_stats, key=lambda x: x['total_contributors'], reverse=True)
    for i, stats in enumerate(sorted_by_contributors[:10], 1):
        print(f"{i:2}. {stats['project_name']:40} - {stats['total_contributors']:,} contributors " +
              f"({stats['config_contributors']:,} config, {stats['config_contributors_percentage']:.1f}%)")

    # Print top 10 projects by config contributors
    print("\n" + "=" * 80)
    print("TOP 10 PROJECTS BY CONFIG CONTRIBUTORS")
    print("=" * 80)
    sorted_by_config = sorted(all_stats, key=lambda x: x['config_contributors'], reverse=True)
    for i, stats in enumerate(sorted_by_config[:10], 1):
        print(f"{i:2}. {stats['project_name']:40} - {stats['config_contributors']:,} config contributors " +
              f"({stats['config_contributors_percentage']:.1f}% of {stats['total_contributors']:,})")

    # Print top 10 projects by config contributor percentage
    print("\n" + "=" * 80)
    print("TOP 10 PROJECTS BY CONFIG CONTRIBUTOR PERCENTAGE")
    print("=" * 80)
    # Filter projects with at least 10 contributors for meaningful percentages
    filtered_stats = [s for s in all_stats if s['total_contributors'] >= 10]
    sorted_by_percentage = sorted(filtered_stats, key=lambda x: x['config_contributors_percentage'], reverse=True)
    for i, stats in enumerate(sorted_by_percentage[:10], 1):
        print(f"{i:2}. {stats['project_name']:40} - {stats['config_contributors_percentage']:.1f}% " +
              f"({stats['config_contributors']:,}/{stats['total_contributors']:,})")

    # Print top 10 projects by unique config files
    print("\n" + "=" * 80)
    print("TOP 10 PROJECTS BY UNIQUE CONFIG FILES")
    print("=" * 80)
    sorted_by_config_files = sorted(all_stats, key=lambda x: x['unique_config_files_count'], reverse=True)
    for i, stats in enumerate(sorted_by_config_files[:10], 1):
        print(f"{i:2}. {stats['project_name']:40} - {stats['unique_config_files_count']:,} unique config files")

    # Save detailed statistics to CSV
    output_file = '../data/contributor_statistics.csv'
    print(f"\n{'=' * 80}")
    print(f"Saving detailed statistics to {output_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'project_name',
            'total_contributors',
            'config_contributors',
            'non_config_contributors',
            'config_contributors_percentage',
            'total_commits',
            'total_config_commits',
            'total_non_config_commits',
            'config_commits_percentage',
            'avg_commits_per_contributor',
            'avg_config_commits_per_contributor',
            'avg_config_commits_per_config_contributor',
            'unique_config_files_count'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in sorted(all_stats, key=lambda x: x['project_name']):
            writer.writerow({
                'project_name': stats['project_name'],
                'total_contributors': stats['total_contributors'],
                'config_contributors': stats['config_contributors'],
                'non_config_contributors': stats['non_config_contributors'],
                'config_contributors_percentage': f"{stats['config_contributors_percentage']:.2f}",
                'total_commits': stats['total_commits'],
                'total_config_commits': stats['total_config_commits'],
                'total_non_config_commits': stats['total_non_config_commits'],
                'config_commits_percentage': f"{stats['config_commits_percentage']:.2f}",
                'avg_commits_per_contributor': f"{stats['avg_commits_per_contributor']:.2f}",
                'avg_config_commits_per_contributor': f"{stats['avg_config_commits_per_contributor']:.2f}",
                'avg_config_commits_per_config_contributor': f"{stats['avg_config_commits_per_config_contributor']:.2f}",
                'unique_config_files_count': stats['unique_config_files_count']
            })

    print(f"Detailed statistics saved successfully!")

    # Save summary statistics to JSON
    summary_file = '../data/contributor_statistics_summary.json'
    print(f"\nSaving summary statistics to {summary_file}")

    # Prepare summary data
    summary = {
        'aggregate_statistics': aggregate_stats,
        'top_10_by_total_contributors': [
            {
                'rank': i,
                'project_name': stats['project_name'],
                'total_contributors': stats['total_contributors'],
                'config_contributors': stats['config_contributors'],
                'config_contributors_percentage': round(stats['config_contributors_percentage'], 2)
            }
            for i, stats in enumerate(sorted_by_contributors[:10], 1)
        ],
        'top_10_by_config_contributors': [
            {
                'rank': i,
                'project_name': stats['project_name'],
                'config_contributors': stats['config_contributors'],
                'total_contributors': stats['total_contributors'],
                'config_contributors_percentage': round(stats['config_contributors_percentage'], 2)
            }
            for i, stats in enumerate(sorted_by_config[:10], 1)
        ],
        'top_10_by_config_contributor_percentage': [
            {
                'rank': i,
                'project_name': stats['project_name'],
                'config_contributors_percentage': round(stats['config_contributors_percentage'], 2),
                'config_contributors': stats['config_contributors'],
                'total_contributors': stats['total_contributors']
            }
            for i, stats in enumerate(sorted_by_percentage[:10], 1)
        ],
        'top_10_by_unique_config_files': [
            {
                'rank': i,
                'project_name': stats['project_name'],
                'unique_config_files_count': stats['unique_config_files_count']
            }
            for i, stats in enumerate(sorted_by_config_files[:10], 1)
        ]
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary statistics saved successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
