#!/usr/bin/env python3
"""
Cross-Project Contributor Identifier

Identifies contributors who work across different projects based on their
configuration file contributions. Outputs a CSV with authors working on
multiple projects, including project count, average commits, and technologies.

Usage:
    python compute_cross_project_contributors.py --input ../../data/contributors/merged
    python compute_cross_project_contributors.py --input ../../data/contributors/merged --limit 50
"""

import argparse
import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from mapping import get_technology


def parse_file_list(value) -> List[Tuple[str, int]]:
    """
    Parse a config files column value into a list of (file_path, count) tuples.
    """
    if pd.isna(value) or value == '' or value == '[]':
        return []

    if isinstance(value, str) and value.strip().startswith('['):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if not item:
                        continue
                    if isinstance(item, tuple) and len(item) == 2:
                        file_path, count = item
                        result.append((str(file_path), int(count)))
                    else:
                        result.append((str(item), 1))
                return result
        except (ValueError, SyntaxError):
            pass

    return []


def extract_email(contributor: str) -> str:
    """
    Extract email from contributor string like 'Name <email@example.com>'.
    Returns the email if found, otherwise returns the original string normalized.
    """
    match = re.search(r'<([^>]+)>', contributor)
    if match:
        return match.group(1).lower().strip()
    return contributor.lower().strip()


def get_technologies_from_files(file_list: List[Tuple[str, int]]) -> Set[str]:
    """Extract unique technologies from a list of config files."""
    technologies = set()
    for file_path, _ in file_list:
        tech = get_technology(file_path)
        if tech:
            technologies.add(tech)
    return technologies


def process_projects(
    input_dir: Path,
    limit: int = None
) -> Dict[str, Dict]:
    """
    Process contributor data from all projects and aggregate by contributor email.

    Args:
        input_dir: Directory containing *_contributors_merged.csv files
        limit: Maximum number of projects to process (None for all)

    Returns:
        Dictionary mapping contributor email to their aggregated data
    """
    csv_files = sorted(input_dir.glob('*_contributors_merged.csv'))

    if not csv_files:
        print(f"Error: No *_contributors_merged.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    if limit:
        csv_files = csv_files[:limit]

    print(f"Processing {len(csv_files)} projects...")

    # Structure: email -> {projects: set, total_commits: int, technologies: set, project_commits: list}
    contributors: Dict[str, Dict] = defaultdict(lambda: {
        'projects': set(),
        'total_commits': 0,
        'technologies': set(),
        'project_commits': []
    })

    for idx, csv_file in enumerate(csv_files, 1):
        project_name = csv_file.stem.replace('_contributors_merged', '')

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  [{idx}/{len(csv_files)}] {project_name}: Failed to read - {e}", file=sys.stderr)
            continue

        config_col = None
        for col in ['Config Commits', 'config_commits']:
            if col in df.columns:
                config_col = col
                break

        files_col = None
        for col in ['Config Files', 'config_files']:
            if col in df.columns:
                files_col = col
                break

        contributor_col = None
        for col in ['Contributor', 'contributor']:
            if col in df.columns:
                contributor_col = col
                break

        if not contributor_col:
            print(f"  [{idx}/{len(csv_files)}] {project_name}: No contributor column found", file=sys.stderr)
            continue

        project_contributors = 0
        for _, row in df.iterrows():
            contributor_name = row[contributor_col]
            email = extract_email(contributor_name)

            config_commits = 0
            if config_col and pd.notna(row[config_col]):
                config_commits = int(row[config_col])

            if config_commits == 0:
                continue

            contributors[email]['projects'].add(project_name)
            contributors[email]['total_commits'] += config_commits
            contributors[email]['project_commits'].append(config_commits)
            contributors[email]['name'] = contributor_name

            if files_col and pd.notna(row[files_col]):
                file_list = parse_file_list(row[files_col])
                technologies = get_technologies_from_files(file_list)
                contributors[email]['technologies'].update(technologies)

            project_contributors += 1

        print(f"  [{idx}/{len(csv_files)}] {project_name}: {project_contributors} config contributors")

    return contributors


def compute_cross_project_contributors(
    contributors: Dict[str, Dict],
    min_projects: int = 2
) -> pd.DataFrame:
    """
    Filter and format contributors who work across multiple projects.

    Args:
        contributors: Dictionary of contributor data from process_projects
        min_projects: Minimum number of projects to be included

    Returns:
        DataFrame with cross-project contributor statistics
    """
    rows = []

    for email, data in contributors.items():
        num_projects = len(data['projects'])

        if num_projects < min_projects:
            continue

        avg_commits = data['total_commits'] / num_projects
        technologies = sorted(data['technologies'])

        rows.append({
            'Author': data.get('name', email),
            'Email': email,
            'Number of Projects': num_projects,
            'Total Config Commits': data['total_commits'],
            'Avg Config Commits per Project': round(avg_commits, 2),
            'Technologies': technologies,
            'Projects': sorted(data['projects'])
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values('Number of Projects', ascending=False)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Identify contributors working across multiple projects',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to directory containing *_contributors_merged.csv files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: ../../data/social/cross_project_contributors.csv)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of projects to process'
    )
    parser.add_argument(
        '--min-projects',
        type=int,
        default=2,
        help='Minimum number of projects to include a contributor (default: 2)'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.is_dir():
        print(f"Error: Input must be a directory, got: {input_path}", file=sys.stderr)
        sys.exit(1)

    contributors = process_projects(input_path, args.limit)

    df = compute_cross_project_contributors(contributors, args.min_projects)

    if df.empty:
        print("\nNo contributors found working across multiple projects.")
        sys.exit(0)

    output_path = Path(args.output) if args.output else Path(__file__).parent.parent.parent / 'data' / 'social' / 'cross_project_contributors.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("CROSS-PROJECT CONTRIBUTOR SUMMARY")
    print("=" * 60)
    print(f"Total cross-project contributors: {len(df)}")
    print(f"Max projects by single contributor: {df['Number of Projects'].max()}")
    print(f"Avg projects per contributor: {df['Number of Projects'].mean():.2f}")

    print("\nTop 10 Cross-Project Contributors:")
    for _, row in df.head(10).iterrows():
        name = row['Author'][:40] + '...' if len(row['Author']) > 40 else row['Author']
        print(f"  {name}: {row['Number of Projects']} projects, "
              f"{row['Avg Config Commits per Project']:.1f} avg commits, "
              f"{len(row['Technologies'])} technologies")

    print("=" * 60)


if __name__ == '__main__':
    main()
