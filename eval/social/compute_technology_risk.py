#!/usr/bin/env python3
"""
Technology Risk Calculator - Orphaned and Endangered Technologies

Identifies technologies at risk due to limited contributor coverage:

1. Orphaned Technologies: Technologies changed by only 1 contributor
   - These technologies have single points of failure (bus factor = 1)

2. Endangered Technologies: Technologies with 2-3 contributors where one
   contributor has >80% of commits for that technology
   - These have nominally more contributors but still concentrated knowledge

Usage:
    # Single file
    python compute_technology_risk.py --input project_contributors.csv

    # Batch processing
    python compute_technology_risk.py --all --input ../../data/projects_contributors_merged
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def extract_technology_contributors(df: pd.DataFrame, config_files_col: str) -> Dict[str, Dict[str, int]]:
    """
    Extract commit counts per technology per contributor.

    Args:
        df: DataFrame with contributor data
        config_files_col: Name of the config files column

    Returns:
        Dictionary mapping technology -> {contributor: commit_count}
    """
    technology_contributors: Dict[str, Dict[str, int]] = {}

    for idx, row in df.iterrows():
        contributor = row.get('Contributor', row.get('contributor', f'Contributor_{idx}'))

        # Skip contributors with no config files
        if pd.isna(row[config_files_col]) or row[config_files_col] == '' or row[config_files_col] == '[]':
            continue

        file_tuples = parse_file_list(row[config_files_col])

        # Skip if parsing failed
        if not file_tuples:
            continue

        for file_path, count in file_tuples:
            tech = get_technology(file_path)
            if tech:
                if tech not in technology_contributors:
                    technology_contributors[tech] = {}
                if contributor not in technology_contributors[tech]:
                    technology_contributors[tech][contributor] = 0
                technology_contributors[tech][contributor] += count

    return technology_contributors


def identify_orphaned_technologies(technology_contributors: Dict[str, Dict[str, int]]) -> List[dict]:
    """
    Identify technologies with only one contributor.

    Args:
        technology_contributors: Dictionary mapping technology -> {contributor: commit_count}

    Returns:
        List of dictionaries with orphaned technology details
    """
    orphaned = []

    for tech, contributors in technology_contributors.items():
        if len(contributors) == 1:
            contributor, commits = list(contributors.items())[0]
            orphaned.append({
                'technology': tech,
                'contributor': contributor,
                'commits': commits,
                'num_contributors': 1
            })

    return sorted(orphaned, key=lambda x: x['commits'], reverse=True)


def identify_endangered_technologies(technology_contributors: Dict[str, Dict[str, int]],
                                      dominance_threshold: float = 0.80) -> List[dict]:
    """
    Identify technologies with 2-3 contributors where one has >80% of commits.

    Args:
        technology_contributors: Dictionary mapping technology -> {contributor: commit_count}
        dominance_threshold: Threshold for dominance (default 0.80 = 80%)

    Returns:
        List of dictionaries with endangered technology details
    """
    endangered = []

    for tech, contributors in technology_contributors.items():
        num_contributors = len(contributors)

        # Only consider technologies with 2-3 contributors
        if num_contributors < 2 or num_contributors > 3:
            continue

        total_commits = sum(contributors.values())
        if total_commits == 0:
            continue

        # Find top contributor
        top_contributor = max(contributors.items(), key=lambda x: x[1])
        top_contributor_name, top_contributor_commits = top_contributor
        dominance = top_contributor_commits / total_commits

        if dominance > dominance_threshold:
            endangered.append({
                'technology': tech,
                'num_contributors': num_contributors,
                'total_commits': total_commits,
                'top_contributor': top_contributor_name,
                'top_contributor_commits': top_contributor_commits,
                'dominance': round(dominance, 4),
                'all_contributors': dict(contributors)
            })

    return sorted(endangered, key=lambda x: x['dominance'], reverse=True)


def process_single_project(input_file: Path, config_files_col: str = 'Config Files',
                           dominance_threshold: float = 0.80) -> Optional[dict]:
    """
    Process a single project file and identify orphaned/endangered technologies.

    Args:
        input_file: Path to CSV file
        config_files_col: Name of the config files column
        dominance_threshold: Threshold for endangered classification

    Returns:
        Dictionary with project analysis results, or None on error
    """
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Warning: Failed to read {input_file.name}: {e}", file=sys.stderr)
        return None

    if config_files_col not in df.columns:
        print(f"Warning: Column '{config_files_col}' not found in {input_file.name}", file=sys.stderr)
        return None

    # Extract project name
    project_name = input_file.stem.replace('_contributors_merged', '')

    # Get technology-contributor mapping
    tech_contributors = extract_technology_contributors(df, config_files_col)

    if not tech_contributors:
        return {
            'project_name': project_name,
            'total_technologies': 0,
            'orphaned_count': 0,
            'endangered_count': 0,
            'orphaned_technologies': [],
            'endangered_technologies': []
        }

    # Identify at-risk technologies
    orphaned = identify_orphaned_technologies(tech_contributors)
    endangered = identify_endangered_technologies(tech_contributors, dominance_threshold)

    return {
        'project_name': project_name,
        'total_technologies': len(tech_contributors),
        'orphaned_count': len(orphaned),
        'endangered_count': len(endangered),
        'orphaned_technologies': orphaned,
        'endangered_technologies': endangered,
        'orphaned_rate': len(orphaned) / len(tech_contributors) if tech_contributors else 0,
        'endangered_rate': len(endangered) / len(tech_contributors) if tech_contributors else 0,
        'at_risk_rate': (len(orphaned) + len(endangered)) / len(tech_contributors) if tech_contributors else 0
    }


def process_all_projects(input_dir: Path, config_files_col: str = 'Config Files',
                         dominance_threshold: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process all *_contributors_merged.csv files in a directory.

    Args:
        input_dir: Directory containing CSV files
        config_files_col: Name of the config files column
        dominance_threshold: Threshold for endangered classification

    Returns:
        Tuple of:
        - DataFrame with summary results for all projects
        - DataFrame with per-technology risk summary across all projects
    """
    csv_files = sorted(input_dir.glob('*_contributors_merged.csv'))

    if not csv_files:
        print(f"Error: No *_contributors_merged.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(csv_files)} projects...")
    results = []

    # Track technology-level statistics across all projects
    tech_stats: Dict[str, Dict[str, int]] = {}  # tech -> {orphaned, endangered, total_projects}

    for idx, csv_file in enumerate(csv_files, 1):
        result = process_single_project(csv_file, config_files_col, dominance_threshold)
        if result:
            results.append({
                'project_name': result['project_name'],
                'total_technologies': result['total_technologies'],
                'orphaned_count': result['orphaned_count'],
                'endangered_count': result['endangered_count'],
                'orphaned_rate': round(result.get('orphaned_rate', 0), 4),
                'endangered_rate': round(result.get('endangered_rate', 0), 4),
                'at_risk_rate': round(result.get('at_risk_rate', 0), 4)
            })

            # Collect orphaned technologies
            for tech_info in result['orphaned_technologies']:
                tech = tech_info['technology']
                if tech not in tech_stats:
                    tech_stats[tech] = {'orphaned': 0, 'endangered': 0, 'total_projects': 0}
                tech_stats[tech]['orphaned'] += 1

            # Collect endangered technologies
            for tech_info in result['endangered_technologies']:
                tech = tech_info['technology']
                if tech not in tech_stats:
                    tech_stats[tech] = {'orphaned': 0, 'endangered': 0, 'total_projects': 0}
                tech_stats[tech]['endangered'] += 1

            # Count total projects per technology (need to re-extract)
            try:
                df = pd.read_csv(csv_file)
                tech_contributors = extract_technology_contributors(df, config_files_col)
                for tech in tech_contributors.keys():
                    if tech not in tech_stats:
                        tech_stats[tech] = {'orphaned': 0, 'endangered': 0, 'total_projects': 0}
                    tech_stats[tech]['total_projects'] += 1
            except Exception:
                pass

            print(f"  [{idx}/{len(csv_files)}] {result['project_name']}: "
                  f"{result['orphaned_count']} orphaned, "
                  f"{result['endangered_count']} endangered "
                  f"(of {result['total_technologies']} technologies)")
        else:
            print(f"  [{idx}/{len(csv_files)}] {csv_file.stem}: Failed")

    # Build technology summary DataFrame
    tech_summary_data = []
    for tech, stats in tech_stats.items():
        total = stats['total_projects']
        tech_summary_data.append({
            'technology': tech,
            'total_projects': total,
            'orphaned_count': stats['orphaned'],
            'endangered_count': stats['endangered'],
            'at_risk_count': stats['orphaned'] + stats['endangered'],
            'orphaned_rate': round(stats['orphaned'] / total, 4) if total > 0 else 0,
            'endangered_rate': round(stats['endangered'] / total, 4) if total > 0 else 0,
            'at_risk_rate': round((stats['orphaned'] + stats['endangered']) / total, 4) if total > 0 else 0
        })

    tech_summary_df = pd.DataFrame(tech_summary_data)
    if not tech_summary_df.empty:
        tech_summary_df = tech_summary_df.sort_values('at_risk_count', ascending=False)

    return pd.DataFrame(results), tech_summary_df


def plot_technology_risk(results_df: pd.DataFrame, output_dir: Path):
    """
    Create visualizations of technology risk distributions.

    Args:
        results_df: DataFrame with risk analysis results
        output_dir: Directory to save the plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to projects with at least 1 technology
    df = results_df[results_df['total_technologies'] > 0].copy()

    if len(df) == 0:
        print("No projects with technologies to plot", file=sys.stderr)
        return

    # Plot 1: Distribution of orphaned technology rate
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['orphaned_rate'], bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(df['orphaned_rate'].mean(), color='red',
               linestyle='--', linewidth=2,
               label=f"Mean: {df['orphaned_rate'].mean():.1%}")
    ax.axvline(df['orphaned_rate'].median(), color='orange',
               linestyle='--', linewidth=2,
               label=f"Median: {df['orphaned_rate'].median():.1%}")
    ax.set_xlabel('Orphaned Technology Rate', fontsize=12)
    ax.set_ylabel('Number of Projects', fontsize=12)
    ax.set_title('Distribution of Orphaned Technology Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'orphaned_rate_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 2: Distribution of endangered technology rate
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['endangered_rate'], bins=20, edgecolor='black', alpha=0.7, color='gold')
    ax.axvline(df['endangered_rate'].mean(), color='red',
               linestyle='--', linewidth=2,
               label=f"Mean: {df['endangered_rate'].mean():.1%}")
    ax.axvline(df['endangered_rate'].median(), color='orange',
               linestyle='--', linewidth=2,
               label=f"Median: {df['endangered_rate'].median():.1%}")
    ax.set_xlabel('Endangered Technology Rate', fontsize=12)
    ax.set_ylabel('Number of Projects', fontsize=12)
    ax.set_title('Distribution of Endangered Technology Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'endangered_rate_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 3: Combined at-risk rate distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df['at_risk_rate'], bins=20, edgecolor='black', alpha=0.7, color='tomato')
    ax.axvline(df['at_risk_rate'].mean(), color='red',
               linestyle='--', linewidth=2,
               label=f"Mean: {df['at_risk_rate'].mean():.1%}")
    ax.axvline(df['at_risk_rate'].median(), color='orange',
               linestyle='--', linewidth=2,
               label=f"Median: {df['at_risk_rate'].median():.1%}")
    ax.set_xlabel('At-Risk Technology Rate (Orphaned + Endangered)', fontsize=12)
    ax.set_ylabel('Number of Projects', fontsize=12)
    ax.set_title('Distribution of At-Risk Technology Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'at_risk_rate_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 4: Scatter plot - Total technologies vs At-risk count
    fig, ax = plt.subplots(figsize=(8, 6))
    at_risk_count = df['orphaned_count'] + df['endangered_count']
    scatter = ax.scatter(df['total_technologies'], at_risk_count,
                         alpha=0.5, s=30, c=df['at_risk_rate'], cmap='YlOrRd')
    plt.colorbar(scatter, ax=ax, label='At-Risk Rate')
    ax.set_xlabel('Total Technologies', fontsize=12)
    ax.set_ylabel('At-Risk Technologies (Orphaned + Endangered)', fontsize=12)
    ax.set_title('Project Size vs At-Risk Technologies', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    # Add diagonal reference line for 100% at-risk
    max_val = max(df['total_technologies'].max(), at_risk_count.max())
    ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5, label='100% at-risk')
    ax.legend()
    plt.tight_layout()
    output_path = output_dir / 'technology_risk_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()

    # Plot 5: Stacked bar showing orphaned vs endangered breakdown
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist([df['orphaned_rate'], df['endangered_rate']],
            bins=20, edgecolor='black', alpha=0.7,
            label=['Orphaned', 'Endangered'], stacked=False)
    ax.set_xlabel('Rate', fontsize=12)
    ax.set_ylabel('Number of Projects', fontsize=12)
    ax.set_title('Orphaned vs Endangered Technology Rates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    output_path = output_dir / 'orphaned_vs_endangered_histogram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {output_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Identify orphaned and endangered technologies in projects',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Definitions:
  Orphaned: Technologies changed by only 1 contributor (bus factor = 1)
  Endangered: Technologies with 2-3 contributors where one has >80%% of commits

Examples:
  # Single project analysis
  python compute_technology_risk.py --input project_contributors.csv

  # Batch processing all projects
  python compute_technology_risk.py --all --input ../../data/projects_contributors_merged
"""
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to CSV file or directory containing contributor data'
    )
    parser.add_argument(
        '--config-files-column',
        type=str,
        default='Config Files',
        help='Name of the config files column (default: Config Files)'
    )
    parser.add_argument(
        '--dominance-threshold',
        type=float,
        default=0.80,
        help='Dominance threshold for endangered classification (default: 0.80)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all *_contributors_merged.csv files in the input directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../../data/social/technology_risk_results.csv',
        help='Output CSV file for batch results (default: ../../data/social/technology_risk_results.csv)'
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
        help='Print detailed output including technology lists'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    # Batch processing mode
    if args.all:
        if not input_path.is_dir():
            print(f"Error: --all requires input to be a directory, got: {input_path}", file=sys.stderr)
            sys.exit(1)

        # Process all projects
        results_df, tech_summary_df = process_all_projects(input_path, args.config_files_column, args.dominance_threshold)

        if len(results_df) == 0:
            print("Error: No projects successfully processed", file=sys.stderr)
            sys.exit(1)

        # Save results to CSV
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

        # Save technology summary to CSV
        tech_summary_csv = output_csv.parent / 'technology_risk_summary.csv'
        tech_summary_df.to_csv(tech_summary_csv, index=False)
        print(f"Technology summary saved to: {tech_summary_csv}")

        # Print summary statistics
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Total projects analyzed: {len(results_df)}")

        # Filter to projects with technologies
        with_tech = results_df[results_df['total_technologies'] > 0]
        print(f"Projects with technologies: {len(with_tech)}")

        if len(with_tech) > 0:
            print(f"\nOrphaned Technologies (1 contributor):")
            print(f"  Total orphaned across all projects: {with_tech['orphaned_count'].sum()}")
            print(f"  Mean orphaned rate:   {with_tech['orphaned_rate'].mean():.1%}")
            print(f"  Median orphaned rate: {with_tech['orphaned_rate'].median():.1%}")
            print(f"  Projects with orphaned: {(with_tech['orphaned_count'] > 0).sum()} ({(with_tech['orphaned_count'] > 0).mean():.1%})")

            print(f"\nEndangered Technologies (2-3 contributors, >80% dominance):")
            print(f"  Total endangered across all projects: {with_tech['endangered_count'].sum()}")
            print(f"  Mean endangered rate:   {with_tech['endangered_rate'].mean():.1%}")
            print(f"  Median endangered rate: {with_tech['endangered_rate'].median():.1%}")
            print(f"  Projects with endangered: {(with_tech['endangered_count'] > 0).sum()} ({(with_tech['endangered_count'] > 0).mean():.1%})")

            print(f"\nCombined At-Risk (Orphaned + Endangered):")
            print(f"  Mean at-risk rate:   {with_tech['at_risk_rate'].mean():.1%}")
            print(f"  Median at-risk rate: {with_tech['at_risk_rate'].median():.1%}")

        print("=" * 70)

        # Create plots
        print("\nGenerating plots...")
        output_plot_dir = Path(args.plot_dir)
        plot_technology_risk(results_df, output_plot_dir)

        sys.exit(0)

    # Single file mode
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = process_single_project(input_path, args.config_files_column, args.dominance_threshold)

    if not result:
        print("Error: Failed to process file", file=sys.stderr)
        sys.exit(1)

    # Print results
    print("\n" + "=" * 70)
    print(f"TECHNOLOGY RISK ANALYSIS: {result['project_name']}")
    print("=" * 70)
    print(f"Total technologies: {result['total_technologies']}")

    print(f"\n--- ORPHANED TECHNOLOGIES ({result['orphaned_count']}) ---")
    print("(Technologies with only 1 contributor - bus factor = 1)")
    if result['orphaned_technologies']:
        for tech in result['orphaned_technologies']:
            print(f"  {tech['technology']}: {tech['commits']} commits by {tech['contributor'][:50]}")
    else:
        print("  None found")

    print(f"\n--- ENDANGERED TECHNOLOGIES ({result['endangered_count']}) ---")
    print(f"(Technologies with 2-3 contributors, one with >{args.dominance_threshold:.0%} of commits)")
    if result['endangered_technologies']:
        for tech in result['endangered_technologies']:
            print(f"  {tech['technology']}: {tech['num_contributors']} contributors, "
                  f"{tech['dominance']:.1%} dominance ({tech['top_contributor_commits']}/{tech['total_commits']} commits)")
            if args.verbose:
                for contrib, commits in sorted(tech['all_contributors'].items(), key=lambda x: -x[1]):
                    print(f"    - {contrib[:50]}: {commits} commits")
    else:
        print("  None found")

    if result['total_technologies'] > 0:
        print(f"\n--- SUMMARY ---")
        print(f"Orphaned rate:   {result['orphaned_rate']:.1%}")
        print(f"Endangered rate: {result['endangered_rate']:.1%}")
        print(f"At-risk rate:    {result['at_risk_rate']:.1%}")

    print("=" * 70)


if __name__ == '__main__':
    main()
