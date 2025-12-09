import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, List
import pandas as pd
import numpy as np
from tqdm import tqdm
from mapping import get_technology

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_gini_coefficient(values: List[float]) -> float:
    """
    Compute Gini coefficient for a list of values.

    Args:
        values: List of numeric values (e.g., commit counts)

    Returns:
        Gini coefficient between 0 (perfect equality) and 1 (perfect inequality)
    """
    if len(values) == 0:
        return 0.0

    if len(values) == 1:
        return 1.0

    # Sort values
    sorted_values = np.sort(np.array(values))
    n = len(sorted_values)

    # Compute Gini coefficient
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

    return gini


def compute_effective_contributors(proportions: List[float]) -> float:
    """
    Compute effective number of contributors (inverse Herfindahl index).

    Args:
        proportions: List of proportions (should sum to 1.0)

    Returns:
        Effective number of contributors
    """
    if len(proportions) == 0:
        return 0.0

    herfindahl = sum(p ** 2 for p in proportions)
    return 1.0 / herfindahl if herfindahl > 0 else 0.0


def analyze_technology_concentration(df_contributors: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze concentration of responsibility per technology.

    Args:
        df_contributors: DataFrame with contributor statistics including Config Files column

    Returns:
        DataFrame with technology-level concentration metrics
    """
    logger.info("Analyzing technology concentration...")

    # Build technology -> contributor -> commit count mapping
    tech_contributors = defaultdict(lambda: defaultdict(float))

    for _, row in df_contributors.iterrows():
        contributor = row['Contributor']
        config_commits = row['Config Commits']
        config_files = row['Config Files']

        if not isinstance(config_files, list):
            continue

        # Map files to technologies
        tech_files = defaultdict(int)
        for file_path in config_files:
            tech = get_technology(file_path)
            if tech:
                tech_files[tech] += 1

        # Distribute commits proportionally to technologies based on file count
        total_files = sum(tech_files.values())
        if total_files == 0:
            continue

        for tech, file_count in tech_files.items():
            # Proportional allocation of commits based on files touched
            tech_commits = config_commits * (file_count / total_files)
            tech_contributors[tech][contributor] += tech_commits

    # Compute metrics for each technology
    results = []
    for tech, contributors in tech_contributors.items():
        commit_counts = list(contributors.values())
        total_commits = sum(commit_counts)
        num_contributors = len(contributors)

        # Compute proportions
        proportions = [count / total_commits for count in commit_counts]

        # Compute metrics
        gini = compute_gini_coefficient(commit_counts)
        enc = compute_effective_contributors(proportions)

        # Find top contributor
        top_contributor = max(contributors.items(), key=lambda x: x[1])
        top_contributor_pct = (top_contributor[1] / total_commits) * 100

        # Assess risk level
        if top_contributor_pct >= 80 or num_contributors == 1:
            risk_level = 'High'
        elif top_contributor_pct >= 60 or enc < 2:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        results.append({
            'Technology': tech,
            'Total Commits': round(total_commits, 2),
            'Num Contributors': num_contributors,
            'Gini Coefficient': round(gini, 3),
            'Effective Num Contributors': round(enc, 2),
            'Top Contributor': top_contributor[0],
            'Top Contributor %': round(top_contributor_pct, 2),
            'Risk Level': risk_level
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Gini Coefficient', ascending=False)

    logger.info(f"Analyzed {len(results)} technologies")
    return df_results


def analyze_contributor_specialization(df_contributors: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze each contributor's technology specialization.

    Args:
        df_contributors: DataFrame with contributor statistics including Config Files column

    Returns:
        DataFrame with contributor-level specialization metrics
    """
    logger.info("Analyzing contributor specialization...")

    results = []

    for _, row in df_contributors.iterrows():
        contributor = row['Contributor']
        config_commits = row['Config Commits']
        config_files = row['Config Files']

        if not isinstance(config_files, list) or config_commits == 0:
            continue

        # Map files to technologies
        tech_files = defaultdict(int)
        for file_path in config_files:
            tech = get_technology(file_path)
            if tech:
                tech_files[tech] += 1

        if len(tech_files) == 0:
            continue

        # Find primary technology
        primary_tech = max(tech_files.items(), key=lambda x: x[1])
        primary_tech_name = primary_tech[0]
        primary_tech_files = primary_tech[1]
        total_files = sum(tech_files.values())

        # Compute Technology Isolation Index
        tii = primary_tech_files / total_files if total_files > 0 else 0

        # Determine if siloed
        is_siloed = tii > 0.8

        results.append({
            'Contributor': contributor,
            'Primary Technology': primary_tech_name,
            'Primary Tech File Count': primary_tech_files,
            'Total Config Files': total_files,
            'Num Technologies': len(tech_files),
            'Technology Isolation Index': round(tii, 3),
            'Is Siloed': is_siloed,
            'Technologies': ', '.join(sorted(tech_files.keys()))
        })

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Technology Isolation Index', ascending=False)

    logger.info(f"Analyzed {len(results)} contributors")
    return df_results


def generate_summary_statistics(df_tech_concentration: pd.DataFrame,
                                df_contributor_specialization: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for the analysis.

    Args:
        df_tech_concentration: Technology concentration analysis results
        df_contributor_specialization: Contributor specialization analysis results

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'total_technologies': len(df_tech_concentration),
        'high_risk_technologies': len(df_tech_concentration[df_tech_concentration['Risk Level'] == 'High']),
        'medium_risk_technologies': len(df_tech_concentration[df_tech_concentration['Risk Level'] == 'Medium']),
        'low_risk_technologies': len(df_tech_concentration[df_tech_concentration['Risk Level'] == 'Low']),
        'orphaned_technologies': len(df_tech_concentration[df_tech_concentration['Num Contributors'] == 1]),
        'avg_gini_coefficient': float(df_tech_concentration['Gini Coefficient'].mean()) if len(df_tech_concentration) > 0 else 0.0,
        'avg_effective_contributors': float(df_tech_concentration['Effective Num Contributors'].mean()) if len(df_tech_concentration) > 0 else 0.0,
        'total_contributors': len(df_contributor_specialization),
        'siloed_contributors': len(df_contributor_specialization[df_contributor_specialization['Is Siloed']]),
        'siloed_contributor_percentage': (len(df_contributor_specialization[df_contributor_specialization['Is Siloed']]) /
                                          len(df_contributor_specialization) * 100) if len(df_contributor_specialization) > 0 else 0,
        'avg_technologies_per_contributor': float(df_contributor_specialization['Num Technologies'].mean()) if len(df_contributor_specialization) > 0 else 0.0,
        'avg_isolation_index': float(df_contributor_specialization['Technology Isolation Index'].mean()) if len(df_contributor_specialization) > 0 else 0.0
    }

    return summary


def analyze_project(project_name: str, contributors_file: str, output_dir: str):
    """
    Analyze technology responsibility for a single project.

    Args:
        project_name: Name of the project
        contributors_file: Path to contributors CSV file
        output_dir: Directory to save output files
    """
    logger.info(f"Analyzing project: {project_name}")

    # Load contributor data
    df_contributors = pd.read_csv(contributors_file)

    # Parse Config Files column if it's a string
    if df_contributors['Config Files'].dtype == object:
        df_contributors['Config Files'] = df_contributors['Config Files'].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

    # Analyze technology concentration
    df_tech_concentration = analyze_technology_concentration(df_contributors)

    # Analyze contributor specialization
    df_contributor_specialization = analyze_contributor_specialization(df_contributors)

    # Generate summary statistics
    summary = generate_summary_statistics(df_tech_concentration, df_contributor_specialization)

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    tech_output = os.path.join(output_dir, f"{project_name}_technology_concentration.csv")
    df_tech_concentration.to_csv(tech_output, index=False)
    logger.info(f"Saved technology concentration analysis to {tech_output}")

    contributor_output = os.path.join(output_dir, f"{project_name}_contributor_specialization.csv")
    df_contributor_specialization.to_csv(contributor_output, index=False)
    logger.info(f"Saved contributor specialization analysis to {contributor_output}")

    summary_output = os.path.join(output_dir, f"{project_name}_summary.json")
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary statistics to {summary_output}")

    # Log key findings
    if summary['total_technologies'] > 0:
        logger.info(f"\n{'='*60}")
        logger.info(f"Summary for {project_name}:")
        logger.info(f"{'='*60}")
        logger.info(f"Total technologies: {summary['total_technologies']}")
        logger.info(f"High-risk technologies: {summary['high_risk_technologies']} ({summary['high_risk_technologies']/summary['total_technologies']*100:.1f}%)")
        logger.info(f"Orphaned technologies (1 contributor): {summary['orphaned_technologies']}")
        logger.info(f"Average Gini coefficient: {summary['avg_gini_coefficient']:.3f}")
        logger.info(f"Average effective contributors per technology: {summary['avg_effective_contributors']:.2f}")
        logger.info(f"\nContributor insights:")
        logger.info(f"Total contributors: {summary['total_contributors']}")
        logger.info(f"Siloed contributors: {summary['siloed_contributors']} ({summary['siloed_contributor_percentage']:.1f}%)")
        logger.info(f"Average technologies per contributor: {summary['avg_technologies_per_contributor']:.2f}")
        logger.info(f"Average isolation index: {summary['avg_isolation_index']:.3f}")
        logger.info(f"{'='*60}\n")


def analyze_all_projects(contributors_dir: str, output_dir: str):
    """
    Analyze technology responsibility for all projects in the contributors directory.

    Args:
        contributors_dir: Directory containing aggregated contributor CSV files
        output_dir: Directory to save output files
    """
    csv_files = [f for f in os.listdir(contributors_dir) if f.endswith('.csv')]

    logger.info(f"Found {len(csv_files)} projects to analyze")

    for csv_file in tqdm(csv_files, desc="Analyzing projects"):
        # Handle both 'aggregated_<project>_contributors.csv' and '<project>_contributors.csv' formats
        if csv_file.startswith('aggregated_'):
            project_name = csv_file.replace('aggregated_', '').replace('_contributors.csv', '')
        else:
            project_name = csv_file.replace('_contributors.csv', '')

        contributors_file = os.path.join(contributors_dir, csv_file)

        try:
            analyze_project(project_name, contributors_file, output_dir)
        except Exception as e:
            logger.error(f"Failed to analyze {project_name}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze technology-specific configuration responsibility and identify knowledge islands"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Analyze a specific project (provide project name without _contributors.csv suffix)"
    )
    parser.add_argument(
        "--contributors-dir",
        type=str,
        default="data/projects_contributors_aggregated/",
        help="Directory containing aggregated contributor CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/technology_responsibility/",
        help="Directory to save analysis results"
    )

    args = parser.parse_args()

    if args.project:
        # Analyze single project - try aggregated file first, then fall back to regular file
        contributors_file = os.path.join(args.contributors_dir, f"aggregated_{args.project}_contributors.csv")
        if not os.path.exists(contributors_file):
            contributors_file = os.path.join(args.contributors_dir, f"{args.project}_contributors.csv")

        if not os.path.exists(contributors_file):
            logger.error(f"Contributors file not found: {contributors_file}")
            exit(1)

        analyze_project(args.project, contributors_file, args.output_dir)
    else:
        # Analyze all projects
        analyze_all_projects(args.contributors_dir, args.output_dir)
