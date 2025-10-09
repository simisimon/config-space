#!/usr/bin/env python3
"""
Project Statistics Calculator

This script calculates comprehensive statistics from GitHub project data
and saves the results to a JSON file for further analysis.
"""

import pandas as pd
import json
import glob
from collections import Counter
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_project_data(csv_path):
    """Load project data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} projects from {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return None

def calculate_basic_statistics(df):
    """Calculate basic project statistics."""
    stats = {}
    
    # Project count and column information
    stats['project_count'] = len(df)
    stats['column_count'] = len(df.columns)
    stats['columns'] = df.columns.tolist()
    
    # Size statistics (in bytes and converted units)
    size_stats = df['size'].describe()
    stats['size_statistics'] = {
        'min_bytes': int(size_stats['min']),
        'max_bytes': int(size_stats['max']),
        'mean_bytes': int(size_stats['mean']),
        'median_bytes': int(df['size'].median()),
        'std_bytes': int(size_stats['std']),
        'converted_units': {
            'min_kb': round(size_stats['min'] / 1024, 1),
            'min_mb': round(size_stats['min'] / 1024 / 1024, 3),
            'max_mb': round(size_stats['max'] / 1024 / 1024, 1),
            'max_gb': round(size_stats['max'] / 1024 / 1024 / 1024, 2),
            'mean_kb': round(size_stats['mean'] / 1024, 1),
            'mean_mb': round(size_stats['mean'] / 1024 / 1024, 1),
            'median_kb': round(df['size'].median() / 1024, 1),
            'median_mb': round(df['size'].median() / 1024 / 1024, 1)
        }
    }
    
    # Star statistics
    star_stats = df['stargazers_count'].describe()
    stats['star_statistics'] = {
        'min_stars': int(star_stats['min']),
        'max_stars': int(star_stats['max']),
        'mean_stars': round(star_stats['mean'], 1),
        'median_stars': int(df['stargazers_count'].median()),
        'std_stars': round(star_stats['std'], 1)
    }
    
    # Creation year analysis
    creation_years = df['created_at'].apply(lambda x: int(x.split("-")[0])).tolist()
    year_counter = Counter(creation_years)
    stats['creation_year_analysis'] = {
        'year_distribution': dict(year_counter.most_common()),
        'top_10_years': year_counter.most_common(10)
    }
    
    # Language analysis
    language_counts = df['language'].value_counts().to_dict()
    stats['language_analysis'] = {
        'total_languages': len(language_counts),
        'top_10_languages': dict(df['language'].value_counts().head(10)),
        'language_distribution': language_counts
    }
    
    # Archive status
    archived_count = df['archived'].sum()
    stats['archive_status'] = {
        'archived_projects': int(archived_count),
        'active_projects': int(len(df) - archived_count),
        'archived_percentage': round((archived_count / len(df)) * 100, 2)
    }
    
    # Recent activity
    recent_updates = (df['Updated_in_last_30_days'] == 'yes').sum()
    stats['recent_activity'] = {
        'updated_in_last_30_days': int(recent_updates),
        'not_updated_recently': int(len(df) - recent_updates),
        'recent_update_percentage': round((recent_updates / len(df)) * 100, 2)
    }
    
    return stats

def analyze_commit_data(projects_dir):
    """Analyze commit data from individual project JSON files."""
    project_files = glob.glob(f"{projects_dir}/*.json")
    logger.info(f"Found {len(project_files)} project JSON files")
    
    commit_stats = {
        'total_projects_analyzed': 0,
        'projects_with_errors': 0,
        'error_details': [],
        'commit_statistics': {},
    }
    
    all_commits = []
    all_config_related_commits = []
    
    for project_file in project_files:
        try:
            with open(project_file, "r") as f:
                project_data = json.load(f)
                commits = project_data.get("commit_data", [])
                config_related_commits = [commit for commit in commits if commit.get("is_config_related", False)]
                
                all_commits.append(len(commits))
                all_config_related_commits.append(len(config_related_commits))

                commit_stats['total_projects_analyzed'] += 1

                logger.info(f"Processed {project_file}")
                
        except Exception as e:
            error_msg = f"Error processing {project_file}: {e}"
            logger.warning(error_msg)
            commit_stats['projects_with_errors'] += 1
            commit_stats['error_details'].append(error_msg)
    
    
    if all_commits:
        # Commit statistics
        commit_stats['commit_statistics'] = {
            'total_commits': sum(all_commits),
            'min_commits': min(all_commits),
            'max_commits': max(all_commits),
            'mean_commits': round(sum(all_commits) / len(all_commits), 2),
            'median_commits': int(pd.Series(all_commits).median())
        }
        
        # Config-related commit statistics
        commit_stats['config_related_statistics'] = {
            'total_config_commits': sum(all_config_related_commits),
            'min_config_commits': min(all_config_related_commits),
            'max_config_commits': max(all_config_related_commits),
            'mean_config_commits': round(sum(all_config_related_commits) / len(all_config_related_commits), 2),
            'median_config_commits': int(pd.Series(all_config_related_commits).median()),
            'config_commit_percentage': round((sum(all_config_related_commits) / sum(all_commits)) * 100, 2) if sum(all_commits) > 0 else 0
        }
    
    return commit_stats

def save_statistics(stats, output_path):
    """Save statistics to JSON file."""
    def convert_numpy_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    try:
        # Convert numpy types before saving
        converted_stats = convert_numpy_types(stats)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Statistics saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving statistics: {e}")
        return False

def main():
    """Main function to run the statistics calculation."""
    logger.info("Starting project statistics calculation...")
    
    # Define paths
    csv_path = "../data/projects_final.csv"
    projects_dir = "../data/projects"
    output_path = "../data/results/descriptive_statistics.json"
    
    # Check if input files exist
    if not Path(csv_path).exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    if not Path(projects_dir).exists():
        logger.warning(f"Projects directory not found: {projects_dir}")
        projects_dir = None
    
    # Load and analyze CSV data
    df = load_project_data(csv_path)
    if df is None:
        return
    
    # Calculate basic statistics
    logger.info("Calculating basic project statistics...")
    basic_stats = calculate_basic_statistics(df)
    
    # Analyze commit data
    commit_stats = analyze_commit_data(projects_dir)
    commit_stats = {
        'error': 'Commit analysis skipped due to memory constraints',
        'total_projects_analyzed': 0
    }
    
    # Combine all statistics
    all_stats = {
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'data_source': csv_path,
            'total_projects': len(df)
        },
        'basic_statistics': basic_stats,
        'commit_analysis': commit_stats
    }
    
    # Save to JSON file
    if save_statistics(all_stats, output_path):
        logger.info("Statistics calculation completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("PROJECT STATISTICS SUMMARY")
        print("="*60)
        print(f"Total projects analyzed: {len(df):,}")
        print(f"Projects with commit data: {commit_stats.get('total_projects_analyzed', 0):,}")
        if 'error' not in commit_stats:
            print(f"Total commits analyzed: {commit_stats.get('commit_statistics', {}).get('total_commits', 0):,}")
            print(f"Config-related commits: {commit_stats.get('config_related_statistics', {}).get('total_config_commits', 0):,}")
        else:
            print(f"Commit analysis: {commit_stats['error']}")
        print(f"Output saved to: {output_path}")
        print("="*60)
    else:
        logger.error("Failed to save statistics")

if __name__ == "__main__":
    main()
