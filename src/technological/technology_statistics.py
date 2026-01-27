import logging
import glob
import json
import pandas as pd
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mapping import get_technology

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True, stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_technology_statistics(project_files):
    """
    Calculate number of unique option names per technology for all projects.
    Returns a DataFrame with projects as rows and technologies as columns.
    Each cell contains the number of unique option names for that technology in that project.
    """
    logger.info("Calculating technology statistics...")

    # Exclude file format types, not actual technologies
    excluded_formats = {'yaml', 'json', 'xml', 'toml', 'configparser'}

    # Dictionary to store project -> technology -> set of option names
    project_tech_options = {}
    # Dictionary to store project -> technology -> count of config files
    project_tech_file_counts = {}
    all_technologies = set()

    for project_file in project_files:
        try:
            with open(project_file, 'r') as f:
                data = json.load(f)

            project_name = data.get('project_name', os.path.basename(project_file).replace('_commit.json', ''))

            # Get concepts and config data - support both old and new formats
            # New format: config_data directly at top level
            # Old format: latest_commit_data.network_data
            if 'config_data' in data:
                config_data = data.get('config_data', {})
            else:
                latest_commit = data.get('latest_commit_data', {})
                config_data = latest_commit.get('network_data', {})
            config_file_data = config_data.get('config_file_data', [])

            if project_name not in project_tech_options:
                project_tech_options[project_name] = {}
                project_tech_file_counts[project_name] = {}

            # Process each config file
            for config_file in config_file_data:
                technology = config_file.get('concept', '')

                if technology in excluded_formats:
                    technology = get_technology(config_file["file_path"])

                if not technology:
                    continue

                all_technologies.add(technology)

                if technology not in project_tech_options[project_name]:
                    project_tech_options[project_name][technology] = set()
                    project_tech_file_counts[project_name][technology] = 0

                # Count config files for this technology
                project_tech_file_counts[project_name][technology] += 1

                # Add unique option names (not values)
                pairs = config_file.get('pairs', [])
                for pair in pairs:
                    option = pair.get('option', '')
                    if option:  # Only add if option is not empty
                        project_tech_options[project_name][technology].add(option)

            logger.info(f"Processed {project_name}")

        except Exception as e:
            logger.error(f"Error processing {project_file}: {e}")
            continue

    # Create matrix: projects x technologies
    logger.info(f"Creating matrix for {len(project_tech_options)} projects and {len(all_technologies)} technologies")

    # Sort technologies for consistent column order
    sorted_technologies = sorted(all_technologies)

    # Build the option count matrix
    option_matrix_data = []
    for project_name, tech_options in project_tech_options.items():
        row = {'project': project_name}
        for tech in sorted_technologies:
            # Count unique option names for this technology in this project
            row[tech] = len(tech_options.get(tech, set()))
        option_matrix_data.append(row)

    option_df = pd.DataFrame(option_matrix_data)

    # Build the file count matrix
    file_matrix_data = []
    for project_name, tech_file_counts in project_tech_file_counts.items():
        row = {'project': project_name}
        for tech in sorted_technologies:
            # Count config files for this technology in this project
            row[tech] = tech_file_counts.get(tech, 0)
        file_matrix_data.append(row)

    file_df = pd.DataFrame(file_matrix_data)

    # Calculate statistics per technology across all projects
    tech_stats = {}
    for tech in sorted_technologies:
        all_options = set()
        project_option_counts = []
        project_file_counts = []

        for project_name, tech_options in project_tech_options.items():
            if tech in tech_options:
                all_options.update(tech_options[tech])
                project_option_counts.append(len(tech_options[tech]))
                project_file_counts.append(project_tech_file_counts[project_name].get(tech, 0))

        avg_options = sum(project_option_counts) / len(project_option_counts) if project_option_counts else 0
        avg_files = sum(project_file_counts) / len(project_file_counts) if project_file_counts else 0

        tech_stats[tech] = {
            'unique_options': len(all_options),
            'avg_options_per_project': avg_options,
            'avg_files_per_project': avg_files,
            'num_projects': len(project_option_counts)
        }

    logger.info("Technology statistics:")
    for tech, stats in sorted(tech_stats.items(), key=lambda x: x[1]['unique_options'], reverse=True):
        logger.info(f"  {tech}: {stats['unique_options']} unique options, "
                   f"avg {stats['avg_options_per_project']:.1f} per project, "
                   f"avg {stats['avg_files_per_project']:.1f} config files "
                   f"({stats['num_projects']} projects)")

    return option_df, file_df, tech_stats


def main():
    parser = argparse.ArgumentParser(description="Technology Statistics")
    parser.add_argument("--input", required=True, help="Name of the directory of a company")
    parser.add_argument("--limit", type=int, help="Limit the number of project files to process")
    args = parser.parse_args()

    # Load project files
    project_files = glob.glob(f"../../data/{args.input}/latest_commit/*.json")
    logger.info(f"Found {len(project_files)} project files")

    if args.limit:
        project_files = project_files[:args.limit]
        logger.info(f"Limited to {len(project_files)} project files")

    # Calculate statistics
    option_df, file_df, tech_stats = get_technology_statistics(project_files)

    # Save option matrix to CSV
    option_matrix_output = f"../../data/{args.input}/technological/technology_option_matrix.csv"
    option_df.to_csv(option_matrix_output, index=False)
    logger.info(f"Saved project-technology option matrix to {option_matrix_output}")

    # Save file count matrix to CSV
    file_matrix_output = f"../../data/{args.input}/technological/technology_file_matrix.csv"
    file_df.to_csv(file_matrix_output, index=False)
    logger.info(f"Saved project-technology file matrix to {file_matrix_output}")

    # Save technology statistics
    stats_data = []
    for tech, stats in tech_stats.items():
        stats_data.append({
            'technology': tech,
            'unique_options': stats['unique_options'],
            'avg_options_per_project': round(stats['avg_options_per_project'], 2),
            'avg_files_per_project': round(stats['avg_files_per_project'], 2),
            'num_projects': stats['num_projects']
        })

    stats_df = pd.DataFrame(stats_data)
    if not stats_df.empty:
        stats_df = stats_df.sort_values('unique_options', ascending=False)
    stats_output = f"../../data/{args.input}/technological/technology_statistics.csv"
    stats_df.to_csv(stats_output, index=False)
    logger.info(f"Saved technology statistics to {stats_output}")

    logger.info("Done!")

if __name__ == "__main__":
    main()