import logging
import glob
import json
import pandas as pd
import sys
import os
import argparse

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
    all_technologies = set()

    for project_file in project_files:
        try:
            with open(project_file, 'r') as f:
                data = json.load(f)

            project_name = data.get('project_name', os.path.basename(project_file).replace('_last_commit.json', ''))

            # Get concepts and config data from latest commit
            latest_commit = data.get('latest_commit_data', {})
            network_data = latest_commit.get('network_data', {})
            config_file_data = network_data.get('config_file_data', [])

            if project_name not in project_tech_options:
                project_tech_options[project_name] = {}

            # Process each config file
            for config_file in config_file_data:
                technology = config_file.get('concept', '')
                if not technology or technology in excluded_formats:
                    continue

                all_technologies.add(technology)

                if technology not in project_tech_options[project_name]:
                    project_tech_options[project_name][technology] = set()

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

    # Build the matrix
    matrix_data = []
    for project_name, tech_options in project_tech_options.items():
        row = {'project': project_name}
        for tech in sorted_technologies:
            # Count unique option names for this technology in this project
            row[tech] = len(tech_options.get(tech, set()))
        matrix_data.append(row)

    df = pd.DataFrame(matrix_data)

    # Calculate statistics per technology across all projects
    tech_stats = {}
    for tech in sorted_technologies:
        all_options = set()
        project_option_counts = []

        for project_name, tech_options in project_tech_options.items():
            if tech in tech_options:
                all_options.update(tech_options[tech])
                project_option_counts.append(len(tech_options[tech]))

        avg_options = sum(project_option_counts) / len(project_option_counts) if project_option_counts else 0

        tech_stats[tech] = {
            'unique_options': len(all_options),
            'avg_options_per_project': avg_options,
            'num_projects': len(project_option_counts)
        }

    logger.info("Technology statistics:")
    for tech, stats in sorted(tech_stats.items(), key=lambda x: x[1]['unique_options'], reverse=True):
        logger.info(f"  {tech}: {stats['unique_options']} unique options, "
                   f"avg {stats['avg_options_per_project']:.1f} per project "
                   f"({stats['num_projects']} projects)")

    return df, tech_stats


def main():
    parser = argparse.ArgumentParser(description="Technology Statistics")
    parser.add_argument("--limit", type=int, help="Limit the number of project files to process")
    args = parser.parse_args()

    # Load project files
    project_files = glob.glob("../data/projects_last_commit/*.json")
    logger.info(f"Found {len(project_files)} project files")

    if args.limit:
        project_files = project_files[:args.limit]
        logger.info(f"Limited to {len(project_files)} project files")

    # Calculate statistics
    df, tech_stats = get_technology_statistics(project_files)

    # Save matrix to CSV
    matrix_output = "../data/technology_composition/technology_option_matrix.csv"
    df.to_csv(matrix_output, index=False)
    logger.info(f"Saved project-technology matrix to {matrix_output}")

    # Save technology statistics
    stats_data = []
    for tech, stats in tech_stats.items():
        stats_data.append({
            'technology': tech,
            'unique_options': stats['unique_options'],
            'avg_options_per_project': round(stats['avg_options_per_project'], 2),
            'num_projects': stats['num_projects']
        })

    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('unique_options', ascending=False)
    stats_output ="../data/technology_composition/technology_statistics.csv"
    stats_df.to_csv(stats_output, index=False)
    logger.info(f"Saved technology statistics to {stats_output}")

    logger.info("Done!")

if __name__ == "__main__":
    main()