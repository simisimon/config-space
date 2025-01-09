from typing import Dict
from cfgnet.network.network_configuration import NetworkConfiguration
from cfgnet.network.nodes import ArtifactNode
from cfgnet.network.network import Network
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint
from tqdm import tqdm
from typing import List
import git
import json
import subprocess
import traceback
import time
import logging
import argparse
import tempfile
import subprocess
import os


CONFIG_FILE_ENDINGS = (".xml", ".yml", ".yaml", "Dockerfile", ".ini", ".properties", ".conf", ".json", ".toml", ".cfg", "settings.py", ".cnf")


MICROSERVICES = [
    {"html_url": "https://github.com/sqshq/piggymetrics", "name": "piggymetrics"},
    {"html_url": "https://github.com/Yin-Hongwei/music-website", "name": "music-website"},
    {"html_url": "https://github.com/pig-mesh/pig", "name": "pig"},
    {"html_url": "https://github.com/macrozheng/mall", "name": "mall"},
    {"html_url": "https://github.com/macrozheng/mall-swarm", "name": "mall-swarm"},
    {"html_url": "https://github.com/linlinjava/litemall", "name": "litemall"},
    {"html_url": "https://github.com/wxiaoqi/Spring-Cloud-Platform", "name": "Spring-Cloud-Platform"},
    {"html_url": "https://github.com/apolloconfig/apollo", "name": "apollo"},
]


class ExcludeWarningsFilter(logging.Filter):
    """Custom filter to exclude WARNING logs."""
    def filter(self, record):
        return record.levelno != logging.WARNING


def configure_logging():
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set base level to INFO

    # Create file handler
    file_handler = logging.FileHandler("../data/analysis.log")
    file_handler.setLevel(logging.INFO)

    # Apply the custom filter to exclude warnings
    file_handler.addFilter(ExcludeWarningsFilter())

    # Set the format for log messages
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)


def checkout_latest_commit(repo, current_branch, latest_commit):
     # Return to the latest commit
    if current_branch:
        # If we were on a branch, return to it
        repo.git.checkout(current_branch)
    else:
        # If we were in a detached HEAD state, checkout the latest commit directly
        repo.git.checkout(latest_commit)


def extract_config_data(repo_path: str):
    """Extract configuration data."""
    network_config = NetworkConfiguration(
        project_root_abs=repo_path,
        enable_static_blacklist=False,
        enable_internal_links=True,
        enable_all_conflicts=True,
        enable_file_type_plugins=True,
        system_level=False
    )

    network = Network.init_network(cfg=network_config)

    artifacts = network.get_nodes(node_type=ArtifactNode)

    config_files_data = []
    for artifact in artifacts:
        pairs = artifact.get_pairs()

        # exclude file options
        pairs = [pair for pair in pairs if pair["option"] != "file"] 

        config_files_data.append({
            "file_path": artifact.rel_file_path,
            "concept": artifact.concept_name,
            "options": len(artifact.get_pairs()),
            "pairs": pairs
        })


    config_files = set(artifact.rel_file_path for artifact in artifacts)
  	
    network_data = {
        "links": len(network.links),
        "config_files": list(config_files),
        "config_files_data": config_files_data
    }

    return network_data


def get_file_diff(repo_path: str, commit, file_path: str):
    """Get file diff for a config file in a given commit."""
    try:
        if commit.parents:
            parent_commit = f"{commit.hexsha}^"
                
            # Run git diff to capture line-by-line changes
            diff_output = subprocess.check_output(
                ['git', 'diff', parent_commit, commit.hexsha, '--', file_path],
                cwd=repo_path,
                text=True
            )
            return diff_output
    except Exception:
        logging.warning(f"Failed to get diff for commit {commit.hexsha} and file {file_path}")
        return None


def analyze_repository(repo_path: str, project_name: str, get_diff: bool = False) -> Dict:
    """Analyze Commit history of repositories and collect stats about the configuration space."""  
    start_time = time.time()
    repo = git.Repo(repo_path)

    # Save the current branch to return to it later
    current_branch = repo.active_branch.name if not repo.head.is_detached else None
    latest_commit = repo.head.commit.hexsha
    parent_commit = None

    # Get all commits in the repository from oldest to newest
    commits = list(repo.iter_commits("HEAD"))[::-1]

    print(f"Number of commits: {len(commits)}")

    config_commit_data = []

    for commit in tqdm(commits, desc="Processing", total=len(commits)):

        is_config_related = False

        # Get commit stats
        stats = commit.stats.total

        # Checkout the commit
        repo.git.checkout(commit.hexsha)

        # check if commit is config-related
        if any(file_path.endswith(CONFIG_FILE_ENDINGS) for file_path in commit.stats.files.keys()):
            is_config_related = True
            
            # Run the external analysis for config-related commits
            network_data = extract_config_data(repo_path=repo_path)

            # Get general stats per config file
            for file_path, file_stats in commit.stats.files.items():
                
                # Get config file data
                if file_path in network_data["config_files"]:
                    file_data = next(filter(lambda x: x["file_path"] == file_path, network_data["config_files_data"]))
                    file_data["insertions"] = file_stats['insertions']
                    file_data["deletions"] = file_stats['deletions']
                    file_data["total_changes"] = file_stats['insertions'] + file_stats['deletions']

                    # Get config file diff
                    if get_diff:
                        diff_output = get_file_diff(
                            repo_path=repo_path,
                            commit=commit,
                            file_path=file_path
                        )

                        file_data["diff"] = diff_output

            config_commit_data.append(
                {   
                    "commit_hash": str(commit.hexsha),
                    "parent_commit": (parent_commit),
                    "is_config_related": is_config_related,
                    "author": f"{commit.author.name} <{commit.author.email}>",
                    "commit_mgs": str(commit.message),
                    "files_changed": stats['files'],
                    "insertions": stats['insertions'],
                    "deletions": stats['deletions'],
                    "network_data": network_data
                }
            )
        
        else:
            config_commit_data.append(
                {   
                    "commit_hash": str(commit.hexsha),
                    "parent_commit": (parent_commit),
                    "is_config_related": is_config_related,
                    "author": f"{commit.author.name} <{commit.author.email}>",
                    "commit_mgs": str(commit.message),
                    "files_changed": stats['files'],
                    "insertions": stats['insertions'],
                    "deletions": stats['deletions'],
                    "network_data": None
                }
            )


    # Return to latest commit
    checkout_latest_commit(
        repo=repo, 
        current_branch=current_branch,
        latest_commit=latest_commit
    )

    print(f"Len commit data: {len(config_commit_data)}, {round(len(config_commit_data)/len(commits), 2)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    
    return {
        "project_name": project_name,
        "analysis_time": elapsed_time,
        "len_commits": len(commits),
        "config_commit_data": config_commit_data
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="Path to the data file containing project details")
    parser.add_argument("--parallel", type=int, default=10, help="Number of parallel processes to use")
    return parser.parse_args()


def process_project(project):
    """Process a single project."""
    project_url = project["html_url"]
    project_name = project["name"]

    # Define the output file path
    output_file = f"../data/analyzed_projects/{project_name}.json"

    # Check if the output file already exists
    if os.path.exists(output_file):
        logging.info(f"Output file already exists for {project_name}. Skipping processing.")
        return

    logging.info(f"Processing project: {project_name}")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            logging.info(f"Cloning {project_name} into {temp_dir}")
            subprocess.run(
                ["git", "clone", project_url, temp_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Analyze repository to get commit config data
            logging.info(f"Analyzing repository: {project_name}")
            commit_data = analyze_repository(repo_path=temp_dir, project_name=project_name, get_diff=True)

            # Store commit data into the output file
            with open(output_file, "w", encoding="utf-8") as dest:
                json.dump(commit_data, dest, indent=4)

            logging.info(f"Analysis for {project_name} stored at {output_file}")

        except Exception as error:
            logging.error(f"Failed to process **{project_name}**: {error}")
            traceback.print_exc()


def run_analysis(args):
    """Run the repository analysis."""
    # Load and validate data
    #with open(args.data_file, "r", encoding="utf-8") as src:
    #    data = json.load(src)

    data = MICROSERVICES

    logging.info(f"Loaded {len(data)} projects for analysis.")

    # TODO: Works not as intendet, stops analysis at a certain commit for mulitple repositories
    #with ProcessPoolExecutor(max_workers=args.parallel) as executor:
    #    list(tqdm(executor.map(process_project, data), total=len(data), desc="Analyzing Projects"))

    for project in data:
        process_project(project=project)

    logging.info("Completed analysis for all projects.")


if __name__ == "__main__":
    args = get_args()

    # Configure logging
    configure_logging()

    # Start analysis
    logging.info("Starting analysis")
    run_analysis(args=args)