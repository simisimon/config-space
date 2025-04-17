from typing import Dict
from cfgnet.network.network_configuration import NetworkConfiguration
from cfgnet.network.nodes import ArtifactNode, OptionNode
from cfgnet.conflicts.conflict_detector import ConflictDetector
from cfgnet.network.network import Network
from pprint import pprint
from tqdm import tqdm
from typing import List
import git
import json
import subprocess
import traceback
import time
import argparse
import tempfile
import subprocess
import os


CONFIG_FILE_ENDINGS = (".xml", ".yml", ".yaml", "Dockerfile", ".ini", ".properties", ".conf", ".json", ".toml", ".cfg", "settings.py", ".cnf")



def checkout_latest_commit(repo, current_branch, latest_commit):
     # Return to the latest commit
    if current_branch:
        # If we were on a branch, return to it
        repo.git.checkout(current_branch)
    else:
        # If we were in a detached HEAD state, checkout the latest commit directly
        repo.git.checkout(latest_commit)


def create_network_from_path(repo_path: str) -> Network:
    """Create network from repo path."""
    network_config = NetworkConfiguration(
        project_root_abs=repo_path,
        enable_static_blacklist=False,
        enable_internal_links=True,
        enable_all_conflicts=True,
        enable_file_type_plugins=True,
        system_level=False
    )
    network = Network.init_network(cfg=network_config)
    return network


def is_equal_pair(pair1, pair2):
    """Compare two pairs without considering the 'line' property."""
    return (
        pair1["option"] == pair2["option"] and
        pair1["value"] == pair2["value"] and
        pair1["type"] == pair2["type"]
    )


def get_file_diff(repo_path: str, commit, file_path: str):
    """Get file diff for a config file in a given commit."""
    try:
        # Run git show to capture the changes introduced by the commit for the specified file
        diff_output = subprocess.check_output(
            ['git', 'show', commit.hexsha, '--', file_path],
            cwd=repo_path,
            text=True
        ).split("diff --git")[-1]

        return "diff --git" + diff_output

    except subprocess.CalledProcessError as e:
        print(f"Git command failed for commit {commit.hexsha} and file {file_path}: {e}")
        return None
    except Exception:
        print(f"Unexpected error while getting diff for commit {commit.hexsha} and file {file_path}")
        traceback.print_exc()
        return None


def is_commit_config_related(commit) -> bool:
    """Check if a commit is config-related."""
    return any(file_path.endswith(CONFIG_FILE_ENDINGS) for file_path in commit.stats.files.keys())


def is_config_file(file_path: str) -> bool:
    """Check if file is a config file."""
    if file_path.endswith(CONFIG_FILE_ENDINGS):
        return True
    return False

def extract_conflicts(new_network: Network, ref_network: Network, commit_hash: str) -> List:
    """Extract conflicts from configuration network."""
    conflicts = []
    if ref_network:
        detected_conflicts = ConflictDetector.detect(
            ref_network=ref_network,
            new_network=new_network,
            enable_all_conflicts=False,
            commit_hash=commit_hash)
        
        for conflict in detected_conflicts:
            conflicts.append({
                "link": str(conflict.link),
                "conflict_type": type(conflict).__name__,
            })

    return conflicts


def extract_config_data(new_network: Network, ref_network: Network) -> Dict:
    """Extract configuration data from configuration network."""
    artifacts = new_network.get_nodes(node_type=ArtifactNode)

    config_file_data = []
    for artifact in artifacts:
        # exclude file options
        pairs = [pair for pair in artifact.get_pairs() if pair["option"] != "file"]

        ref_pairs = []
        if ref_network:
            ref_artifact = ref_network.find_artifact_node(artifact)
            if ref_artifact:
                ref_pairs = [pair for pair in ref_artifact.get_pairs() if pair["option"] != "file"]
        
        added_pairs = [pair for pair in pairs if not any(is_equal_pair(pair, ref_pair) for ref_pair in ref_pairs)]
        removed_pairs = [pair for pair in ref_pairs if pair not in pairs]
        modified_pairs = [
            {   
                "artifact": artifact.rel_file_path,
                "option": added_pair["option"],
                "prev_value": next((p["value"] for p in removed_pairs if p["option"] == added_pair["option"]), ""),
                "curr_value": added_pair["value"],
                "line": added_pair["line"],
                "type": added_pair["type"]
            }
            for added_pair in added_pairs
            if any(removed_pair["option"] == added_pair["option"] and removed_pair["value"] != added_pair["value"] for removed_pair in removed_pairs)
        ]

        # Remove modified pairs from added and removed lists
        added_pairs = [pair for pair in added_pairs if pair["option"] not in [mp["option"] for mp in modified_pairs]]
        removed_pairs = [pair for pair in removed_pairs if pair["option"] not in [mp["option"] for mp in modified_pairs]]

        config_file_data.append({
            "file_path": artifact.rel_file_path,
            "concept": artifact.concept_name,
            "options": len(artifact.get_pairs()),
            "pairs": pairs,
            "added_pairs": added_pairs,
            "removed_pairs": removed_pairs,
            "modified_pairs": modified_pairs,
        })

    concepts = set(artifact.concept_name for artifact in artifacts)
    total_options = sum(len(artifact.get_pairs()) for artifact in artifacts)

    network_data = {
        "links": len(new_network.links),
        "concepts": list(concepts),
        "config_file_data": config_file_data,
        "total_options": total_options,
    }

    return network_data


def analyze_repository(repo_path: str, project_name: str) -> Dict:
    """Analyze Commit history of repositories and collect stats about the configuration space."""  
    start_time = time.time()
    repo = git.Repo(repo_path)

    # Save the current branch to return to it later
    current_branch = repo.active_branch.name if not repo.head.is_detached else None
    latest_commit = repo.head.commit.hexsha

    # Get all commits in the repository from oldest to newest
    commits = list(repo.iter_commits("HEAD"))[::-1]

    print(f"Number of commits: {len(commits)}")

    commit_data = []
    ref_network = None

   
    for commit in tqdm(commits, desc="Processing", total=len(commits)):
        try:
            is_config_related = False

            # Stash changes before checkout
            if repo.is_dirty(untracked_files=True):
                repo.git.stash('push')

            # Check if this is the latest commit
            is_latest_commit = (commit.hexsha == latest_commit)

            repo.git.checkout(commit.hexsha)

            if is_commit_config_related(commit) or is_latest_commit:
                is_config_related = True

                new_network = create_network_from_path(repo_path=repo_path)
                network_data = extract_config_data(new_network=new_network, ref_network=ref_network)
                conflicts = extract_conflicts(new_network=new_network, ref_network=ref_network, commit_hash=str(commit.hexsha))
                modified_files = commit.stats.files.keys() 

                config_files = network_data["config_file_data"]
                for config_file in config_files:
                    if config_file["file_path"] in modified_files:
                        stats = commit.stats.files[config_file["file_path"]]
                        config_file["is_modified"] = True
                        config_file["insertions"] = stats["insertions"]
                        config_file["deletions"] = stats["deletions"]
                        config_file["lines"] = stats["lines"]

                        # Get diff for the file
                        diff_output = get_file_diff(
                            repo_path=repo_path,
                            commit=commit,
                            file_path=config_file["file_path"]
                        )
                        config_file["file_diff"] = diff_output
                    else:
                        config_file["is_modified"] = False
                        config_file["insertions"] = None
                        config_file["deletions"] = None
                        config_file["lines"] = None
                
                commit_data.append(
                    {   
                        "commit_hash": str(commit.hexsha),
                        "is_latest_commit": is_latest_commit,
                        "is_config_related": is_config_related,
                        "author": f"{commit.author.name} <{commit.author.email}>",
                        "commit_mgs": str(commit.message),
                        "network_data": network_data,
                        "conflicts": conflicts
                    }
                )

                ref_network = new_network

            else:
                commit_data.append(
                    {   
                        "commit_hash": str(commit.hexsha),
                        "is_latest_commit": is_latest_commit,
                        "is_config_related": is_config_related,
                        "author": f"{commit.author.name} <{commit.author.email}>",
                        "commit_mgs": str(commit.message),
                        "network_data": {},
                        "conflicts": []
                    }
                )
        except Exception as error:
            print(f"Failed to process commit {commit.hexsha}: {error}")
            traceback.print_exc()
            commit_data.append(
                {   
                    "commit_hash": str(commit.hexsha),
                    "is_latest_commit": is_latest_commit,
                    "is_config_related": is_config_related,
                    "author": f"{commit.author.name} <{commit.author.email}>",
                    "commit_mgs": str(commit.message),
                    "network_data": {},
                    "conflicts": []
                }
            )


    # Return to latest commit
    checkout_latest_commit(
        repo=repo, 
        current_branch=current_branch,
        latest_commit=latest_commit
    )

    print(f"Len commit data: {len(commit_data)}, {round(len(commit_data)/len(commits), 2)}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    
    return {
        "project_name": project_name,
        "analysis_time": elapsed_time,
        "len_commits": len(commits),
        "commit_data": commit_data
    }


def process_project(project_url: str, project_name: str):
    """Process a single project."""

    # Define the output file path
    #output_file = f"/tmp/ssimon/config-space/experiments/{project_name}.json"
    output_file = f"../data/microservice_projects/{project_name}.json"

    print(f"Processing project: {project_name}")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"Cloning {project_name} into {temp_dir}")
            subprocess.run(
                ["git", "clone", project_url, temp_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Analyze repository to get commit config data
            print(f"Analyzing repository: {project_name}")
            commit_data = analyze_repository(repo_path=temp_dir, project_name=project_name)

            # Store commit data into the output file
            with open(output_file, "w", encoding="utf-8") as dest:
                json.dump(commit_data, dest, indent=4)

            print(f"Analysis for {project_name} stored at {output_file}")

        except Exception as error:
            print(f"Failed to process **{project_name}**: {error}")
            traceback.print_exc()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="https://github.com/simisimon/test-config-repo", help="Url of the repository to analyze")
    parser.add_argument("--name", type=str, default="test-config-repo", help="Name of the repository to analyze")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Start analysis
    print(f"Starting analysis for project: {args.name}")
    
    process_project(
        project_url=args.url,
        project_name=args.name
    )

    print(f"Completed analysis for project: {args.name}.")