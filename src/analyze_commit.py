#!/usr/bin/env python3
"""
Extract configuration data from a specific commit.

Usage:
    # Analyze a single project
    python analyze_commit.py --url https://github.com/disney/groovity --name groovity --commit abc123

    # Analyze all projects from a CSV file
    python analyze_commit.py --csv ../data/disney_projects_final.csv

    Output: ../data/{prefix}/latest_commit/{project_name}_commit.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import tempfile
import traceback
from typing import Dict, List

import git
from cfgnet.network.network import Network
from cfgnet.network.network_configuration import NetworkConfiguration
from cfgnet.network.nodes import ArtifactNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def extract_config_data(network: Network) -> Dict:
    """Extract configuration data from configuration network."""
    artifacts = network.get_nodes(node_type=ArtifactNode)

    config_file_data = []
    for artifact in artifacts:
        pairs = [pair for pair in artifact.get_pairs() if pair["option"] != "file"]

        config_file_data.append({
            "file_path": artifact.rel_file_path,
            "concept": artifact.concept_name,
            "options": len(pairs),
            "pairs": pairs,
        })

    concepts = set(artifact.concept_name for artifact in artifacts)
    total_options = sum(len(artifact.get_pairs()) for artifact in artifacts)

    links = network.links
    link_data = [{
        "source_value": link.node_a.name,
        "source_type": link.node_a.config_type.name,
        "source_option": link.node_a.get_options(),
        "source_artifact": link.artifact_a.rel_file_path,
        "source_concept": link.artifact_a.concept_name,
        "target_value": link.node_b.name,
        "target_type": link.node_b.config_type.name,
        "target_option": link.node_b.get_options(),
        "target_artifact": link.artifact_b.rel_file_path,
        "target_concept": link.artifact_b.concept_name,
    } for link in links]

    return {
        "links": len(network.links),
        "link_data": link_data,
        "concepts": list(concepts),
        "config_file_data": config_file_data,
        "total_options": total_options,
    }


def analyze_commit(repo_path: str, project_name: str, commit_sha: str) -> Dict:
    """Analyze a single commit and extract config data."""
    logger.info(f"Analyzing commit {commit_sha} for project {project_name}")

    repo = git.Repo(repo_path)

    logger.info(f"Checking out commit {commit_sha}")
    repo.git.checkout(commit_sha)

    commit = repo.commit(commit_sha)

    logger.info("Building configuration network...")
    network = create_network_from_path(repo_path=repo_path)

    logger.info("Extracting configuration data...")
    config_data = extract_config_data(network=network)

    return {
        "project_name": project_name,
        "commit_sha": commit_sha,
        "author": f"{commit.author.name} <{commit.author.email}>",
        "commit_msg": str(commit.message).strip(),
        "commit_date": commit.authored_datetime.isoformat(),
        "config_data": config_data,
    }


def process_project(project_url: str, project_name: str, commit_sha: str, output_dir: str) -> None:
    """Clone repo, checkout commit, and extract config data."""
    output_file = f"{output_dir}/{project_name}_commit.json"

    if os.path.exists(output_file):
        logger.info(f"Output file {output_file} already exists. Skipping {project_name}.")
        return

    logger.info(f"Processing project: {project_name}")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            logger.info(f"Cloning {project_url} into {temp_dir}")
            subprocess.run(
                ["git", "clone", project_url, temp_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            result = analyze_commit(
                repo_path=temp_dir,
                project_name=project_name,
                commit_sha=commit_sha
            )

            logger.info(f"Writing results to {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)

            logger.info(f"Done! Found {result['config_data']['total_options']} config options across {len(result['config_data']['config_file_data'])} files")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            traceback.print_exc()
        except Exception as e:
            logger.error(f"Failed to process {project_name}: {e}")
            traceback.print_exc()


def read_projects_from_csv(csv_path: str) -> List[Dict]:
    """Read project info from CSV file."""
    projects = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clone_url = row.get("clone_url")
            name = row.get("name")
            commit_sha = row.get("latest_commit_sha")

            if not clone_url or not name or not commit_sha:
                logger.warning(f"Skipping row with missing data: {row.get('name', 'unknown')}")
                continue

            projects.append({
                "url": clone_url,
                "name": name,
                "commit": commit_sha,
            })
    return projects


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract configuration data from a specific commit"
    )
    parser.add_argument("--url", help="Git clone URL of the repository")
    parser.add_argument("--name", help="Project name")
    parser.add_argument("--commit", help="Commit SHA to analyze")
    parser.add_argument("--csv", help="CSV file with projects (columns: clone_url, name, latest_commit_sha)")
    parser.add_argument("--output-dir", default="../data", help="Output directory for results")
    args = parser.parse_args()

    if args.csv:
        logger.info(f"Reading projects from CSV: {args.csv}")
        projects = read_projects_from_csv(args.csv)
        logger.info(f"Found {len(projects)} projects to process")

        # Derive output dir from CSV filename: disney_projects.csv -> ../data/disney/latest_commit
        csv_basename = os.path.basename(args.csv)
        prefix = csv_basename.replace("_projects_final.csv", "").replace(".csv", "")
        output_dir = f"../data/{prefix}/latest_commit"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        for i, project in enumerate(projects, 1):
            logger.info(f"[{i}/{len(projects)}] Processing {project['name']}")
            process_project(
                project_url=project["url"],
                project_name=project["name"],
                commit_sha=project["commit"],
                output_dir=output_dir
            )
    else:
        if not args.url or not args.name or not args.commit:
            parser.error("Either --csv or all of --url, --name, and --commit are required")

        process_project(
            project_url=args.url,
            project_name=args.name,
            commit_sha=args.commit,
            output_dir=args.output_dir
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
