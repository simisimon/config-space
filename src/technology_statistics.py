import logging
import argparse
import glob
import json
import numpy as np
import pandas as pd
import os
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_project_files(limit: int | None = None):
    project_files = glob.glob("../data/projects/*.json")
    logger.info(f"Found {len(project_files)} project files")
    if limit:
        project_files = project_files[:limit]
        logger.info(f"Using {len(project_files)} project files")
    return project_files


def extract_latest_commit_data(json_data):
    latest_commit = next((c for c in json_data["commit_data"] if c.get("is_latest_commit")), None)
    if not latest_commit and json_data["commit_data"]:
        latest_commit = json_data["commit_data"][-1]
    return latest_commit


def analyze_project(project_file: str):
    logger.info(f"Analyzing {project_file}")
    with open(project_file, 'r') as f:
        data = json.load(f)
    
    latest_commit = extract_latest_commit_data(data)
    if not latest_commit:
        return []

    #concepts = latest_commit["network_data"].get("concepts", [])
    config_files = latest_commit["network_data"].get("config_file_data", [])
    # TODO
    concepts = [file["concept"] for file in config_files]

    file_options = [f.get("options", 0) for f in config_files]
    avg_options = round(sum(file_options) / len(file_options), 2) if file_options else 0

    result = []
    for concept in set(concepts):
        files_for_concept = [f for f in config_files if f["concept"] == concept]
        file_count = len(files_for_concept)
        total_options = sum(f.get("options", 0) for f in files_for_concept)
        avg_options = round(total_options / file_count, 2) if file_count else 0

        result.append({
            "Project": data.get("project_name", os.path.basename(project_file)),
            "Technology": concept,
            "File Count": file_count,
            "Average Options per File": avg_options
        })
    return result


def compute_technology_statistics(project_files: List[str]):
    aggregated_data = []
    for file_name in project_files:
        aggregated_data.extend(analyze_project(file_name))
    return pd.DataFrame(aggregated_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Limit number of projects to process")
    args = parser.parse_args()
    project_files = load_project_files(args.limit)
    df_technologies = compute_technology_statistics(project_files)

    # Calculate average files per technology per project
    tech_proj_file_counts = df_technologies.groupby(["Technology", "Project"])["File Count"].sum().reset_index()
    avg_files_per_tech_per_proj = tech_proj_file_counts.groupby("Technology")["File Count"].mean().round(2)
    avg_files_per_tech_per_proj.name = "Avg_Files_Per_Project"

    df_grouped = df_technologies.groupby("Technology").agg(
        Usage_Count=("Technology", "count"),
        Total_Projects=("Project", "nunique"),
        Avg_Options_Per_File=("Average Options per File", "mean"),
        Total_Files=("File Count", "sum")
    ).reset_index()

    df_grouped["Avg_Options_Per_File"] = df_grouped["Avg_Options_Per_File"].round(2)
    df_grouped = df_grouped.merge(avg_files_per_tech_per_proj, on="Technology")
    df_grouped.to_csv("../data/results/technological_composition/technology_statistics.csv", index=False)