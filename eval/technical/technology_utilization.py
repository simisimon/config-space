import pandas as pd
import json
import javaproperties
import os
import glob
import re
import argparse
import logging
import sys
from tqdm import tqdm
from typing import List, Set, Dict
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from mapping import get_technology


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_properties_file(file_path: str) -> set:
    with open(file_path, "r", encoding="utf-8") as f:
        props = javaproperties.load(f)
    return set(k.strip() for k in props.keys())


def get_matches(project_options: Set[str], ref_options: Set[str]) -> Dict[str, List[str]]:
    matched_ref_to_project = {}

    for ref_opt in ref_options:
        pattern = '^' + re.escape(ref_opt).replace(r'\*', r'.+') + '$'
        regex = re.compile(pattern)
        
        matches = [opt for opt in project_options if regex.match(opt)]

        if matches:
            matched_ref_to_project[ref_opt] = matches

    return matched_ref_to_project


def get_options_per_technology(technology_files: List):
    data = []

    common_file_types = ["json", "yaml", "configparser"]

    for file_path in technology_files:
        technology = os.path.basename(file_path).replace(".properties", "")
        if technology in common_file_types:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            props = javaproperties.load(f)
            total = len(props)
            with_defaults = sum(1 for v in props.values() if v is not None and str(v).strip() != "")
        
        data.append((technology, total, with_defaults))

    df = pd.DataFrame(data, columns=["Technology", "Total Options", "With Defaults"])
    df.sort_values(by="Total Options", ascending=False, inplace=True)

    # Plotting grouped bars
    x = range(len(df))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - width/2 for i in x], df["Total Options"], width=width, label="Total Options", color="skyblue")
    ax.bar([i + width/2 for i in x], df["With Defaults"], width=width, label="With Default Values", color="orange")

    ax.set_xlabel('Technology', fontsize=9)
    ax.set_ylabel('Number of Options', fontsize=9)
    ax.set_title('Options per Technology (Total vs. With Defaults)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Technology"], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)

    fig.tight_layout(pad=1.0)
    
    fig.savefig("../data/technology_utilization/options_per_technology.png", dpi=300)
    df.to_csv("../data/technology_utilization/options_per_technology.csv", index=False)

    return df.drop(columns=["With Defaults"])


def get_options_per_project(technology_files: List, df_options: pd.DataFrame) -> pd.DataFrame:
    project_options = df_options.copy()
    project_options["option"] = project_options["option"].str.strip()

    # Create a mapping from technology name to property file
    tech_to_file = {}
    for technology_file in technology_files:
        technology = technology_file.split("/")[-1].split(".properties")[0]
        tech_to_file[technology.lower()] = technology_file

    # Get all unique technologies in the project
    all_project_technologies = project_options["concept"].dropna().unique()

    # Prepare result table
    results = []

    # Process all technologies found in the project
    for technology in all_project_technologies:
        technology_lower = technology.lower()

        # Get all options for this technology (including duplicates)
        tech_options_df = project_options[project_options["concept"].str.lower() == technology_lower]
        all_options_list = tech_options_df["option"].tolist()

        # Get unique options
        project_subset = set(all_options_list)

        if not project_subset:
            continue

        # Count unique files for this technology
        num_files = tech_options_df["file_path"].nunique()

        # Check if this technology has a property file
        if technology_lower in tech_to_file:
            # Technology has a property file - compute full metrics
            ref_options = parse_properties_file(tech_to_file[technology_lower])
            ref_to_proj = get_matches(project_subset, ref_options)
            matched_refs = set(ref_to_proj.keys())
            matched_project_options = sorted({opt for opts in ref_to_proj.values() for opt in opts})
            unmatched = [opt for opt in project_subset if opt not in matched_project_options]

            results.append({
                "Technology": technology,
                "Total Options": len(ref_options),
                "Number of Files": num_files,
                "Options Set (Total)": len(all_options_list),
                "Options Set (Unique)": len(project_subset),
                "Matched Options": len(matched_refs),
                "Unmatched Options": len(unmatched),
                "Percentage Used": round(len(matched_refs) / len(ref_options) * 100, 2) if ref_options else 0.0,
                "Matched": list(matched_project_options)
            })
        else:
            # Technology doesn't have a property file - only compute basic metrics
            results.append({
                "Technology": technology,
                "Total Options": None,
                "Number of Files": num_files,
                "Options Set (Total)": len(all_options_list),
                "Options Set (Unique)": len(project_subset),
                "Matched Options": None,
                "Unmatched Options": None,
                "Percentage Used": None,
                "Matched": []
            })

    return pd.DataFrame(results)


def extract_latest_options(project_file: str) -> pd.DataFrame:
    with open(project_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    config_data = []
    latest_commit = data["latest_commit_data"]

    if not latest_commit["is_latest_commit"]:
        raise Exception("The latest commit is not the last commit in the history.")

    # TODO: Remove duplicate of options
    for config_file in latest_commit["network_data"]["config_file_data"]:
        for pair in config_file["pairs"]:
            concept = config_file["concept"]
            if concept in {"json", "yaml", "configparser", "xml", "toml"}:
                concept = get_technology(config_file["file_path"])

            config_data.append({
                "concept": concept,
                "file_path": config_file["file_path"],
                "option": pair["option"],
                "value": pair["value"],
                "type": pair["type"],
                
            })

    df_options = pd.DataFrame(config_data)
    return df_options


def load_project_files(limit: int | None = None):
    project_files = glob.glob("../data/projects_last_commit/*.json")
    logger.info(f"Found {len(project_files)} project files")
    if limit:
        project_files = project_files[:limit]
        logger.info(f"Using {len(project_files)} project files")
    return project_files


def load_project(project_file: str) -> dict:
    with open(project_file, "r", encoding="utf-8") as f:
        logger.info(f"Load data for project file: {project_file}")
        data = json.load(f)
    return data

def aggregate_option_per_technology(option_files: List[str]) -> pd.DataFrame:
    dfs = []
    for f in option_files:
        try:
            df = pd.read_csv(f)
            df["__project__"] = Path(f).stem.replace("_technology_utilization", "")
            # normalize the technology key for grouping (handles whitespace/case)
            df["__tech_norm__"] = df["Technology"].astype(str).str.strip().str.lower()
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error processing {f}: {e}")

    full = pd.concat(dfs, ignore_index=True)

    # choose a canonical display label for each normalized technology (most common original form)
    label_map = (
        full.groupby("__tech_norm__")["Technology"]
            .agg(lambda s: Counter(s).most_common(1)[0][0])
            .to_dict()
    )

    # Aggregate statistics for each technology
    agg_stats = (
        full.groupby("__tech_norm__", dropna=False)
            .agg({
                "__project__": "nunique",  # number of projects
                "Number of Files": "mean",  # avg number of config files
                "Options Set (Total)": ["mean", "min", "max"]  # avg, min, max options set
            })
            .reset_index()
    )

    # Flatten column names
    agg_stats.columns = [
        "__tech_norm__",
        "projects_used",
        "avg_num_files",
        "avg_options_set",
        "min_options_set",
        "max_options_set"
    ]

    # Add technology display name
    agg_stats["technology"] = agg_stats["__tech_norm__"].map(label_map)

    # Round numeric columns
    for col in ["avg_num_files", "avg_options_set", "min_options_set", "max_options_set"]:
        agg_stats[col] = agg_stats[col].round(2)

    # Select and order columns
    out = agg_stats[["technology", "projects_used", "avg_num_files", "avg_options_set", "min_options_set", "max_options_set"]]
    out = out.sort_values("technology").reset_index(drop=True)

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of projects to process")
    args = parser.parse_args()
    project_files = load_project_files(args.limit)
    property_files = glob.glob("../data/technologies/*.properties")

    # Get options per technology
    df_technology_options = get_options_per_technology(property_files)

    for project_file in tqdm(project_files, desc="Processing Project Files"):
        project_name = project_file.split("/")[-1].replace("_last_commit.json", "")
        output_file = f"../data/projects_technology_utilization/{project_name}_technology_utilization.csv"
        
        try:
            # Extract options from the project file
            df_latest_project_options = extract_latest_options(project_file)

            # Calculate technology utilization
            df_project_options = get_options_per_project(property_files, df_latest_project_options)
            df_project_options.to_csv(output_file, index=False)
        except Exception as e:
            logger.error(f"Error processing project {project_name}: {e}")
            continue

    # Combine all project options into a single DataFrame
    all_project_option_files = glob.glob("../data/projects_technology_utilization/*.csv")

    df_aggregated = aggregate_option_per_technology(all_project_option_files)

    df_aggregated.to_csv("../data/results/options_per_technology_aggregated.csv", index=False)

    # test_projects = [
    #     "../data/test_projects/piggymetrics_last_commit.json",
    #     "../data/test_projects/test-config-repo_last_commit.json",
    #     "../data/test_projects/Avalonia_last_commit.json"
    # ]

    # property_files = glob.glob("../data/technologies/*.properties")

    # # Get options per technology (only needs to be done once)
    # df_technology_options = get_options_per_technology(property_files)

    # for project_file in test_projects:
    #     project_name = project_file.split("/")[-1].replace("_last_commit.json", "")
    #     output_file = f"../data/test_projects/{project_name}_technology_utilization.csv"

    #     #if os.path.exists(output_file):
    #     #    logger.info(f"Options file already exists for {project_name}, skipping.")
    #     #    continue

    #     try:
    #         logger.info(f"Processing project: {project_name}")

    #         # Extract options from the project file
    #         df_latest_project_options = extract_latest_options(project_file)

    #         # Calculate technology utilization
    #         df_project_options = get_options_per_project(property_files, df_latest_project_options)

    #         # Save results
    #         df_project_options.to_csv(output_file, index=False)
    #         logger.info(f"Saved technology utilization to {output_file}")

    #     except Exception as e:
    #         logger.error(f"Error processing project {project_name}: {e}")
    #         continue

