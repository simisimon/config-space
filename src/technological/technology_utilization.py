#!/usr/bin/env python3
"""
Technology Utilization Calculator

Computes statistics about configuration space utilization per technology.
Output is saved to ../../data/technological/utilization/ by default.

Usage:
    # Single file - process one project
    python technology_utilization.py --input ../../data/test_projects/piggymetrics_last_commit.json

    # Batch processing - process all projects in a directory
    python technology_utilization.py --input ../../data/projects_last_commit --all

    # Batch processing with limit
    python technology_utilization.py --input ../../data/projects_last_commit --all --limit 100

    # Custom output directory
    python technology_utilization.py --input ../../data/projects_last_commit --all \\
        --output-dir ../../data/results/per_project
"""

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

# Import the technology mapping function
sys.path.insert(0, str(Path(__file__).parent.parent))
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
    
    fig.savefig("../../data/technological/options_per_technology_summary.png", dpi=300)
    df.to_csv("../../data/technological/options_per_technology_summary.csv", index=False)

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

            results.append({
                "Technology": technology,
                "Total Options": len(ref_options),
                "Number of Files": num_files,
                "Options Set (Total)": len(all_options_list),
                "Options Set (Unique)": len(project_subset),
                "Matched Options": len(matched_refs),
                "Percentage Used": round(len(matched_refs) / len(ref_options) * 100, 2) if ref_options else 0.0
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
                "Percentage Used": None
            })

    return pd.DataFrame(results)


def extract_latest_options(project_file: str) -> pd.DataFrame:
    with open(project_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    config_data = []

    # Support both old and new formats
    if "config_data" in data:
        # New format: config_data directly at top level
        config_file_data = data["config_data"]["config_file_data"]
    else:
        # Old format: latest_commit_data.network_data
        latest_commit = data["latest_commit_data"]
        if not latest_commit["is_latest_commit"]:
            raise Exception("The latest commit is not the last commit in the history.")
        config_file_data = latest_commit["network_data"]["config_file_data"]

    # TODO: Remove duplicate of options
    for config_file in config_file_data:
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


def process_single_project(project_file: Path, property_files: List[str], output_dir: Path = None) -> pd.DataFrame | None:
    """
    Process a single project file and compute technology utilization.

    Args:
        project_file: Path to project JSON file
        property_files: List of property file paths for ground truth
        output_dir: Optional output directory for per-project CSV

    Returns:
        DataFrame with technology utilization results, or None on error
    """
    try:
        project_name = project_file.stem.replace("_last_commit", "").replace("_commit", "")

        # Extract options from the project file
        df_options = extract_latest_options(str(project_file))

        # Calculate technology utilization
        df_result = get_options_per_project(property_files, df_options)
        df_result["project"] = project_name

        # Save per-project CSV if output directory is provided
        if output_dir:
            output_file = output_dir / f"{project_name}_technology_utilization.csv"
            df_result.drop(columns=["project"]).to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")

        return df_result

    except Exception as e:
        logger.error(f"Error processing {project_file.name}: {e}")
        return None


def process_projects(name: str, property_files: List[str], output_dir: Path = None, limit: int = None) -> pd.DataFrame:
    """
    Process all project JSON files in a directory.

    Args:
        input_dir: Directory containing project JSON files
        property_files: List of property file paths for ground truth
        output_dir: Optional output directory for per-project CSVs
        limit: Optional limit on number of projects to process

    Returns:
        DataFrame with combined results from all projects
    """
    # Support both old (*_last_commit.json) and new (*_commit.json) patterns
    project_files = glob.glob(f"../../data/{name}/latest_commit/*.json")
    if not project_files:
        project_files = glob.glob(f"../../data/{name}/last_commit/*.json")

    if not project_files:
        logger.error(f"No project JSON files found for company {name}")
        sys.exit(1)

    if limit:
        project_files = project_files[:limit]

    logger.info(f"Processing {len(project_files)} projects...")

    all_results = []
    for project_file in tqdm(project_files, desc="Processing projects"):
        result = process_single_project(Path(project_file), property_files, output_dir)
        if result is not None:
            all_results.append(result)

    if not all_results:
        logger.error("No projects successfully processed")
        sys.exit(1)

    return pd.concat(all_results, ignore_index=True)


def aggregate_results(df: pd.DataFrame, property_files: List[str]) -> pd.DataFrame:
    """
    Aggregate results across all projects.

    Args:
        df: Combined DataFrame from all projects
        property_files: List of property file paths for ground truth

    Returns:
        Aggregated DataFrame with statistics per technology
    """
    # Build ground truth lookup
    ground_truth = {}
    for pf in property_files:
        tech = Path(pf).stem.lower()
        with open(pf, "r", encoding="utf-8") as f:
            props = javaproperties.load(f)
            ground_truth[tech] = len(props)

    # Normalize technology names for grouping
    df["__tech_norm__"] = df["Technology"].astype(str).str.strip().str.lower()

    # Get canonical display name for each technology
    label_map = (
        df.groupby("__tech_norm__")["Technology"]
          .agg(lambda s: Counter(s).most_common(1)[0][0])
          .to_dict()
    )

    # Aggregate statistics
    agg = df.groupby("__tech_norm__", dropna=False).agg({
        "project": "nunique",
        "Number of Files": "mean",
        "Options Set (Total)": "mean",
        "Options Set (Unique)": "mean"
    }).reset_index()

    agg.columns = [
        "__tech_norm__",
        "num_projects",
        "avg_files_per_project",
        "avg_options_set_total",
        "avg_options_set_unique"
    ]

    # Add technology display name and ground truth
    agg["technology"] = agg["__tech_norm__"].map(label_map)
    agg["ground_truth"] = agg["__tech_norm__"].map(ground_truth)

    # Round numeric columns
    for col in ["avg_files_per_project", "avg_options_set_total", "avg_options_set_unique"]:
        agg[col] = agg[col].round(2)

    # Select and order columns
    result = agg[[
        "technology",
        "ground_truth",
        "num_projects",
        "avg_files_per_project",
        "avg_options_set_total",
        "avg_options_set_unique"
    ]]

    return result.sort_values("technology").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute technology utilization statistics for configuration options",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Name of the directory of a company"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of projects to process (only applies with --all)"
    )
    parser.add_argument(
        "--technologies",
        type=str,
        default="../../data/technologies",
        help="Directory containing technology property files (default: ../../data/technologies)"
    )

    args = parser.parse_args()

    tech_dir = Path(args.technologies)
    property_files = glob.glob(str(tech_dir / "*.properties"))

    if not property_files:
        logger.error(f"No property files found in {tech_dir}")
        sys.exit(1)

    logger.info(f"Loaded {len(property_files)} technology property files")

    output_dir = Path(f"../../data/{args.input}/technological/utilization")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process all projects
    df_all = process_projects(args.input, property_files, output_dir, args.limit)

    # Aggregate results
    df_aggregated = aggregate_results(df_all, property_files)

    # Save aggregated results
    output_csv = Path(f"../../data/{args.input}/technological/aggregated_technology_utilization.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_aggregated.to_csv(output_csv, index=False)
    logger.info(f"Saved aggregated results to: {output_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATED TECHNOLOGY UTILIZATION")
    print("=" * 60)
    print(f"Projects processed: {df_all['project'].nunique()}")
    print(f"Technologies found: {len(df_aggregated)}")
    print("\nTop 10 most used technologies:")
    top10 = df_aggregated.nlargest(10, "num_projects")
    for _, row in top10.iterrows():
        gt = f"(GT: {int(row['ground_truth'])})" if pd.notna(row['ground_truth']) else "(no GT)"
        print(f"  {row['technology']}: {row['num_projects']} projects, "
              f"avg {row['avg_options_set_unique']:.1f} unique options {gt}")
    print("=" * 60)


if __name__ == "__main__":
    main()

