import logging
import argparse
import glob
import json
import pandas as pd
import plotly.express as px
import numpy as np
from typing import List
import sys
import os
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Tuple
from upsetplot import UpSet, from_memberships


import kaleido
kaleido.get_chrome_sync()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True, stream=sys.stdout)
logger = logging.getLogger(__name__)


FILE_TYPES = {
    "yaml": ["ansible", "ansible playbook", "kubernetes", "docker compose", "github action", "circleci", "elasticsearch", "flutter", "heroku", "spring", "travis", "yaml"],
    "properties": ["alluxio", "spring", "kafka", "gradle", "gradle wrapper", "maven wrapper", "properties"],
    "json": ["angular", "tsconfig", "nodejs", "cypress", "json"],
    "xml": ["maven", "android", "hadoop common", "hadoop hbase", "hadoop hdfs", "mapreduce", "yarn", "xml"],
    "toml": ["cargo", "netlify", "poetry", "toml"],
    "conf": ["mongodb", "nginx", "postgresql", "rabbitmq", "redis", "apache", "conf"],
    "ini": ["mysql", "php", "ini"],
    "cfg": ["zookeeper"],
    "other": ["docker", "django"]
}

COLORS = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
    "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
    "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"
]


def load_project_files(limit: int | None = None):
    project_files = glob.glob("../data/projects_last_commit/*.json")
    logger.info(f"Found {len(project_files)} project files")
    if limit:
        project_files = project_files[:limit]
        logger.info(f"Using {len(project_files)} project files")
    return project_files


def extract_technologies(project_files: List[str], output_file: str) -> pd.DataFrame:
    """
    Extracts technologies from project configuration files and saves to CSV.
    """
    project_technologies = []
    for project_file in project_files:
        project_name = project_file.split("/")[-1].replace(".json", "")
        technologies = []
        try:
            with open(project_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                latest_commit = data["latest_commit_data"]
                config_data = latest_commit["network_data"]["config_file_data"]

                for config_file in config_data:
                    # TODO: Check if file is actually a config file or just contains data
                    technologies.append(config_file["concept"])

            project_technologies.append(
                {
                    "project": project_name.split("_last_commit")[0],
                    "technologies": sorted(set(technologies))
                }
            )
        except Exception as e:
            logging.error(f"Error processing {project_file}: {e}")
            continue

    df = pd.DataFrame(project_technologies)
    df.to_csv(output_file, index=False)

    return df


def norm(s: str) -> str:
    # Lowercase, replace -, _, multiple spaces → single space; trim
    return " ".join(str(s).lower().replace("-", " ").replace("_", " ").split())


def build_concept_to_filetype(FILE_TYPES) -> dict[str, str]:
    m = {}
    for ext, concepts in FILE_TYPES.items():
        for c in concepts:
            m[norm(c)] = ext
    return m


def get_technology_landscape(df: pd.DataFrame):
    """
    Creates a treemap visualization of the technology landscape.
    """
    concept_to_filetype = build_concept_to_filetype(FILE_TYPES)

    tech_counts = {}
    for _, row in df.iterrows():
        project_technologies = row["technologies"]
        for raw_concept in project_technologies:
            normalized_concept = " ".join(raw_concept.lower().split("-")).strip()
            file_type = concept_to_filetype.get(normalized_concept, "other")

            # Rename technology to 'Other' only if concept exactly equals file_type
            concept_label = "other" if normalized_concept == file_type else normalized_concept

            key = (file_type, concept_label)
            tech_counts[key] = tech_counts.get(key, 0) + 1

    data_for_df = []
    for (file_type, concept), count in tech_counts.items():
        data_for_df.append({
            "File Type": file_type,
            "Technology": concept,
            "Count": count
        })

    df_counts = pd.DataFrame(data_for_df)
    df_counts["Scaled Count"] = df_counts["Count"].apply(lambda x: np.log1p(x))
    df_counts["Label"] = df_counts["Technology"] + " (" + df_counts["Count"].astype(str) + ")"

    fig = px.treemap(
        df_counts,
        path=["File Type", "Label"],
        values="Count",
        color="Count",
        color_discrete_sequence="Viridis",
        title=f"Technology Landscape Across {len(df)} Projects"
    )

    fig.update_traces(root_color="lightgrey", textfont=dict(family="Arial Bold, Arial, sans-serif", size=14, color="black"))
    fig.update_layout(
        width=1200,
        height=800,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    fig.write_image("../data/results/technology_landscape.png", scale=2)


def extract_technology_combinations(df: pd.DataFrame) -> Dict[Tuple[str, ...], int]:
    """
    Extracts all combinations of concepts (length ≥ 2) from the per-project 'technologies' column.
    Returns a dictionary mapping each combination (as a sorted tuple) to its frequency across projects.
    """
    combo_counts: Dict[Tuple[str, ...], int] = {}
    file_type_keys = set(FILE_TYPES.keys())

    for _, row in df.iterrows():
        technologies = row.get("technologies", [])
        if not isinstance(technologies, (list, tuple)) or len(technologies) < 2:
            continue

        # Normalize once per project
        concepts = []
        for t in technologies:
            norm = " ".join(str(t).lower().split("-")).strip()
            concepts.append(norm)

        concepts = sorted(set(concepts))
        if len(concepts) < 2:
            continue

        for r in range(2, len(concepts) + 1):
            for combo in itertools.combinations(concepts, r):
                # Skip combinations that contain any file type key (e.g., 'yaml', 'json', ...)
                if any(item in file_type_keys for item in combo):
                    continue
                combo_counts[combo] = combo_counts.get(combo, 0) + 1

    return combo_counts


def create_technology_combination_plot(df: pd.DataFrame, num_combos: int = 30):
    """
    Creates an UpSet plot for the most common technology combinations.
    """
    print("Hallo")

    combinations = extract_technology_combinations(df=df)
    combinations_counter = Counter(combinations)

    num_projects = len(df)

    memberships = []
    counts = []

    for combo, count in combinations_counter.most_common(num_combos):
        memberships.append(combo)
        counts.append(count)

    data = from_memberships(memberships, data=counts)

    plt.figure(figsize=(10, 6))
    plot = UpSet(data, show_counts=True, element_size=None, totals_plot_elements=0).plot()
    plot["intersections"].set_ylabel("# Projects")
    plt.suptitle(f"Technology Combinations Across {num_projects} Projects")
    plt.savefig("../data/results/technology_combinations.png")


def get_technology_statistics(project_files: List[str]):
    """
    Computes and saves technology statistics to CSV.
    """
    aggregated_data = []
    for file_name in project_files:
        logger.info(f"Analyzing {file_name}")

        with open(file_name, 'r') as f:
            data = json.load(f)
        
        latest_commit = data["latest_commit_data"]

        #concepts = latest_commit["network_data"].get("concepts", [])
        config_files = latest_commit["network_data"].get("config_file_data", [])
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
                "Project": data.get("project_name", os.path.basename(file_name)),
                "Technology": concept,
                "File Count": file_count,
                "Average Options per File": avg_options
            })
        aggregated_data.extend(result)

    df = pd.DataFrame(aggregated_data)
    tech_proj_file_counts = df.groupby(["Technology", "Project"])["File Count"].sum().reset_index()
    avg_files_per_tech_per_proj = tech_proj_file_counts.groupby("Technology")["File Count"].mean().round(2)
    avg_files_per_tech_per_proj.name = "Avg_Files_Per_Project"

    df_grouped = df.groupby("Technology").agg(
        Usage_Count=("Technology", "count"),
        Total_Projects=("Project", "nunique"),
        Total_Files=("File Count", "sum"),
        Avg_Options_Per_File=("Average Options per File", "mean")
    ).reset_index()

    df_grouped["Avg_Options_Per_File"] = df_grouped["Avg_Options_Per_File"].round(2)
    df_grouped = df_grouped.merge(avg_files_per_tech_per_proj, on="Technology")
    df_grouped.to_csv("../data/results/technology_statistics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of projects to process")
    args = parser.parse_args()
    project_files = load_project_files(args.limit)
    df_technologies = extract_technologies(project_files, "../data/results/project_technologies.csv")

    # Get technology landscape
    get_technology_landscape(df=df_technologies)

    # Get technology combinations
    create_technology_combination_plot(df=df_technologies)

    # Get technology statistics
    get_technology_statistics(project_files)