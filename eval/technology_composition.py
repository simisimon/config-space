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
import ast
import itertools
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Tuple
from upsetplot import UpSet, from_memberships
from mapping import get_technology


import kaleido
kaleido.get_chrome_sync()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True, stream=sys.stdout)
logger = logging.getLogger(__name__)


EXCLUDED_DIRS = ("docs/", "data/", "lib/", "benchmark/", "annotations/", "examples/")

FILE_TYPES = {
    "yaml": ["dependabot", "codecov", "buildkite", "ansible", "ansible playbook", "kubernetes", "docker compose", 
             "github-action", "goreleaser", "mkdocs", "swiftlint", "sourcery", "circleci", "elasticsearch", 
             "flutter", "mockery", "codeclimate", "heroku", "spring", "travis", "bandit", "amplify", "drone", 
             "yaml", "buf", "github", "gitpod", "appveyor", "pnpm", "rubocop", "gitbook", "jitpack", "pre-commit", 
             "snapscraft", "eslint", "markdownlint", "stylelint", "postcss", "mocha", "yarn", "golangci-lint", "jekyll"
             "codebeaver", "crowdin", "readthedocs", "graphql", "swagger", "chart testing", "codebuild", "lefthook",
             "hugoreleaser", "triagebot", "jazzy", "clomonitor", "prometheus", "helm", "cspell", "azure pipelines", "logstash"
             "verdaccio", "github", "github issues", "github funding", "github config", "github codespaces"],
    "properties": ["alluxio", "spring", "kafka", "gradle", "cirrus", "gradle wrapper", "maven wrapper", "properties", "log4j"],
    "json": ["angular", "eslint", "prettier", "lerna", "firebase", "renovate", "stripe", "tsconfig", "nodejs", 
             "vercel", "npm", "cypress", "devcontainer", "deno", "cmake", "bower", "json", "babel", "turborepo", 
             "vscode", "apify", "gocrazy", "jest", "markdownlint", "stylelint", "postcss", "mocha", "golangci-lint", "wrangler",
             "vcpkg", "changesets", "fuel", "knip", "tsdoc", "nodemon", "graphql", "swagger", "nixpacks", "lefthook",
             "bundlemon", "cspell"],
    "xml": ["maven", "android", "hadoop common", "hadoop hbase", "hadoop hdfs", "mapreduce", "xml", "yarn", "log4j"],
    "toml": ["cargo", "netlify", "poetry", "toml", "rustfmt", "flyio", "taplo", "cross", "cargo make", "stylua", 
             "trunk", "rust", "clippy", "ruff", "typos", "golangci-lint", "jekyll", "wrangler", "graphql",
             "nixpacks", "lefthook", "deepsource", "git cliff", "reuse", "kodiak", "streamlit", "mdbook", "lychee",
             "cdbindgen", "triagebot", "taoskeeper", "cspell"],
    "conf": ["mongodb", "nginx", "postgresql", "rabbitmq", "redis", "apache", "conf"],
    "ini": ["mysql", "php", "ini", "mypy", "tox", "grafana"],
    "cfg": ["zookeeper"],
    "python": ["django"],
    "other": ["docker"],
}

COLORS = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
    "#80b1d3", "#fdb462", "#b3de69", "#fccde5",
    "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"
]


def load_project_files(limit: int | None = None, refresh: bool = False):
    if not refresh:
        logger.info("Skipping project file loading")
        return
    
    project_files = glob.glob("../data/projects_last_commit/*.json")
    logger.info(f"Found {len(project_files)} project files")
    if limit:
        project_files = project_files[:limit]
        logger.info(f"Using {len(project_files)} project files")
    return project_files


def extract_technologies(project_files: List[str], output_file: str, refresh: bool = False) -> pd.DataFrame:
    """
    Extracts technologies from project configuration files and saves to CSV.
    """
    if os.path.exists(output_file) and not refresh:
        logger.info(f"Loading existing technology data from {output_file}")
        return pd.read_csv(output_file)

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
                    # Skip files in exlcuded directories
                    if config_file["file_path"].startswith(EXCLUDED_DIRS):
                        continue

                    if config_file["concept"] in ["json", "xml", "yaml", "toml", "configparser"]:
                        concept = get_technology(config_file["file_path"])
                        if concept:
                            technologies.append(concept)
                        else:
                            if config_file["concept"].lower() == "nodejs":
                                technologies.append("npm")
                            else:
                                technologies.append(config_file["concept"])
                    else:
                        if config_file["concept"].lower() == "nodejs":
                            technologies.append("npm")
                        else:
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


def get_technology_landscape(data_file: str, output_file: str, refresh: bool = False):
    """
    Creates a treemap visualization of the technology landscape.
    """
    if not refresh:
        logger.info("Skipping technology landscape generation")
        return

    df = pd.read_csv(data_file)
    concept_to_filetype = build_concept_to_filetype(FILE_TYPES)

    tech_counts = {}
    for _, row in df.iterrows():
        project_technologies = row["technologies"]
        # Parse string representation of list to actual list
        if isinstance(project_technologies, str):
            project_technologies = ast.literal_eval(project_technologies)
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
        color="File Type",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title=f"Technology Landscape Across {len(df)} Projects"
    )

    fig.update_traces(
        root_color="lightgrey",
        textfont=dict(family="Arial Black, Arial Bold, sans-serif", size=16, color="black"),
        marker=dict(line=dict(width=2, color="white"))
    )
    fig.update_layout(
        width=1200,
        height=800,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    fig.write_image(output_file, scale=2)


def extract_technology_combinations(df: pd.DataFrame, max_combos: int = 10) -> Dict[Tuple[str, ...], int]:
    """
    Extracts all combinations of concepts (length ≥ 2) from the per-project 'technologies' column.
    Returns a dictionary mapping each combination (as a sorted tuple) to its frequency across projects.
    """
    combo_counts: Dict[Tuple[str, ...], int] = {}
    file_type_keys = set(FILE_TYPES.keys())

    for _, row in df.iterrows():
        technologies = row.get("technologies", [])
        # Parse string representation of list to actual list
        if isinstance(technologies, str):
            technologies = ast.literal_eval(technologies)
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
        

        for r in range(2, max_combos):
            for combo in itertools.combinations(concepts, r):
                # Skip combinations that contain any file type key (e.g., 'yaml', 'json', ...)
                if any(item in file_type_keys for item in combo):
                    continue
                combo_counts[combo] = combo_counts.get(combo, 0) + 1

    return combo_counts


def create_technology_combination_plot(data_file: str, num_combos: int, refresh: bool = False):
    """
    Creates an UpSet plot for the most common technology combinations.
    """
    if not refresh:
        logger.info("Skipping technology combination plot generation")
        return

    df = pd.read_csv(data_file)
    combinations = extract_technology_combinations(df=df)
    combinations_counter = Counter(combinations)

    num_projects = len(df)

    memberships = []
    counts = []

    for combo, count in combinations_counter.most_common(num_combos):
        memberships.append(combo)
        counts.append(count)

    data = from_memberships(memberships, data=counts)

    # Create figure with better proportions
    plt.figure(figsize=(20, 14))

    # Create UpSet plot with visible labels
    plot = UpSet(
        data,
        show_counts=True,
        element_size=50,  # Circle size
        totals_plot_elements=0,
        min_subset_size=0,
        sort_by='cardinality',
        sort_categories_by=None
    ).plot()

    # Adjust label sizes and spacing
    plot["intersections"].set_ylabel("# Projects", fontsize=16, fontweight='bold')
    plot["intersections"].tick_params(axis='y', labelsize=12)
    plot["intersections"].tick_params(axis='x', labelsize=10)

    # Set the matrix y-axis labels (technology names)
    if "matrix" in plot and plot["matrix"] is not None:
        plot["matrix"].tick_params(axis='y', labelsize=14)
        labels = plot["matrix"].get_yticklabels()
        plot["matrix"].set_yticklabels(labels, fontweight='bold')

    plt.suptitle(f"Technology Combinations Across {num_projects} Projects", fontsize=20, fontweight='bold', y=0.995)
    plt.savefig("../data/technology_composition/technology_combinations.png", dpi=300, bbox_inches='tight', pad_inches=0.3)


def filter_technologies(technologies_str):
    if isinstance(technologies_str, str):
        technologies = ast.literal_eval(technologies_str)
    else:
        technologies = technologies_str

    # Remove file types
    filtered = [tech for tech in technologies if tech.lower() not in ["yaml", "json", "toml", "xml", "configparser", "properties"]]
    return filtered

def filter_projects(data_file: str, output_file: str, refresh: bool = False):
    """
    Filters projects with less than two technologies after removing file types.
    """
    if not refresh:
        logger.info("Skipping filtering as refresh is False.")
        return

    df = pd.read_csv(data_file)
    df["technologies"] = df["technologies"].apply(filter_technologies)
    df["tech_count"] = df["technologies"].apply(len)
    df_filtered = df[df["tech_count"] >= 2].drop(columns=["tech_count"])
    df_filtered.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of projects to process")
    parser.add_argument("--refresh", action="store_true", help="Refresh technology extraction even if CSV exists")  
    args = parser.parse_args()
    
    # Load project files
    #project_files = load_project_files(
    #    args.limit, 
    #    refresh=args.refresh
    #)

    # Extract technologies
    #df_technologies = extract_technologies(
    #    project_files=project_files, 
    #    output_file="../data/technology_composition/project_technologies.csv", 
    #    refresh=args.refresh
    #)

    # Get technology landscape
    #get_technology_landscape(
    #    data_file="../data/technology_composition/project_technologies.csv", 
    #    output_file="../data/technology_composition/technology_landscape.png",
    #    refresh=args.refresh
    #)

    #filter_projects(
    #    data_file="../data/technology_composition/project_technologies.csv", 
    #    output_file="../data/technology_composition/project_technologies_filtered.csv",
    #    refresh=args.refresh
    #)

    # Create new landscape with filtered data
    #get_technology_landscape(
    #    data_file="../data/technology_composition/project_technologies_filtered.csv", 
    #    output_file="../data/technology_composition/technology_landscape_filtered.png",
    #    refresh=args.refresh
    #)

    # Get technology combinations
    create_technology_combination_plot(
        data_file="../data/technology_composition/project_technologies_filtered.csv", 
        num_combos=50,
        refresh=args.refresh
    )