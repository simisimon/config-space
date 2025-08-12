import logging
import argparse
import glob
import json
import pandas as pd
import plotly.express as px
import numpy as np
from typing import List

import kaleido
kaleido.get_chrome_sync()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    project_files = glob.glob("../data/projects/*.json")
    logger.info(f"Found {len(project_files)} project files")
    if limit:
        project_files = project_files[:limit]
        logger.info(f"Using {len(project_files)} project files")
    return project_files


def create_technology_landscape(project_files: List[str]):
    concept_to_filetype = {}
    for ext, concepts in FILE_TYPES.items():
        for concept in concepts:
            concept_to_filetype[concept.lower()] = ext
        tech_counts = {}

    for project_file in project_files:
        logger.info(f"Processing {project_file}")
        try:
            with open(project_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                latest_commit = next(
                    filter(lambda commit: commit["is_latest_commit"] == True, data["commit_data"]), None
                )

                config_data = latest_commit["network_data"]["config_file_data"]

                for config_file in config_data:
                    raw_concept = config_file["concept"]
                    normalized_concept = " ".join(raw_concept.lower().split("-")).strip()
                    file_type = concept_to_filetype.get(normalized_concept, "other")

                    # Rename technology to 'Other' only if concept exactly equals file_type
                    concept_label = "other" if normalized_concept == file_type else normalized_concept

                    key = (file_type, concept_label)
                    tech_counts[key] = tech_counts.get(key, 0) + 1

        except Exception as e:
            logging.error(f"Error processing {project_file}: {e}")
            continue

    data_for_df = []
    for (file_type, concept), count in tech_counts.items():
        data_for_df.append({
            "File Type": file_type,
            "Technology": concept,
            "Count": count
        })

    df = pd.DataFrame(data_for_df)

    # Log-scaling for visualization
    df["Scaled Count"] = df["Count"].apply(lambda x: np.log1p(x))  # log(1 + x)

    # Optional: Enrich label for display
    df["Label"] = df["Technology"] + " (" + df["Count"].astype(str) + ")"
    
    # Plot
    fig = px.treemap(
        df,
        path=["File Type", "Label"],
        values="Scaled Count",
        color="File Type",
        color_discrete_sequence=COLORS,
        #title="Technology Landscape"
    )

    fig.update_traces(root_color="lightgrey", textfont=dict(family="Arial Bold, Arial, sans-serif", size=14, color="black"))
    fig.update_layout(
        width=1200,
        height=800,
        margin=dict(t=50, l=25, r=25, b=25)
    )
    fig.write_image("../data/results/technological_composition/technology_landscape.png", scale=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of projects to process")
    args = parser.parse_args()
    project_files = load_project_files(args.limit)
    create_technology_landscape(project_files)
