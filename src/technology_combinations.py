import logging
import argparse
import glob
import json
import numpy as np
from typing import List
from upsetplot import UpSet, from_memberships
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


FILE_TYPES = ["yaml", "properties", "json", "xml", "toml"]


def load_project_files(limit: int | None = None):
    project_files = glob.glob("../data/projects/*.json")
    logger.info(f"Found {len(project_files)} project files")
    if limit:
        project_files = project_files[:limit]
        logger.info(f"Using {len(project_files)} project files")
    return project_files



from typing import List, Dict, Tuple
import json
import itertools
from collections import Counter

def extract_technology_combinations(project_files: List[str]) -> Dict[Tuple[str, ...], int]:
    """
    Extracts all combinations of concepts (length ≥ 2) from the latest commit of each project.
    Returns a dictionary mapping each combination (as a sorted tuple) to its frequency across projects.
    """
    combo_counts: Dict[Tuple[str, ...], int] = {}

    for project_file in project_files:
        logger.info(f"Processing {project_file}")
        try:
        
            with open(project_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            commit_data = data.get("commit_data", [])
            latest_commit = next((c for c in commit_data if c.get("is_latest_commit")), None)
            if not latest_commit:
                continue

            concepts = sorted(set(latest_commit["network_data"].get("concepts", [])))
            if len(concepts) < 2:
                continue

            for r in range(2, len(concepts) + 1):
                for combo in itertools.combinations(concepts, r):
                    if any(file_type in combo for file_type in FILE_TYPES):
                        # Skipping technology combination as it contains a file type"
                        continue
                    combo_counts[combo] = combo_counts.get(combo, 0) + 1
        except Exception as e:
            # Handle any errors that occur during file reading or JSON parsing
            logger.error(f"Error processing {project_file}: {e}")
            continue
        
    return combo_counts


def create_technology_combination_plot(combos_counter: Counter, num_combos: int = 20):
    memberships = []
    counts = []

    for combo, count in combos_counter.most_common(num_combos):
        memberships.append(combo)
        counts.append(count)

    data = from_memberships(memberships, data=counts)

    # Plot
    plt.figure(figsize=(10, 6))
    plot = UpSet(data, show_counts=True, element_size=None, totals_plot_elements=0).plot()
    plot["intersections"].set_ylabel("# Projects")
    #plt.suptitle("Technology Combinations Across Projects")
    plt.savefig("../data/results/technological_composition/technology_combinations.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of projects to process")
    args = parser.parse_args()
    project_files = load_project_files(args.limit)
    combinations = extract_technology_combinations(project_files)
    combinations_counter = Counter(combinations)
    create_technology_combination_plot(combinations_counter)