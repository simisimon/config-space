import json
import matplotlib.pyplot as plt
from typing import List
from collections import defaultdict

from typing import List
import matplotlib.pyplot as plt

def plot_option_evolution(data: List):
    last_count = 0
    commit_hashes = []
    option_counts = []

    for commit in data["commit_data"]:
        commit_hash = commit["commit_hash"]
        if commit["is_config_related"]:
            last_count = commit["network_data"]["total_options"]
        commit_hashes.append(commit_hash)
        option_counts.append(last_count)

    step = max(1, len(commit_hashes) // 20)
    shortened_x = [commit_hashes[i][:10] for i in range(0, len(commit_hashes), step)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(option_counts)), option_counts, marker='o', markersize=4, linestyle='-')

    ax.set_xticks(range(0, len(option_counts), step))
    ax.set_xticklabels(shortened_x, rotation=90, fontsize=8)

    ax.set_xlabel('Commit Hash', fontsize=9)
    ax.set_ylabel('Total Options', fontsize=9)
    ax.set_title('Evolution of Configuration Options', fontsize=11)
    ax.tick_params(axis='y', labelsize=8)

    fig.tight_layout(pad=1.0)
    return fig


def plot_technology_evolution(data: List):
    concepts = set()
    all_commits = data["commit_data"]
    x_ticks = [commit["commit_hash"] for commit in all_commits]

    concept_lines = defaultdict(lambda: [None] * len(x_ticks))

    for i, commit in enumerate(all_commits):
        if commit.get("is_config_related") and "network_data" in commit and "config_files_data" in commit["network_data"]:
            concept_counts = defaultdict(int)
            for config_file in commit["network_data"]["config_files_data"]:
                concept = config_file.get("concept")
                options = config_file.get("options", 0)
                if concept:
                    concepts.add(concept)
                    concept_counts[concept] += options
            for concept, count in concept_counts.items():
                concept_lines[concept][i] = count

    # Forward fill
    for concept in concepts:
        last_value = None
        for i in range(len(x_ticks)):
            if concept_lines[concept][i] is None:
                concept_lines[concept][i] = last_value
            else:
                last_value = concept_lines[concept][i]

    shortened_x_ticks = [commit[:10] for commit in x_ticks]
    step = max(1, len(shortened_x_ticks) // 20)

    fig, ax = plt.subplots(figsize=(10, 4))
    for concept in sorted(concepts):
        ax.plot(shortened_x_ticks, concept_lines[concept], marker='o', markersize=4, label=concept)

    ax.set_xlabel("Commits", fontsize=9)
    ax.set_ylabel("Number of Options", fontsize=9)
    ax.set_title("Technology Evolution", fontsize=11)
    ax.set_xticks(range(0, len(shortened_x_ticks), step))
    ax.set_xticklabels(shortened_x_ticks[::step], rotation=90, ha="right", fontsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.legend(title="Technologies", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, title_fontsize=9)
    fig.tight_layout(pad=1.0)

    return fig

def plot_artifact_evolution(data: List):
    file_paths = set()
    file_lines = defaultdict(lambda: [None] * len(data["commit_data"]))
    x_ticks = [commit["commit_hash"] for commit in data["commit_data"]]

    # Step 1: Fill data per file for config-related commits
    for i, commit in enumerate(data["commit_data"]):
        if commit.get("is_config_related") and "network_data" in commit and "config_files_data" in commit["network_data"]:
            for config_file in commit["network_data"]["config_files_data"]:
                file_path = config_file.get("file_path")
                options = config_file.get("options", 0)
                if file_path:
                    file_paths.add(file_path)
                    file_lines[file_path][i] = options

    # Step 2: Forward-fill values for non-config commits or missing file entries
    for file_path in file_paths:
        last_value = None
        for i in range(len(x_ticks)):
            if file_lines[file_path][i] is None:
                file_lines[file_path][i] = last_value
            else:
                last_value = file_lines[file_path][i]

    # Step 3: Shorten commit hashes
    shortened_x_ticks = [commit[:7] for commit in x_ticks]

    # Step 4: Plot with larger size
    plt.figure(figsize=(20, 10))
    for file_path in sorted(file_paths):
        plt.plot(shortened_x_ticks, file_lines[file_path], marker="o", label=file_path)

    plt.xlabel("Commits")
    plt.ylabel("Number of Options")
    plt.xticks(ticks=range(0, len(shortened_x_ticks), 10), labels=shortened_x_ticks[::10], rotation=90, ha="right")
    plt.legend(title="Config Files", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()


def plot_artifact_evolution(data: List):
    file_paths = set()
    file_lines = defaultdict(lambda: [None] * len(data["commit_data"]))
    x_ticks = [commit["commit_hash"] for commit in data["commit_data"]]

    # Fill values for config-related commits
    for i, commit in enumerate(data["commit_data"]):
        if commit.get("is_config_related") and "network_data" in commit and "config_files_data" in commit["network_data"]:
            for config_file in commit["network_data"]["config_files_data"]:
                file_path = config_file.get("file_path")
                options = config_file.get("options", 0)
                if file_path:
                    file_paths.add(file_path)
                    file_lines[file_path][i] = options

    # Forward-fill missing values
    for file_path in file_paths:
        last_value = None
        for i in range(len(x_ticks)):
            if file_lines[file_path][i] is None:
                file_lines[file_path][i] = last_value
            else:
                last_value = file_lines[file_path][i]

    shortened_x_ticks = [commit[:10] for commit in x_ticks]
    step = max(1, len(shortened_x_ticks) // 20)

    # Create compact figure
    fig, ax = plt.subplots(figsize=(10, 4))
    for file_path in sorted(file_paths):
        ax.plot(shortened_x_ticks, file_lines[file_path], marker="o", markersize=4, label=file_path)

    ax.set_xlabel("Commits", fontsize=9)
    ax.set_ylabel("Number of Options", fontsize=9)
    ax.set_title("Configuration Artifact Evolution", fontsize=11)

    ax.set_xticks(range(0, len(shortened_x_ticks), step))
    ax.set_xticklabels(shortened_x_ticks[::step], rotation=90, ha="right", fontsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.legend(title="Config Files", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, title_fontsize=9)
    fig.tight_layout(pad=1.0)

    return fig