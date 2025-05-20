import pandas as pd
from typing import List
from collections import defaultdict
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def get_contributors_stats(project_data: List) -> pd.DataFrame:
    # Extract relevant commit data
    commit_data = project_data.get("commit_data", [])

    # Dictionary to store contributor stats
    contributors_stats = defaultdict(lambda: {
        "config_commits": 0,
        "non_config_commits": 0,
        "files_changed": defaultdict(int)
    })

    # Process each commit
    for commit in commit_data:
        author = commit["author"].lower()
        is_config_related = commit["is_config_related"]
        changed_files = commit["network_data"].get("config_file_data", []) if commit["network_data"] else []

        # Count config and non-config commits
        if is_config_related:
            contributors_stats[author]["config_commits"] += 1
        else:
            contributors_stats[author]["non_config_commits"] += 1

        # Count files changed per contributor
        for file in changed_files:
            if file["is_modified"]:
                contributors_stats[author]["files_changed"][file["file_path"]] += 1

    # Create two separate DataFrames: one for commit statistics and one for changed files
    commit_stats_rows = []
    changed_files_rows = []

    for contributor, stats in contributors_stats.items():
        commit_stats_rows.append({
            "Contributor": contributor,
            "Config Commits": stats["config_commits"],
            "Non-Config Commits": stats["non_config_commits"]
        })
        
        for file, count in stats["files_changed"].items():
            changed_files_rows.append({
                "Contributor": contributor,
                "Changed File": file,
                "File Change Count": count
            })

    # Convert to DataFrames
    df_commits_stats = pd.DataFrame(commit_stats_rows)
    df_changed_files = pd.DataFrame(changed_files_rows)
    df_commits_stats = df_commits_stats.sort_values(by="Config Commits", ascending=False)

    return df_commits_stats, df_changed_files


def plot_contributors_and_files(df_changed_files: pd.DataFrame):
    pivot_df = df_changed_files.pivot(index="Contributor", columns="Changed File", values="File Change Count").fillna(0)
    fig, ax = plt.subplots(figsize=(20, 10))  # Match size with plot_artifact_evolution
    sns.heatmap(pivot_df, cmap="Oranges", linewidths=0.5, annot=False, ax=ax)
    ax.set_xlabel("Changed Configuration Files")
    ax.set_ylabel("Contributors")
    return fig