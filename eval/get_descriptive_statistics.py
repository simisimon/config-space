import pandas as pd
import json
import os
from typing import Dict, Any

def get_descriptive_statistics(project_csv: str, results_files: str):
    projects_df = pd.read_csv(project_csv)

    with open(results_files, "r") as results_f:
        results_data = json.load(results_f)

    for _, row in projects_df.iterrows():
        print("Processing project:", row["full_name"])
        name = row["name"]
        full_name = row["full_name"].replace("/", "_")

        #if full_name in results_data:
        #    print("Skipping already processed project:", full_name)
        #    continue

        project_stats = {
            "total_commits": 0,
            "total_config_commits": 0,
            "total_contributors": 0,
            "total_config_contributors": 0,
            "total_technologies": 0,
            "num_stars": row["stargazers_count"],
            "created_at": row["created_at"],
            "size_in_kb": row["size"],
            
        }

        # Get technologies used
        technologies = get_technologies(name, full_name)
        project_stats["total_technologies"] = technologies

        # Get contributors data
        total_contributors, total_config_contributors = get_contributors(name, full_name)
        project_stats["total_contributors"] = total_contributors
        project_stats["total_config_contributors"] = total_config_contributors

        # Get commits
        total_commits, total_config_commits = get_commits(name, full_name)
        project_stats["total_commits"] = total_commits
        project_stats["total_config_commits"] = total_config_commits

        # Add project statistics to results data
        results_data[full_name] = project_stats

        with open(results_files, "w") as results_f:
            json.dump(results_data, results_f, indent=4)


def get_technologies(name: str, full_name: str) -> int:
    project_files = os.listdir("../data/projects_last_commit")
    project_file_full_name = f"{full_name}_last_commit.json"
    project_file_name = f"{name}_last_commit.json"

    if project_file_full_name in project_files:
        project_file = project_file_full_name
    elif project_file_name in project_files:
        project_file = project_file_name
    else:
        print("No last commit file found for project:", full_name)
        return
    
    with open(f"../data/projects_last_commit/{project_file}", "r") as f:
        project_data = json.load(f)
        technologies = project_data["latest_commit_data"]["network_data"].get("concepts", [])
        return len(technologies)

def get_commits(name: str, full_name: str) -> tuple[int, int]:
    contributors_statistics = pd.read_csv("../data/contributor_statistics.csv")
    for _, row in contributors_statistics.iterrows():
        if row["project_name"] == full_name:
            return row["total_commits"], row["total_config_commits"]
        elif row["project_name"] == name:
            return row["total_commits"], row["total_config_commits"]
        
    return None, None

def get_contributors(name: str, full_name: str) -> tuple[int, int]:
    contributors_statistics = pd.read_csv("../data/contributor_statistics.csv")
    for _, row in contributors_statistics.iterrows():
        if row["project_name"] == full_name:
            return row["total_contributors"], row["config_contributors"]
        elif row["project_name"] == name:
            return row["total_commits"], row["total_config_commits"]
    
    return None, None

def analyze_descriptive_statistics(results_file: str) -> Dict[str, Any]:
    """
    Analyze the descriptive statistics data and return comprehensive insights.

    Args:
        results_file: Path to the descriptive_statistics_results.json file

    Returns:
        Dictionary containing analysis results with statistics and insights
    """
    with open(results_file, "r") as f:
        data = json.load(f)

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame.from_dict(data, orient='index')

    # Projects with data are those with commits and contributors (not null)
    df_clean = df[df['total_commits'].notna() & df['total_contributors'].notna()]

    analysis_results = {
        "total_projects": len(data),
        "projects_with_data": len(df_clean),
        "projects_with_missing_data": len(data) - len(df_clean),
        "statistics": {}
    }

    # Numeric columns to analyze
    numeric_columns = [
        "total_commits", "total_config_commits", "total_contributors",
        "total_config_contributors", "total_technologies", "num_stars", "size_in_kb"
    ]

    # Compute statistics for each numeric column
    for col in numeric_columns:
        if col in df_clean.columns:
            col_data = df_clean[col].dropna()
            if len(col_data) > 0:
                analysis_results["statistics"][col] = {
                    "count": int(len(col_data)),
                    "mean": round(float(col_data.mean()), 2),
                    "median": round(float(col_data.median()), 2),
                    "std": round(float(col_data.std()), 2),
                    "min": round(float(col_data.min()), 2),
                    "max": round(float(col_data.max()), 2),
                    "q25": round(float(col_data.quantile(0.25)), 2),
                    "q75": round(float(col_data.quantile(0.75)), 2)
                }

    # Calculate derived metrics
    df_clean['config_commit_ratio'] = df_clean['total_config_commits'] / df_clean['total_commits']
    df_clean['config_contributor_ratio'] = df_clean['total_config_contributors'] / df_clean['total_contributors']

    # Add derived metrics to analysis
    for metric in ['config_commit_ratio', 'config_contributor_ratio']:
        col_data = df_clean[metric].dropna().replace([float('inf'), -float('inf')], pd.NA).dropna()
        if len(col_data) > 0:
            analysis_results["statistics"][metric] = {
                "count": int(len(col_data)),
                "mean": round(float(col_data.mean()), 4),
                "median": round(float(col_data.median()), 4),
                "std": round(float(col_data.std()), 4),
                "min": round(float(col_data.min()), 4),
                "max": round(float(col_data.max()), 4),
                "q25": round(float(col_data.quantile(0.25)), 4),
                "q75": round(float(col_data.quantile(0.75)), 4)
            }

    # Find top projects by various metrics
    analysis_results["top_projects"] = {}

    for col in ["num_stars", "total_commits", "total_technologies", "size_in_kb"]:
        if col in df_clean.columns:
            top_5 = df_clean.nlargest(5, col)[col].to_dict()
            analysis_results["top_projects"][f"top_5_by_{col}"] = {
                str(k): round(float(v), 2) if pd.notna(v) else None for k, v in top_5.items()
            }

    # Correlation analysis
    correlation_cols = [col for col in numeric_columns if col in df_clean.columns]
    if len(correlation_cols) > 1:
        correlations = df_clean[correlation_cols].corr()
        analysis_results["correlations"] = {
            "config_commits_vs_total_commits": round(float(correlations.loc["total_config_commits", "total_commits"]), 4)
                if "total_config_commits" in correlations.columns and "total_commits" in correlations.index else None,
            "config_contributors_vs_total_contributors": round(float(correlations.loc["total_config_contributors", "total_contributors"]), 4)
                if "total_config_contributors" in correlations.columns and "total_contributors" in correlations.index else None,
            "technologies_vs_stars": round(float(correlations.loc["total_technologies", "num_stars"]), 4)
                if "total_technologies" in correlations.columns and "num_stars" in correlations.index else None,
            "size_vs_commits": round(float(correlations.loc["size_in_kb", "total_commits"]), 4)
                if "size_in_kb" in correlations.columns and "total_commits" in correlations.index else None,
        }

    return analysis_results

if __name__ == "__main__":
    project_csv = "../data/projects_final.csv"
    results_files = "../data/descriptive_statistics_results.json"

    # Uncomment to collect descriptive statistics
    # get_descriptive_statistics(project_csv, results_files)

    # Analyze the descriptive statistics
    analysis = analyze_descriptive_statistics(results_files)

    # Print analysis results
    print(json.dumps(analysis, indent=2))
