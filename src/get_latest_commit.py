import tempfile
import shutil
import git
import os
import pandas as pd

def get_latest_commit(repo_url, temp_dir):
    """
    Clone the repository and retrieve the latest commit.
    """
    try:
        # Clone the repository
        print(f"Cloning repository {repo_url}")
        repo = git.Repo.clone_from(repo_url, temp_dir)

        # Get the latest commit
        latest_commit = repo.head.commit
        return latest_commit.hexsha

    except Exception as e:
        print(f"Error processing {repo_url}: {e}")
        

def main(csv_file_path):
    # Read the CSV file containing repository URLs
    df = pd.read_csv(csv_file_path)
    
    # Assume the column with URLs is named 'repo_url'
    repo_urls = df['repo_url']

    # Create a list to store results
    latest_commits = []

    for repo_url in repo_urls:
        # Create a temporary directory for cloning the repo
        with tempfile.TemporaryDirectory() as temp_dir:
            commit_info = get_latest_commit(repo_url, temp_dir)
            latest_commits.append(commit_info)

    # Convert the results to a DataFrame for easy export
    df["latest_commit"] = latest_commits
    df.to_csv("data/projects_with_commits.csv", index=False)

if __name__ == "__main__":
    main('data/projects.csv')
