import tempfile
import shutil
import git
import os




def analyze_repository(repo_path):
    # Placeholder for the analysis logic
    # Example: Count number of Python files
    python_files = [f for f in os.listdir(repo_path) if f.endswith('.py')]
    print(f"Number of Python files: {len(python_files)}")

def main():
    # Step 1: Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory at {temp_dir}")

    try:
        # Step 2: Clone the GitHub repository into the temporary directory
        repo_url = "https://github.com/your/repo.git"  # Replace with the actual repo URL
        print(f"Cloning repository {repo_url} into {temp_dir}")
        repo = git.Repo.clone_from(repo_url, temp_dir)
        
        # Step 3: Extract the latest commit
        latest_commit = repo.head.commit
        print(f"Latest commit: {latest_commit.hexsha}")
        print(f"Author: {latest_commit.author.name}")
        print(f"Date: {latest_commit.committed_datetime}")
        print(f"Message: {latest_commit.message}")

        # Step 4: Perform analysis on the repository
        analyze_repository(temp_dir)
    
    finally:
        # Step 5: Clean up the temporary directory
        print(f"Cleaning up temporary directory at {temp_dir}")
        shutil.rmtree(temp_dir)
        print("Temporary directory removed")

if __name__ == "__main__":
    main()
