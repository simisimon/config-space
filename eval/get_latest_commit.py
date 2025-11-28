import json
import os
import glob
import re


def get_batch_number(path):
    match = re.search(r'batch_(\d+)\.json$', path)
    return int(match.group(1)) if match else -1


def get_latest_commit_data(project_dirs):
    for dir in project_dirs:
        print(f"Processing directory: {dir}")
        project_files = sorted(glob.glob(f"../data/projects/{dir}/*.json"))

        project_name = project_files[0].split("/")[-2]
        output_file = f"../data/projects_last_commit/{project_name}_last_commit.json"
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping...")
            continue
            
        if len(project_files) == 1:
            project_file = project_files[0]
        else:
            project_file = max(project_files, key=get_batch_number)

        try:
            with open(project_file, 'r') as f:
                data = json.load(f)
                
            latest_commit = data["commit_data"][-1]
        
            # Check if latest commit is indeed the latest
            if not latest_commit["is_latest_commit"]:
                raise ValueError("Latest commit is not latest according to is_latest_commit flag")
            
            # Check if latest commit  contains data
            if latest_commit["is_latest_commit"]:
                if not latest_commit["network_data"]:
                    raise ValueError("Latest commit's network data is empty")

            del data["commit_data"]

            data["latest_commit_data"] = latest_commit

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Error processing {project_file}: {e}")
            continue

if __name__ == "__main__":
    project_dirs = os.listdir("../data/projects/")
    get_latest_commit_data(project_dirs)