import argparse
import csv
import os
import re


def url_to_folder_name(url: str) -> str:
    """Convert a GitHub URL to the folder name format used by analysis_job.sh."""
    folder = url
    folder = re.sub(r'^https?://', '', folder)
    folder = folder.rstrip('/')
    folder = re.sub(r'\.git$', '', folder)
    folder = re.sub(r'^github\.com/', '', folder)
    folder = folder.replace('/', '_')
    return folder


def get_latest_commit_files(latest_commit_dir: str) -> set:
    """Get set of project names from latest_commit directory."""
    if not os.path.exists(latest_commit_dir):
        return set()

    project_names = set()
    for filename in os.listdir(latest_commit_dir):
        if filename.endswith("_commit.json"):
            # Remove _commit.json suffix to get project name
            project_name = filename[:-len("_commit.json")]
            project_names.add(project_name)
    return project_names


def main():
    parser = argparse.ArgumentParser(description="Check which projects have been processed")
    parser.add_argument("company", help="Company name (e.g., netflix)")
    args = parser.parse_args()

    company = args.company
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", company, f"{company}_projects_final.csv")
    projects_dir = os.path.join(base_dir, "data", company, "projects")
    latest_commit_dir = os.path.join(base_dir, "data", company, "latest_commit")

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    if not os.path.exists(projects_dir):
        print(f"Projects directory not found: {projects_dir}")
        return

    existing_folders = set(os.listdir(projects_dir))
    latest_commit_projects = get_latest_commit_files(latest_commit_dir)

    processed = []
    missing = []
    csv_project_names = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("html_url") or row.get("clone_url", "")
            name = row.get("full_name", "")
            project_name = row.get("name", "")
            folder_name = url_to_folder_name(url)

            csv_project_names.append(project_name)

            if folder_name in existing_folders:
                processed.append((name, folder_name))
            else:
                missing.append((name, folder_name))

    # Check latest_commit alignment
    csv_project_set = set(csv_project_names)
    missing_from_latest = csv_project_set - latest_commit_projects
    extra_in_latest = latest_commit_projects - csv_project_set

    print(f"Company: {company}")
    print(f"Total projects in CSV: {len(processed) + len(missing)}")
    print(f"Processed (in projects dir): {len(processed)}")
    print(f"Missing (from projects dir): {len(missing)}")
    print()

    # Latest commit directory stats
    print(f"Projects in latest_commit dir: {len(latest_commit_projects)}")
    if not os.path.exists(latest_commit_dir):
        print(f"  (directory does not exist: {latest_commit_dir})")
    print()

    if len(csv_project_set) == len(latest_commit_projects) and not missing_from_latest:
        print("CSV and latest_commit directory are aligned.")
    else:
        if missing_from_latest:
            print(f"Missing from latest_commit ({len(missing_from_latest)}):")
            for name in sorted(missing_from_latest):
                print(f"  - {name}")
        if extra_in_latest:
            print(f"Extra in latest_commit (not in CSV) ({len(extra_in_latest)}):")
            for name in sorted(extra_in_latest):
                print(f"  - {name}")
    print()

    if missing:
        print("Missing from projects directory:")
        for name, folder in missing:
            print(f"  - {name} ({folder})")


if __name__ == "__main__":
    main()
