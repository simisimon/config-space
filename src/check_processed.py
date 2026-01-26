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


def main():
    parser = argparse.ArgumentParser(description="Check which projects have been processed")
    parser.add_argument("company", help="Company name (e.g., netflix)")
    args = parser.parse_args()

    company = args.company
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", company, f"{company}_projects_final.csv")
    projects_dir = os.path.join(base_dir, "data", company, "projects")

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    if not os.path.exists(projects_dir):
        print(f"Projects directory not found: {projects_dir}")
        return

    existing_folders = set(os.listdir(projects_dir))

    processed = []
    missing = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("html_url") or row.get("clone_url", "")
            name = row.get("full_name", "")
            folder_name = url_to_folder_name(url)

            if folder_name in existing_folders:
                processed.append((name, folder_name))
            else:
                missing.append((name, folder_name))

    print(f"Company: {company}")
    print(f"Total projects in CSV: {len(processed) + len(missing)}")
    print(f"Processed: {len(processed)}")
    print(f"Missing: {len(missing)}")
    print()

    if missing:
        print("Missing projects:")
        for name, folder in missing:
            print(f"  - {name} ({folder})")


if __name__ == "__main__":
    main()
