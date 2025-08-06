import os
import json


def get_project_names_from_folders():
    projects_dir = 'projects'
    project_processed = []
    folders = os.listdir(projects_dir)

    for folder in folders:
        folder_path = os.path.join(projects_dir, folder)
        if os.path.isdir(folder_path):
            has_json = any(f.endswith('.json') for f in os.listdir(folder_path))
            if has_json:
                project_processed.append(folder)
                # Uncomment the next line to delete the folder and its contents
                # shutil.rmtree(folder_path)

    return project_processed

def get_failed_project_names_from_folders():
    projects_dir = 'projects'
    project_failed = []
    folders = os.listdir(projects_dir)

    for folder in folders:
        folder_path = os.path.join(projects_dir, folder)
        if os.path.isdir(folder_path):
            has_json = any(f.endswith('.json') for f in os.listdir(folder_path))
            if not has_json:
                project_failed.append(folder)

    return project_failed


def update_projects():
    project_names = get_project_names_from_folders()
    #print(project_names)

    new_lines = []

    print("Length of project names:", len(project_names))

    with open('projects_final.txt', 'r') as f:
        lines = f.readlines()

    print("Length of lines:", len(lines))

    for line in lines:
        if not any(f"--name={project_name}" in line for project_name in project_names):
            #lines.remove(line)
            new_lines.append(line)

    print(len(new_lines))

    with open('projects_final_updated.txt', 'w') as f:
        f.writelines(new_lines)


def failed_projects():
    failed_project_names = get_failed_project_names_from_folders()

    #print(failed_project_names)

    with open('projects_final_updated.txt', 'r') as f:
        lines = f.readlines()

    failed = []
    failed_json = []
    for line in lines:
        if any(f"--name={project_name}" in line for project_name in failed_project_names):
            failed.append(line)
            line_parts = line.split(" ")
            project_name = ""
            project_url = ""
            for part in line_parts:
                if "--name" in part:
                    project_name = part.split("=")[-1].strip()
                if "--url" in part:
                    project_url = part.split("=")[-1].strip()

            if project_name and project_url:
                failed_json.append({
                    "name": project_name,
                    "url": project_url
                })

    print("Failed projects count:", len(failed_json))

    with open('projects_final_failed.txt', 'w') as f:
        f.writelines(failed)

    with open('projects_final_failed_json.json', 'w') as f:
        json.dump(failed_json, f, indent=4)


if __name__ == "__main__":
    #update_projects()
    failed_projects()