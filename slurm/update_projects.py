import os
import json


def get_all_project_names_from_folders():
    projects_dir = 'projects'
    all_project_processed = []
    folders = os.listdir(projects_dir)

    for folder in folders:
        folder_path = os.path.join(projects_dir, folder)
        if os.path.isdir(folder_path):
            all_project_processed.append(folder)

    return all_project_processed

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



def failed_projects():
    failed_project_names = get_failed_project_names_from_folders()

    with open('projects_final.txt', 'r') as f:
        lines = f.readlines()

    failed = []
    failed_json = []
    for line in lines:
        line_parts = line.split(" ")
        project_name = line_parts[3].split("=")[-1].strip()
        project_url = line_parts[2].split("=")[-1].strip()
        for failed_project_name in failed_project_names:
            if "_" in failed_project_name:
                name = failed_project_name.split("_")[-1]
                account = failed_project_name.split("_")[0]
            
                if name == project_name and account in project_url:
                    failed.append(line)
                    failed_json.append({"name": name, "url": project_url})
                    break

            else:
                name = failed_project_name
                if name == project_name:
                    failed.append(line)
                    failed_json.append({"name": name, "url": project_url})
                    break
    
    print("Failed projects count:", len(failed))

    with open('projects_final_failed.json', 'w') as f:
        json.dump(failed_json, f, indent=4)

    with open('projects_final_failed.txt', 'w') as f:
        f.writelines(failed)


def missing_projects():

    with open('projects_final.txt', 'r') as f:
        lines = f.readlines()

    print("Total projects:", len(lines))

    all_project_processed = set([name.strip().lower() for name in get_all_project_names_from_folders()])

    print("Processed projects:", len(all_project_processed))

    print("Expected missing projects:", len(lines) - len(all_project_processed))

    # Extrahiere alle (name, url)-Kombinationen aus der Datei
    all_projects= set()
    for line in lines:
        name = None
        account = None
        parts = line.split()
        for part in parts:
            if part.startswith("--name="):
                name = part.split("=", 1)[-1].strip().lower()
            if part.startswith("--url="):
                account = part.split("=", 1)[-1].split("/")[-2].strip().lower()
        if name is not None and account is not None:
            full_name = f"{account}_{name}"
            key = (name, full_name, line)
            all_projects.add(key)
    
    #print("All projects:", all_project_names)

    missed_projects = []
    for project in all_projects:
        if project[0] in all_project_processed or project[1] in all_project_processed:
            continue
        else:
            missed_projects.append(project[2])
            continue

    print("Missed projects count:", len(missed_projects))
    print("Missed projects:")
    for project in missed_projects:
        print(f"{project}")

def get_projects_with_equal_name():    
    
    with open('projects_final.txt', 'r') as f:
        lines = f.readlines()

    # Finde und drucke alle Projektnamen, die mehrfach mit unterschiedlichen URLs vorkommen
    from collections import defaultdict
    name_to_lines = defaultdict(list)
    for line in lines:
        pname = None
        purl = None
        parts = line.split()
        for part in parts:
            if part.startswith("--name="):
                pname = part.split("=", 1)[1].strip().lower()
            if part.startswith("--url="):
                purl = part.split("=", 1)[1].strip().lower()
        if pname is not None and purl is not None:
            name_to_lines[pname].append(line.strip())

    print("Projekte mit gleichem Namen aber unterschiedlichen URLs:")
    for pname, llist in name_to_lines.items():
        if len(llist) > 1:
            print(f"Projektname: {pname}")
            for l in llist:
                print(f"  {l}")


if __name__ == "__main__":
    failed_projects()
    #missing_projects()
    #get_projects_with_equal_name()