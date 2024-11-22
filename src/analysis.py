import pandas as pd
import json


project_files = [
    "../data/github_projects/projects_raw_1.json",
    "../data/github_projects/projects_raw_2.json",
    "../data/github_projects/projects_raw_3.json",
    "../data/github_projects/projects_raw_4.json",
    "../data/github_projects/projects_raw_5.json"
]


def get_projects():
    all_data = []

    for project_file in project_files:
        with open(project_file, "r", encoding="utf-8") as src:
            all_data += json.load(src)

    return all_data


def analyze():
    pass


def analyze_network():
    pass


def get_file_diff():
    pass


def main():
    
    project_data = get_projects()


    for project in project_data:
        
        project_name = ""
        stats = analyze()




    



if __name__ == "__main__":
    main()