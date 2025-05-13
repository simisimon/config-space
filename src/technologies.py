import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import javaproperties
import pandas as pd
import re
from typing import List, Set, Dict

def parse_properties_file(file_path: str) -> set:
    with open(file_path, "r", encoding="utf-8") as f:
        props = javaproperties.load(f)
    return set(k.strip().upper() for k in props.keys())


def show_options_per_technology(technology_files):
    data = []

    common_file_types = ["json", "yaml", "configparser"]

    for file_path in technology_files:
        technology = os.path.basename(file_path).replace(".properties", "")
        if technology in common_file_types:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            props = javaproperties.load(f)
            total = len(props)
            with_defaults = sum(1 for v in props.values() if v is not None and str(v).strip() != "")
        
        data.append((technology, total, with_defaults))

    df = pd.DataFrame(data, columns=["Technology", "Total Options", "With Defaults"])
    df.sort_values(by="Total Options", ascending=False, inplace=True)

    # Plotting grouped bars
    x = range(len(df))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([i - width/2 for i in x], df["Total Options"], width=width, label="Total Options", color="skyblue")
    ax.bar([i + width/2 for i in x], df["With Defaults"], width=width, label="With Default Values", color="orange")

    ax.set_xlabel('Technology', fontsize=9)
    ax.set_ylabel('Number of Options', fontsize=9)
    ax.set_title('Options per Technology (Total vs. With Defaults)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Technology"], rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)

    fig.tight_layout(pad=1.0)
    return fig


def parse_properties_file(file_path: str) -> set:
    with open(file_path, "r", encoding="utf-8") as f:
        props = javaproperties.load(f)
    return set(k.strip() for k in props.keys())




def get_matches(project_options: Set[str], ref_options: Set[str]) -> Dict[str, List[str]]:
    matched_ref_to_project = {}

    for ref_opt in ref_options:
        pattern = '^' + re.escape(ref_opt).replace(r'\*', r'.+') + '$'
        regex = re.compile(pattern)
        
        matches = [opt for opt in project_options if regex.fullmatch(opt)]
        if matches:
            matched_ref_to_project[ref_opt] = matches

    return matched_ref_to_project



def get_options_per_project(technology_files: List, df_option: pd.DataFrame):
    project_options = df_option.copy()
    project_options["option"] = project_options["option"].str.strip()

    # Prepare result table
    results = []

    # Iterate over all reference files
    for technology_file in technology_files:
        technology = technology_file.split("/")[-1].split(".properties")[0]
        ref_options = parse_properties_file(technology_file)

        project_subset = set(project_options[project_options["concept"].str.lower() == technology.lower()]["option"])
        ref_to_proj = get_matches(project_subset, ref_options)
        matched_refs = set(ref_to_proj.keys())
        matched_project_options = sorted({opt for opts in ref_to_proj.values() for opt in opts})
        unmatched = [opt for opt in project_subset if opt not in matched_project_options]

        print("technology:", technology)
        print("Unmatched Options:", unmatched)

        results.append({
            "Technology": technology,
            "Total Options": len(ref_options),
            "Options Set": len(project_subset),
            "Matched Options": len(matched_refs),
            "Unmatched Options": len(unmatched),
            "Percentage Used": round(len(matched_refs) / len(ref_options) * 100, 2) if ref_options else 0.0,
            "Matched": list(matched_project_options)
        })


    return pd.DataFrame(results)

