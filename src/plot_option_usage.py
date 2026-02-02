import argparse
import json
import sys
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mapping import get_technology


def load_option_usage(data_dir: str, technology: str,
                      collect_values: bool = False) -> tuple:
    """
    Load all project config files and compute per-option usage for a technology.

    Returns:
        option_counts: dict mapping option name -> number of projects that set it
        total_projects: int
        option_values: dict mapping option name -> set of observed values
                       (only returned when *collect_values* is True)
    """
    data_path = Path(data_dir)
    option_counts = defaultdict(int)
    option_values = defaultdict(lambda: defaultdict(int)) if collect_values else None
    total_projects = 0

    json_files = list(data_path.glob("*_commit.json")) + list(data_path.glob("*_last_commit.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            if "config_data" in data:
                network_data = data["config_data"]
                concepts = network_data.get("concepts", [])
            elif "latest_commit_data" in data:
                network_data = data["latest_commit_data"].get("network_data", {})
                concepts = network_data.get("concepts", [])
            else:
                continue

            # Check if the technology is used by this project
            technology_found = technology in concepts
            if not technology_found:
                for config_file in network_data.get("config_file_data", []):
                    if config_file.get("concept") in ["yaml", "json", "xml", "toml", "configparser"]:
                        if get_technology(config_file["file_path"]) == technology:
                            technology_found = True
                            break

            if not technology_found:
                continue

            # Collect unique options (and optionally values) for the technology
            project_options = set()
            for config_file in network_data.get("config_file_data", []):
                concept = config_file.get("concept")
                is_target = concept == technology
                if not is_target and concept in ["yaml", "json", "xml", "toml", "configparser"]:
                    is_target = get_technology(config_file["file_path"]) == technology

                if is_target:
                    for pair in config_file.get("pairs", []):
                        option = pair.get("option", "")
                        if option:
                            project_options.add(option)
                            if collect_values:
                                value = pair.get("value", "")
                                if value != "":
                                    val_str = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
                                    option_values[option][val_str] += 1

            if project_options:
                total_projects += 1
                for option in project_options:
                    option_counts[option] += 1

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    if collect_values:
        return dict(option_counts), total_projects, {k: dict(v) for k, v in option_values.items()}
    return dict(option_counts), total_projects


def plot_option_histogram(option_counts: dict, total_projects: int,
                          technology: str, output_path: str, top_n: int = None):
    """Create a bar chart of option usage percentages."""
    if not option_counts:
        print("No options found for this technology.")
        return

    # Compute percentages and sort descending
    options = sorted(option_counts.keys(),
                     key=lambda o: option_counts[o], reverse=True)
    if top_n is not None:
        options = options[:top_n]
    percentages = [(option_counts[o] / total_projects) * 100 for o in options]

    fig, ax = plt.subplots(figsize=(max(10, len(options) * 0.35), 8))
    x = np.arange(len(options))
    ax.bar(x, percentages, color="steelblue", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Configuration Option")
    ax.set_ylabel("% of Projects Using This Option")
    title = f"Option Usage for '{technology}' (n={total_projects} projects)"
    if top_n is not None:
        title += f" — top {len(options)}"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(options, rotation=90, ha="right", fontsize=8)
    ax.set_xlim(-0.5, len(options) - 0.5)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a histogram of configuration option usage for a technology."
    )
    ALL_COMPANIES = ["netflix", "uber", "disney", "airbnb", "google", "facebook"]

    parser.add_argument("--input", default=None,
                        help="Company/directory name (e.g. netflix)")
    parser.add_argument("--technology", required=True,
                        help="Technology to analyse (e.g. github-action)")
    parser.add_argument("--all", action="store_true", dest="all_companies",
                        help="Aggregate across all companies")
    parser.add_argument("--top", type=int, default=None,
                        help="Only show the top N most common options")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: auto-generated)")
    args = parser.parse_args()

    if not args.all_companies and not args.input:
        parser.error("Either --input or --all is required")

    if args.all_companies:
        combined_counts = defaultdict(int)
        combined_total = 0
        for company in ALL_COMPANIES:
            data_dir = f"../data/{company}/latest_commit"
            if not Path(data_dir).exists():
                print(f"Skipping {company}: {data_dir} not found")
                continue
            counts, total = load_option_usage(data_dir, args.technology)
            print(f"  {company}: {total} projects, {len(counts)} options")
            combined_total += total
            for option, count in counts.items():
                combined_counts[option] += count

        option_counts = dict(combined_counts)
        total_projects = combined_total
        label = "all companies"

        output_path = args.output
        if output_path is None:
            out_dir = Path("../data/all_companies/option_usage")
            os.makedirs(out_dir, exist_ok=True)
            output_path = str(out_dir / f"{args.technology}_option_usage.png")
    else:
        data_dir = f"../data/{args.input}/latest_commit"
        if not Path(data_dir).exists():
            print(f"Data directory not found: {data_dir}")
            sys.exit(1)
        option_counts, total_projects = load_option_usage(data_dir, args.technology)
        label = args.input

        output_path = args.output
        if output_path is None:
            out_dir = Path(f"../data/{args.input}/option_usage")
            os.makedirs(out_dir, exist_ok=True)
            output_path = str(out_dir / f"{args.technology}_option_usage.png")

    print(f"Found {total_projects} projects using '{args.technology}' "
          f"with {len(option_counts)} distinct options ({label})")

    if total_projects == 0:
        print("No projects found for this technology.")
        sys.exit(0)

    plot_option_histogram(option_counts, total_projects, args.technology,
                          output_path, top_n=args.top)
