import argparse
import json
import os
import glob
import csv
from collections import defaultdict

# Skip common/low-information values that create huge transactions
SKIP_VALUES = {
    "true",
    "false",
    "*",
    "main",
    "master",
    "push",
    "pull_request",
}


def load_project_data(json_path):
    """Load option-value pairs from a project JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    pairs = []
    config_data = data.get("config_data", {})
    for file_entry in config_data.get("config_file_data", []):
        for pair in file_entry.get("pairs", []):
            option = pair.get("option", "")
            value = pair.get("value", "")
            if option and value:
                pairs.append((option, value))

    return data.get("project_name", os.path.basename(json_path)), pairs


def build_transactions(all_project_pairs):
    """Build transactions where each transaction is a set of options sharing the same value in a project.

    A transaction is created for each (project, value) combination that has
    at least 2 options with that value.
    """
    transactions = []

    for project_name, pairs in all_project_pairs:
        value_to_options = defaultdict(set)
        for option, value in pairs:
            # NEW: skip common values that explode transaction size
            v = str(value).strip()
            if not v:
                continue
            if v.lower() in SKIP_VALUES:
                continue

            value_to_options[v].add(option)

        for value, options in value_to_options.items():
            if len(options) >= 2:
                transactions.append({
                    "project": project_name,
                    "value": value,
                    "options": frozenset(options),
                })

    return transactions


def mine_frequent_itemsets(transactions, min_support, min_size=2, max_size=5):
    """Mine frequent itemsets using Apriori-like level-wise approach."""
    num_transactions = len(transactions)
    if num_transactions == 0:
        return {}

    min_count = max(1, int(min_support * num_transactions))

    # Level 1: count individual options
    item_counts = defaultdict(int)
    for t in transactions:
        for option in t["options"]:
            item_counts[option] += 1

    frequent_items = {
        item for item, count in item_counts.items() if count >= min_count
    }

    results = {}
    current_level_sets = [frozenset([item]) for item in frequent_items]

    # Add size-1 itemsets if requested
    if min_size <= 1:
        for item in frequent_items:
            key = frozenset([item])
            count = item_counts[item]
            projects = set()
            values = set()
            for t in transactions:
                if item in t["options"]:
                    projects.add(t["project"])
                    values.add(t["value"])
            results[key] = {
                "support": count / num_transactions,
                "count": count,
                "num_projects": len(projects),
                "projects": sorted(projects),
                "values": sorted(values),
            }

    for size in range(2, max_size + 1):
        # Generate candidates from frequent items of previous level
        candidates = set()
        for i, s1 in enumerate(current_level_sets):
            for s2 in current_level_sets[i + 1:]:
                candidate = s1 | s2
                if len(candidate) == size:
                    candidates.add(candidate)

        if not candidates:
            break

        # Count support for each candidate
        candidate_counts = defaultdict(int)
        candidate_projects = defaultdict(set)
        candidate_values = defaultdict(set)

        for t in transactions:
            for candidate in candidates:
                if candidate.issubset(t["options"]):
                    key = candidate
                    candidate_counts[key] += 1
                    candidate_projects[key].add(t["project"])
                    candidate_values[key].add(t["value"])

        # Filter by minimum support
        next_level_sets = []
        for candidate, count in candidate_counts.items():
            if count >= min_count and len(candidate) >= min_size:
                support = count / num_transactions
                results[candidate] = {
                    "support": support,
                    "count": count,
                    "num_projects": len(candidate_projects[candidate]),
                    "projects": sorted(candidate_projects[candidate]),
                    "values": sorted(candidate_values[candidate]),
                }
                next_level_sets.append(candidate)

        current_level_sets = next_level_sets
        if not current_level_sets:
            break

    return results


def save_results(results, output_path):
    rows = []
    for itemset, info in results.items():
        rows.append({
            "itemset_size": len(itemset),
            "options": " | ".join(sorted(itemset)),
            "support": round(info["support"], 4),
            "count": info["count"],
            "num_projects": info["num_projects"],
            "projects": ", ".join(info["projects"]),
            "values": ", ".join(info["values"]),
        })

    rows.sort(key=lambda r: (-r["count"], r["itemset_size"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "itemset_size", "options", "support", "count",
            "num_projects", "projects", "values",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} frequent itemsets to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute frequent itemsets of options sharing the same value across projects")
    parser.add_argument("--input", required=True, help="Input directory name (e.g., netflix). Looks for data in ../data/{input}/latest_commit/*.json")
    parser.add_argument("--min_support", type=float, default=0.01, help="Minimum support threshold")
    parser.add_argument("--min_size", type=int, default=1, help="Minimum itemset size")
    parser.add_argument("--max_size", type=int, default=5, help="Maximum itemset size")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", args.input, "latest_commit")

    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    all_project_pairs = []
    for json_file in json_files:
        project_name, pairs = load_project_data(json_file)
        all_project_pairs.append((project_name, pairs))

    print(f"Loaded {len(all_project_pairs)} projects")
    print("Building transactions...")
    transactions = build_transactions(all_project_pairs)
    print(f"Built {len(transactions)} transactions")

    results = mine_frequent_itemsets(
        transactions,
        min_support=args.min_support,
        min_size=args.min_size,
        max_size=args.max_size,
    )

    output_dir = os.path.join(script_dir, "..", "data", args.input)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "frequent_itemsets.csv")

    save_results(results, output_path)


if __name__ == "__main__":
    main()
