import argparse
import json
import os
import glob
import csv
from collections import defaultdict

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

# Skip common/low-information values that create huge transactions in value-equality mode
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
    """
    Load extracted pairs from a project JSON file.

    Returns:
      project_name: str
      entries: list of dicts with:
        - concept: str
        - file_path: str
        - option: str
        - value: any (kept as-is; downstream turns it into str)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    project_name = data.get("project_name", os.path.basename(json_path))

    entries = []
    config_data = data.get("config_data", {})
    for file_entry in config_data.get("config_file_data", []):
        concept = file_entry.get("concept", "") or "UNKNOWN"
        file_path = file_entry.get("file_path", "") or ""
        for pair in file_entry.get("pairs", []):
            option = pair.get("option", "")
            value = pair.get("value", "")
            if option and value:
                entries.append(
                    {
                        "concept": concept,
                        "file_path": file_path,
                        "option": option,
                        "value": value,
                    }
                )

    return project_name, entries


def _as_str(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    # deterministic fallback for lists/dicts/numbers
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _item_concept_option(concept, option):
    # concept-aware to avoid cross-tech collisions
    return f"{concept}::{option}"


def build_transactions_value_equality(all_project_entries):
    """
    Build transactions where each transaction is a set of OPTIONS sharing the same VALUE in a project.

    A transaction is created for each (project, value) combination that has at least 2 options with that value.
    Options are concept-aware: "concept::option".
    """
    transactions = []

    for project_name, entries in all_project_entries:
        value_to_options = defaultdict(set)

        for e in entries:
            option = e["option"]
            value = _as_str(e["value"]).strip()
            if not value:
                continue
            if value.lower() in SKIP_VALUES:
                continue

            concept = e["concept"]
            value_to_options[value].add(_item_concept_option(concept, option))

        for value, options in value_to_options.items():
            if len(options) >= 2:
                transactions.append(
                    {
                        "project": project_name,
                        "value": value,
                        "options": frozenset(options),
                    }
                )

    return transactions


def build_transactions_cooccurrence(all_project_entries):
    """
    Build transactions where each transaction is the set of OPTIONS that co-occur within a project,
    ignoring values. Options are concept-aware: "concept::option".

    One transaction per project.
    """
    transactions = []
    for project_name, entries in all_project_entries:
        opts = set()
        for e in entries:
            # still require option+value existence (consistent with your current loader)
            concept = e["concept"]
            option = e["option"]
            opts.add(_item_concept_option(concept, option))

        if opts:
            # keep field names compatible with existing miner/saver
            transactions.append(
                {
                    "project": project_name,
                    "value": "",  # not used; present to keep output schema stable
                    "options": frozenset(opts),
                }
            )
    return transactions


def _parse_line_number(line_str):
    """
    Parse line number from string. Returns int or None if unknown/invalid.
    """
    if not line_str or line_str == "Unknown":
        return None
    try:
        return int(line_str)
    except (ValueError, TypeError):
        return None


def build_structural_sequences(all_project_entries, concept_filter=None):
    """
    Build ordered sequences of options based on line numbers within config files.

    For each (project, file_path, concept), extract options ordered by their line number.
    This captures structural co-occurrences: options that appear near each other in files.

    Args:
        all_project_entries: List of (project_name, entries) tuples
        concept_filter: If specified, only include entries matching this concept.
                       If None, include all concepts.

    Returns:
        List of sequence dicts with: project, file_path, concept, sequence
        where sequence is a list of option names ordered by line number.
    """
    sequences = []

    for project_name, entries in all_project_entries:
        # Group entries by (file_path, concept)
        by_file_concept = defaultdict(list)
        for e in entries:
            concept = e.get("concept") or "UNKNOWN"
            if concept_filter and concept.lower() != concept_filter.lower():
                continue

            file_path = e.get("file_path") or ""
            line_num = _parse_line_number(e.get("line"))
            option = e.get("option", "")

            if not option:
                continue

            by_file_concept[(file_path, concept)].append({
                "option": option,
                "line": line_num,
            })

        for (file_path, concept), file_entries in by_file_concept.items():
            if len(file_entries) < 2:
                continue

            # Separate entries with known vs unknown line numbers
            with_line = [(e["option"], e["line"]) for e in file_entries if e["line"] is not None]
            without_line = [e["option"] for e in file_entries if e["line"] is None]

            # Sort entries with line numbers
            with_line_sorted = sorted(with_line, key=lambda x: x[1])

            # Build sequence: sorted entries first, then unknowns at end
            seq = [opt for opt, _ in with_line_sorted] + without_line

            if len(seq) >= 2:
                sequences.append({
                    "project": project_name,
                    "file_path": file_path,
                    "concept": concept,
                    "sequence": seq,
                })

    return sequences


def mine_frequent_ngrams(sequences, ngram_min=2, ngram_max=5, min_support=0.1):
    """
    Mine frequent contiguous n-grams from sequences.
    Support is computed over DISTINCT PROJECTS (not sequences), to align with your RQs.
    """
    # map ngram(tuple) -> set(projects), set(concepts), count_occurrences
    projects_by_ngram = defaultdict(set)
    concepts_by_ngram = defaultdict(set)
    occ_by_ngram = defaultdict(int)

    all_projects = {s["project"] for s in sequences}
    num_projects = len(all_projects)
    if num_projects == 0:
        return {}

    for s in sequences:
        project = s["project"]
        concept = s.get("concept", "UNKNOWN")
        seq = s["sequence"]
        L = len(seq)
        if L < ngram_min:
            continue

        for n in range(ngram_min, min(ngram_max, L) + 1):
            for i in range(0, L - n + 1):
                ng = tuple(seq[i : i + n])
                projects_by_ngram[ng].add(project)
                concepts_by_ngram[ng].add(concept)
                occ_by_ngram[ng] += 1

    min_projects = max(1, int(min_support * num_projects))
    results = {}
    for ng, projs in projects_by_ngram.items():
        if len(projs) >= min_projects:
            results[ng] = {
                "ngram_size": len(ng),
                "support": len(projs) / num_projects,
                "num_projects": len(projs),
                "projects": sorted(projs),
                "concepts": sorted(concepts_by_ngram[ng]),
                "occurrences": occ_by_ngram[ng],
            }
    return results


def mine_frequent_itemsets(transactions, min_support, min_size=2, max_size=5):
    """Mine frequent itemsets using FP-Growth (via mlxtend)."""
    num_transactions = len(transactions)
    if num_transactions == 0:
        return {}

    # Count item frequencies and pre-filter to frequent items only
    # (anti-monotone property: infrequent items can't be in frequent itemsets)
    item_counts = defaultdict(int)
    for t in transactions:
        for item in t["options"]:
            item_counts[item] += 1
    min_count = max(1, int(min_support * num_transactions))
    frequent_items = {item for item, cnt in item_counts.items() if cnt >= min_count}

    # Build filtered transactions (only frequent items) for encoding
    transaction_lists = [
        [item for item in t["options"] if item in frequent_items]
        for t in transactions
    ]
    transaction_lists = [tl for tl in transaction_lists if tl]
    if not transaction_lists:
        return {}

    print(f"  {len(frequent_items)} frequent items (of {len(item_counts)} total)")

    te = TransactionEncoder()
    te_array = te.fit_transform(transaction_lists)
    df = pd.DataFrame(te_array, columns=te.columns_)
    df = df.astype(pd.SparseDtype("bool", fill_value=False))

    # FP-Growth mining
    freq_df = fpgrowth(df, min_support=min_support, max_len=max_size, use_colnames=True)
    if freq_df.empty:
        return {}

    freq_df = freq_df[freq_df["itemsets"].apply(len) >= min_size]
    if freq_df.empty:
        return {}

    # Build inverted index (item -> transaction indices) for fast enrichment
    item_to_tids = defaultdict(set)
    for tid, t in enumerate(transactions):
        for item in t["options"]:
            item_to_tids[item].add(tid)

    # Pre-extract project/value arrays for fast lookup by transaction index
    tx_projects = [t["project"] for t in transactions]
    tx_values = [t.get("value", "") for t in transactions]

    # Iterate without pandas overhead (iterrows is very slow)
    itemsets_list = freq_df["itemsets"].tolist()
    supports_list = freq_df["support"].tolist()

    results = {}
    for itemset_raw, support in zip(itemsets_list, supports_list):
        itemset = frozenset(itemset_raw)

        # Intersect tid-sets to find matching transactions
        items = iter(itemset)
        tids = item_to_tids[next(items)].copy()
        for item in items:
            tids &= item_to_tids[item]

        projects = {tx_projects[tid] for tid in tids}
        values = {tx_values[tid] for tid in tids if tx_values[tid]}

        results[itemset] = {
            "support": support,
            "count": len(tids),
            "num_projects": len(projects),
            "projects": sorted(projects),
            "values": sorted(values),
        }

    return results


def save_results_itemsets(results, output_path):
    rows = []
    for itemset, info in results.items():
        rows.append(
            {
                "itemset_size": len(itemset),
                "options": " | ".join(sorted(itemset)),
                "support": round(info["support"], 4),
                "count": info["count"],
                "num_projects": info["num_projects"],
                "projects": ", ".join(info["projects"]),
                "values": ", ".join(info.get("values", [])),
            }
        )

    rows.sort(key=lambda r: (-r["count"], r["itemset_size"]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "itemset_size",
                "options",
                "support",
                "count",
                "num_projects",
                "projects",
                "values",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} frequent itemsets to {output_path}")


def save_results_ngrams(results, output_path):
    rows = []
    for ng, info in results.items():
        rows.append(
            {
                "ngram_size": info["ngram_size"],
                "ngram": " -> ".join(ng),
                "support": round(info["support"], 4),
                "num_projects": info["num_projects"],
                "occurrences": info["occurrences"],
                "concepts": ", ".join(info.get("concepts", [])),
                "projects": ", ".join(info["projects"]),
            }
        )

    rows.sort(key=lambda r: (-r["num_projects"], -r["ngram_size"], -r["occurrences"]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ngram_size", "ngram", "support", "num_projects", "occurrences", "concepts", "projects"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} frequent n-grams to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mine (1) value-equality itemsets, (2) co-occurrence itemsets, and (3) structural n-grams (line-ordered options)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory name (e.g., netflix). Looks for data in ../data/{input}/latest_commit/*.json",
    )

    parser.add_argument(
        "--mode",
        choices=["value_equality", "cooccurrence", "structural"],
        default="value_equality",
        help="Mining mode: value_equality (options sharing same value), "
             "cooccurrence (options in same project), structural (options ordered by line number).",
    )

    # For itemset modes
    parser.add_argument("--min_support", type=float, default=0.01, help="Minimum support threshold")
    parser.add_argument("--min_size", type=int, default=1, help="Minimum itemset size")
    parser.add_argument("--max_size", type=int, default=5, help="Maximum itemset size")

    # For structural sequence n-grams
    parser.add_argument("--concept_filter", default=None, help="Filter by concept (e.g., github-action, gradle). If not set, includes all concepts.")
    parser.add_argument("--ngram_min", type=int, default=2, help="Minimum n-gram size (structural mode)")
    parser.add_argument("--ngram_max", type=int, default=5, help="Maximum n-gram size (structural mode)")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", args.input, "latest_commit")

    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    all_project_entries = []
    for json_file in json_files:
        project_name, entries = load_project_data(json_file)
        all_project_entries.append((project_name, entries))

    print(f"Loaded {len(all_project_entries)} projects")
    output_dir = os.path.join(script_dir, "..", "data", args.input, "frequent_itemsets")
    os.makedirs(output_dir, exist_ok=True)

    if args.mode in ("value_equality", "cooccurrence"):
        print(f"Building transactions ({args.mode})...")
        if args.mode == "value_equality":
            transactions = build_transactions_value_equality(all_project_entries)
            out_name = "frequent_itemsets_value_equality.csv"
        else:
            transactions = build_transactions_cooccurrence(all_project_entries)
            out_name = "frequent_itemsets_cooccurrence.csv"

        print(f"Built {len(transactions)} transactions")

        results = mine_frequent_itemsets(
            transactions,
            min_support=args.min_support,
            min_size=args.min_size,
            max_size=args.max_size,
        )

        output_path = os.path.join(output_dir, out_name)
        save_results_itemsets(results, output_path)

    else:  # structural mode
        filter_msg = f"concept={args.concept_filter}" if args.concept_filter else "all concepts"
        print(f"Building structural sequences ({filter_msg})...")
        sequences = build_structural_sequences(all_project_entries, concept_filter=args.concept_filter)
        print(f"Built {len(sequences)} sequences from {len(set(s['project'] for s in sequences))} projects")

        print("Mining frequent n-grams...")
        results = mine_frequent_ngrams(
            sequences,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            min_support=args.min_support,
        )

        suffix = f"_{args.concept_filter}" if args.concept_filter else ""
        output_path = os.path.join(output_dir, f"frequent_structural_ngrams{suffix}.csv")
        save_results_ngrams(results, output_path)


if __name__ == "__main__":
    main()
