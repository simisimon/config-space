import argparse
import glob
import logging
import json
import re
import os
from tqdm import tqdm
import unicodedata
from collections import defaultdict
from email.utils import parseaddr
from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd

# Patterns to identify bot contributors
BOT_HINTS = [
    r"\bbot\b", r"\[bot\]$", r"noreply@", r"buildkite", r"jenkins", r"circleci",
    r"ci@", r"actions@github.com", r"dependabot", r"renovate"
]

BOT_REGEX = re.compile("|".join(BOT_HINTS), re.I)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_project_files(limit: int | None = None) -> list[str]:
    project_files = glob.glob("../data/projects/*.json")
    logger.info(f"Found {len(project_files)} project files")
    if limit is not None:
        project_files = project_files[:limit]
        logger.info(f"Using {len(project_files)} project files")
    return project_files


def is_bot(name: str, email: str) -> bool:
    text = f"{name} {email}".lower()
    return bool(BOT_REGEX.search(text))


def normalize_name(name: str) -> str:
    """
    Unicode-normalize, trim, and collapse internal whitespace.
    Keep original casing (experts often care).
    """
    name = unicodedata.normalize("NFKC", (name or "").strip().strip('"').strip("'"))
    name = re.sub(r"\s+", " ", name)
    return name



def normalize_email(email: str, collapse_gmail_dots: bool = True) -> str:
    """
    Simple, deterministic normalization:
    - lowercase local and domain
    - drop `+tag` suffix in local part
    - optionally collapse dots for gmail/googlemail
    """
    email = (email or "").strip().lower()
    if "@" not in email:
        return email

    local, domain = email.split("@", 1)

    # remove plus tag
    if "+" in local:
        local = local.split("+", 1)[0]

    # optionally collapse dots for gmail-only
    if collapse_gmail_dots and domain in {"gmail.com", "googlemail.com"}:
        local = local.replace(".", "")
        domain = "gmail.com"  # coalesce googlemail -> gmail

    return f"{local}@{domain}"


def cluster_contributors_by_email(df: pd.DataFrame):
    """
    Maps each 'Contributor' row (string like 'Name <mail>') to a canonical key
    (normalized email if present; otherwise a lowercased name/raw fallback).
    Also returns clusters (canonical_key -> list of original 'Contributor' values).
    """
    clusters = defaultdict(list)
    contributor_map = {}

    for _, row in df.iterrows():
        name_email = row["Contributor"]
        ident = extract_contributor_identity(name_email)
        key = ident["canonical_key"]
        clusters[key].append(name_email)
        contributor_map[name_email] = key

    return clusters, contributor_map


def extract_contributor_identity(author_field: str) -> dict:
    """
    Parse the typical 'Name <email>' shape.
    Returns a dict with normalized name and canonical email key.
    """
    disp_name, email = parseaddr(author_field or "")
    disp_name = normalize_name(disp_name)
    canonical_email = normalize_email(email)
    return {
        "display_name": disp_name,
        "email": email.lower() if email else "",
        "canonical_key": canonical_email if canonical_email else (disp_name or author_field or "").lower(),
        "raw": author_field,
    }


def aggregate_contributor_stats(
    df: pd.DataFrame,
    contributor_map: dict,
    project_total_commits: int,
    exclude_bots_from_core: bool = True
) -> pd.DataFrame:
    """
    Aggregates per canonical contributor key.
    Preserves aliases (the original 'Name <email>' strings) and name variants.
    Computes totals and the 'Core Developer' flag against the project total.
    """
    aggregated = defaultdict(lambda: {
        "Config": 0,
        "NonConfig": 0,
        "Aliases": set(),
        "Names": set(),
        "IsBot": False,
    })

    for _, row in df.iterrows():
        original = row["Contributor"]
        key = contributor_map.get(original, original)
        # parse once to capture normalized name and email for bot detection
        ident = extract_contributor_identity(original)

        aggregated[key]["Config"] += int(row["Config Commits"])
        aggregated[key]["NonConfig"] += int(row["Non-Config Commits"])
        aggregated[key]["Aliases"].add(original)
        if ident["display_name"]:
            aggregated[key]["Names"].add(ident["display_name"])
        if BOT_REGEX.search(f"{ident['display_name']} {ident['email']}"):
            aggregated[key]["IsBot"] = True

    rows = []
    thr = 0.05 * float(project_total_commits)

    for key, s in aggregated.items():
        total_cfg = s["Config"]
        total_non = s["NonConfig"]
        total = total_cfg + total_non
        is_bot = s["IsBot"]
        core = (total >= thr) and (not is_bot if exclude_bots_from_core else True)

        rows.append({
            "Normalized Contributor": key,
            "Total Config Commits": total_cfg,
            "Total Non-Config Commits": total_non,
            "Total Commits": total,
            "Core Developer": core,
            "Is Bot": is_bot,
            "Aliases": sorted(s["Aliases"]),
            "Names": sorted(s["Names"]),
        })

    return pd.DataFrame(rows)

def canonical_display_name(names: List[str]) -> str:
    names = [normalize_name(n) for n in names if str(n).strip()]
    if not names:
        return ""
    return sorted(names, key=lambda s: (len(s), s))[-1]

def merge_rows_by_display_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rows that share ANY normalized display name (case-insensitive),
    even if their canonical names differ (e.g., 'sqshq' vs 'Alexander Lukyanchikov').
    Bots stay isolated.
    """
    import re

    # Bot detection (if not already present)
    BOT_REGEX = re.compile(r"(?:\bbot\b|\bnoreply\b|actions@github\.com|dependabot|renovate)", re.I)
    if "Is Bot" not in df.columns:
        df["Is Bot"] = df.apply(
            lambda r: bool(BOT_REGEX.search(" ".join((r.get("Names") or []) + (r.get("Aliases") or [])))),
            axis=1
        )

    # Ensure numeric
    for c in ["Total Config Commits", "Total Non-Config Commits", "Total Commits"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0

    def norm_name(s: str) -> str:
        s = (s or "").strip().strip('"').strip("'")
        s = re.sub(r"\s+", " ", s)
        return s.lower()

    # Build name -> row indices index (exclude bots)
    name_index: dict[str, set[int]] = {}
    for idx, row in df[~df["Is Bot"]].iterrows():
        for n in set(norm_name(x) for x in (row.get("Names") or []) if str(x).strip()):
            if not n:
                continue
            name_index.setdefault(n, set()).add(idx)

    # Union-Find over row indices that share any name
    parent: dict[int, int] = {}
    def find(x: int) -> int:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, idxs in name_index.items():
        idxs = list(idxs)
        for i in range(1, len(idxs)):
            union(idxs[0], idxs[i])

    # Build components
    components: dict[int, list[int]] = {}
    for idx in df.index:
        if df.loc[idx, "Is Bot"]:
            components[idx] = [idx]  # keep bot isolated
        else:
            r = find(idx)
            components.setdefault(r, []).append(idx)

    # Merge within each component
    merged_rows = []
    for rep, idxs in components.items():
        sub = df.loc[idxs]

        # representative = highest Total Commits
        rep_row = sub.sort_values("Total Commits", ascending=False).iloc[0].to_dict()

        aliases, names = set(), set()
        tcfg = tnon = ttot = 0
        is_bot = bool(sub["Is Bot"].any())

        for _, r in sub.iterrows():
            aliases.update(r.get("Aliases", []))
            names.update(r.get("Names", []))
            tcfg += int(r.get("Total Config Commits", 0))
            tnon += int(r.get("Total Non-Config Commits", 0))
            ttot += int(r.get("Total Commits", 0))

        rep_row["Aliases"] = sorted(aliases)
        rep_row["Names"] = sorted(names)
        rep_row["Total Config Commits"] = int(tcfg)
        rep_row["Total Non-Config Commits"] = int(tnon)
        rep_row["Total Commits"] = int(ttot)
        rep_row["Is Bot"] = is_bot

        merged_rows.append(rep_row)

    out = pd.DataFrame(merged_rows)
    sort_cols = [c for c in ["Core Developer", "Total Commits", "Total Config Commits"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    return out


def get_contributors(project_data: dict) -> pd.DataFrame:
    """Extract contributors from project data, aggregate by canonical identity, and flag core developers."""
    commit_data = project_data.get("commit_data", [])
    total_commits_project = int(project_data.get("len_commits", 0))

    contributors_stats = defaultdict(lambda: {
        "config_commits": 0,
        "non_config_commits": 0,
        "files_changed": defaultdict(int)
    })

    for commit in tqdm(commit_data, desc="Processing commits"):
        author = commit["author"]  # keep raw; we’ll parse later
        is_config_related = commit["is_config_related"]

        if is_config_related:
            contributors_stats[author]["config_commits"] += 1
        else:
            contributors_stats[author]["non_config_commits"] += 1

    commit_stats_rows = []
    for contributor, stats in contributors_stats.items():
        commit_stats_rows.append({
            "Contributor": contributor,  # still "Name <email>" or similar
            "Config Commits": stats["config_commits"],
            "Non-Config Commits": stats["non_config_commits"]
        })

    df_contributors = pd.DataFrame(commit_stats_rows)

    # Cluster by normalized email and aggregate
    _, contributor_map = cluster_contributors_by_email(df_contributors)
    df_clustered = aggregate_contributor_stats(
        df_contributors,
        contributor_map,
        project_total_commits=total_commits_project,
        exclude_bots_from_core=True,
    )

    # Second-pass: merge by exact normalized display name across different emails
    df_merged = merge_rows_by_display_name(df_clustered)

    # Recompute core flag after name-based merges (totals may have changed)
    threshold = 0.05 * float(total_commits_project)
    df_merged["Core Developer"] = (df_merged["Total Commits"] >= threshold) & (~df_merged["Is Bot"])

    # Convenience: number of aliases
    df_merged["Number of Aliases"] = df_merged["Aliases"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    return df_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of projects to process")
    args = parser.parse_args()
    project_files = load_project_files(limit=args.limit)

    for project_file in tqdm(project_files, desc="Processing projects..."):
        project_name = project_file.split("/")[-1].replace(".json", "")
        logger.info(f"Processing test project: {project_name}")

        if os.path.exists(f"../data/project_contributors/{project_name}_contributors.csv"):
            logger.info(f"Contributors file for {project_name} already exists. Skipping.")
            continue

        with open(project_file, 'r') as f:
            project_data = json.load(f)
            df_contributors = get_contributors(project_data)
            df_contributors.to_csv(f"../data/project_contributors/{project_name}_contributors.csv", index=False)
