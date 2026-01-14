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
from typing import List, Dict, Set, Tuple
import pandas as pd
from difflib import SequenceMatcher


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Bot patterns to identify automated contributors
BOT_PATTERNS = [
    r'bot',
    r'automated',
    r'dependabot',
    r'renovate',
    r'greenkeeper',
    r'snyk',
    r'github-actions',
    r'codecov',
    r'travis',
    r'circleci',
    r'jenkins',
    r'semantic-release',
    r'allcontributors',
    r'stalebot',
    r'imgbot',
    r'pyup',
    r'pre-commit-ci',
    r'noreply@github\.com',
    r'users\.noreply\.github\.com'
]

def normalize_name(name: str) -> str:
    """Normalize a name for comparison by removing accents, lowercasing, and removing extra whitespace."""
    # Remove accents and diacritics
    name = unicodedata.normalize('NFKD', name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    # Lowercase and strip whitespace
    name = name.lower().strip()
    # Remove multiple spaces
    name = re.sub(r'\s+', ' ', name)
    return name

def parse_git_identity(identity: str) -> Tuple[str, str]:
    """Parse git identity string 'Name <email>' into name and email components."""
    name, email = parseaddr(identity)
    if not name:
        # If parseaddr fails, try manual parsing
        match = re.match(r'^(.+?)\s*<(.+?)>$', identity)
        if match:
            name, email = match.groups()
        else:
            # Use the whole string as name if no email found
            name = identity
            email = ""
    return name.strip(), email.strip().lower()

def is_bot(identity: str) -> bool:
    """Check if a contributor identity appears to be a bot."""
    identity_lower = identity.lower()
    for pattern in BOT_PATTERNS:
        if re.search(pattern, identity_lower):
            return True
    return False

def name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names using sequence matching."""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    return SequenceMatcher(None, norm1, norm2).ratio()

def merge_identities(df: pd.DataFrame, similarity_threshold: float = 0.80) -> pd.DataFrame:
    """
    Merge contributor identities based on name and email matching.

    Uses the following heuristics:
    1. Exact email match -> same person
    2. Exact normalized name match -> likely same person
    3. High name similarity (>threshold) with same email domain -> likely same person
    4. Filter out bots

    Args:
        df: DataFrame with contributor data
        similarity_threshold: Threshold for name similarity (0-1)

    Returns:
        DataFrame with merged identities
    """
    logger.info("Starting identity resolution and bot filtering...")

    # Parse all identities
    identities = []
    for contributor in df['Contributor']:
        name, email = parse_git_identity(contributor)
        identities.append({
            'original': contributor,
            'name': name,
            'email': email,
            'normalized_name': normalize_name(name),
            'email_domain': email.split('@')[-1] if '@' in email else '',
            'is_bot': is_bot(contributor)
        })

    # Build canonical identity mapping
    canonical_map = {}  # maps original identity -> canonical identity
    identity_groups = defaultdict(list)  # maps canonical -> list of aliases

    # First pass: Group by exact email match
    email_to_canonical = {}
    for identity in identities:
        if identity['is_bot']:
            continue  # Skip bots

        email = identity['email']
        if email and email not in email_to_canonical:
            email_to_canonical[email] = identity['original']
            identity_groups[identity['original']].append(identity['original'])
            canonical_map[identity['original']] = identity['original']
        elif email:
            canonical = email_to_canonical[email]
            identity_groups[canonical].append(identity['original'])
            canonical_map[identity['original']] = canonical

    # Second pass: Group by normalized name match (for entries without email or different emails)
    name_to_canonical = {}
    for identity in identities:
        if identity['is_bot'] or identity['original'] in canonical_map:
            continue

        norm_name = identity['normalized_name']
        if not norm_name:
            continue

        # Check if this normalized name already has a canonical identity
        found_match = False
        for existing_norm, existing_canonical in name_to_canonical.items():
            similarity = name_similarity(norm_name, existing_norm)
            if similarity >= similarity_threshold:
                identity_groups[existing_canonical].append(identity['original'])
                canonical_map[identity['original']] = existing_canonical
                found_match = True
                break

        if not found_match:
            name_to_canonical[norm_name] = identity['original']
            identity_groups[identity['original']].append(identity['original'])
            canonical_map[identity['original']] = identity['original']

    # Log bot filtering results
    bot_count = sum(1 for i in identities if i['is_bot'])
    logger.info(f"Filtered out {bot_count} bot identities")

    # Log identity merging results
    num_merged = len(df) - len(identity_groups)
    logger.info(f"Merged {num_merged} duplicate identities into {len(identity_groups)} unique contributors")

    # Create merged dataframe
    merged_rows = []
    for canonical, aliases in identity_groups.items():
        # Get all rows for these aliases
        alias_rows = df[df['Contributor'].isin(aliases)]

        # Get aliases excluding the canonical identity
        other_aliases = [a for a in aliases if a != canonical]

        # Aggregate the data
        merged_row = {
            'Contributor': canonical,
            'Config Commits': alias_rows['Config Commits'].sum(),
            'Non-Config Commits': alias_rows['Non-Config Commits'].sum(),
            'Avg Config Files Per Commit': alias_rows['Avg Config Files Per Commit'].mean(),
            'Config Files': list(set(sum(alias_rows['Config Files'].tolist(), []))),
            'Aliases': sorted(list(set(other_aliases))) if other_aliases else []
        }
        merged_rows.append(merged_row)

    df_merged = pd.DataFrame(merged_rows)
    return df_merged

def get_project_file(project_dir: str):
    """Load project JSON files from the specified directory."""
    project_name = project_dir.split("/")[-1]
    json_files = glob.glob(f"../data/projects/{project_dir}/*.json")

    if any("batch" in f for f in json_files):
        json_files = [f for f in json_files if re.search(r"batch_\d+\.json$", f)]
        # Sort batch files in ascending order by batch number
        json_files = sorted(json_files, key=lambda x: int(re.search(r"batch_(\d+)\.json$", x).group(1)))
        return json_files

    if os.path.join(f"../data/projects/{project_dir}/{project_name}.json") in json_files:
        return os.path.join(f"../data/projects/{project_dir}/{project_name}.json")

    return None


def _get_contributors(project_file: str) -> pd.DataFrame:
    """Extract contributors from project data, aggregate by canonical identity, and flag core developers."""
    with open(project_file, 'r') as f:
        project_data = json.load(f)

        commit_data = project_data.get("commit_data", [])
        total_commits_project = int(project_data.get("len_commits", 0))

        contributors_stats = defaultdict(lambda: {
            "config_commits": 0,
            "non_config_commits": 0,
            "config_files": set(),
            "total_config_files_changed": 0
        })

        for commit in tqdm(commit_data, desc="Processing commits"):
            author = commit["author"]  # keep raw; we'll parse later
            is_config_related = commit["is_config_related"]

            if is_config_related:
                contributors_stats[author]["config_commits"] += 1

                # Extract config files from this commit
                network_data = commit.get("network_data", {})
                config_file_data = network_data.get("config_file_data", [])
                num_files_in_commit = 0
                for config_file in config_file_data:
                    file_path = config_file.get("file_path")
                    if file_path:
                        contributors_stats[author]["config_files"].add(file_path)
                        num_files_in_commit += 1

                contributors_stats[author]["total_config_files_changed"] += num_files_in_commit
            else:
                contributors_stats[author]["non_config_commits"] += 1

        commit_stats_rows = []
        for contributor, stats in contributors_stats.items():
            # Calculate average config files per config commit
            avg_files_per_commit = (
                stats["total_config_files_changed"] / stats["config_commits"]
                if stats["config_commits"] > 0 else 0
            )

            commit_stats_rows.append({
                "Contributor": contributor,  # still "Name <email>" or similar
                "Config Commits": stats["config_commits"],
                "Non-Config Commits": stats["non_config_commits"],
                "Avg Config Files Per Commit": round(avg_files_per_commit, 2),
                "Config Files": list(stats["config_files"])
            })

        df_contributors = pd.DataFrame(commit_stats_rows)

        return df_contributors


def get_contributors(project_dirs: List):
    """Process project files to extract and aggregate contributor statistics."""
    failed_projects = []
    
    for project_dir in project_dirs:
        try:
            project_files = get_project_file(project_dir)

            # Skip if contributors file already exists
            if os.path.exists(f"../data/projects_contributors/{project_dir}_contributors.csv"):
                continue

            logger.info(f"Processing project directory: {project_dir}")

            # Process single project file
            if isinstance(project_files, str):
                project_file = project_files
                df_contributors = _get_contributors(project_file)
                df_contributors.to_csv(f"../data/projects_contributors/{project_dir}_contributors.csv", index=False)

            # Process batch of project files
            if isinstance(project_files, list):
                logger.info(f"Processing {len(project_files)} batch files for {project_dir}")
                all_contributors = []

                for batch_file in project_files:
                    logger.info(f"Processing batch file: {batch_file}")
                    df_batch = _get_contributors(batch_file)
                    all_contributors.append(df_batch)

                # Concatenate all batch dataframes
                df_combined = pd.concat(all_contributors, ignore_index=True)

                # Aggregate by contributor
                df_aggregated = df_combined.groupby("Contributor").agg({
                    "Config Commits": "sum",
                    "Non-Config Commits": "sum",
                    "Avg Config Files Per Commit": "mean",  # Average of averages
                    "Config Files": lambda x: list(set(sum(x.tolist(), [])))  # Flatten and deduplicate
                }).reset_index()

                df_aggregated.to_csv(f"../data/projects_contributors/{project_dir}_contributors.csv", index=False)
        except Exception as e:
            logger.error(f"Failed to process project {project_dir}: {e}")
            failed_projects.append(project_dir)


def aggregate_contributors(contributors_dir: str = "../data/projects_contributors/"):
    """
    Aggregate contributor identities across all contributor CSV files.

    This function:
    1. Reads all contributor CSV files from the specified directory
    2. Applies identity resolution to merge duplicate contributors
    3. Filters out bot contributors
    4. Saves aggregated results with 'aggregated_' prefix

    Args:
        contributors_dir: Directory containing contributor CSV files
    """
    logger.info(f"Starting contributor aggregation in {contributors_dir}")

    # Find all contributor CSV files (excluding already aggregated ones)
    csv_files = [f for f in os.listdir(contributors_dir)]

    logger.info(f"Found {len(csv_files)} contributor files to process")

    for csv_file in csv_files:
        try:
            file_path = os.path.join(contributors_dir, csv_file)
            logger.info(f"Processing {csv_file}")

            # Read the contributor file
            df = pd.read_csv(file_path)

            # Parse the Config Files column (it's stored as string representation of list)
            df['Config Files'] = df['Config Files'].apply(eval)

            # Apply identity resolution and bot filtering
            df_aggregated = merge_identities(df)

            # Save aggregated results
            output_file = f"../data/projects_contributors_aggregated/aggregated_{csv_file}"
            df_aggregated.to_csv(output_file, index=False)
            logger.info(f"Saved aggregated results to {output_file}")

        except Exception as e:
            logger.error(f"Failed to process {csv_file}: {e}")
            continue

    logger.info("Contributor aggregation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", action="store_true",
                       help="Aggregate existing contributor files to merge identities and filter bots")
    args = parser.parse_args()

    if args.aggregate:
        # Run identity aggregation on existing contributor files
        aggregate_contributors()
    else:
        # Extract contributors from project files
        project_dirs = os.listdir("../../data/projects/")
        get_contributors(project_dirs)

