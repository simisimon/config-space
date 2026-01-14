#!/usr/bin/env python3
"""
Configuration Knowledge Distribution Metrics Calculator

Computes inequality, concentration, and specialization metrics from contributor data.

Global metrics:
- Gini coefficient for configuration commits

Technology-centric metrics:
- ENC (Effective Number of Contributors)
- TCS (Top Contributor Share)
- Orphaned/Endangered technology flags
- KDP (Knowledge Diffusion Potential)

Contributor-centric metrics:
- TII (Technology Isolation Index)
"""

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import the technology mapping function
sys.path.insert(0, str(Path(__file__).parent))
from mapping import get_technology


def detect_column(df: pd.DataFrame, candidates: List[str],
                  contains_all: Optional[List[str]] = None,
                  contains_any: Optional[List[str]] = None) -> Optional[str]:
    """
    Auto-detect a column based on keyword matching.

    Args:
        df: DataFrame to search
        candidates: Exact column names to check first
        contains_all: All keywords must be present (case-insensitive)
        contains_any: At least one keyword must be present (case-insensitive)

    Returns:
        Column name if found, None otherwise
    """
    # Check exact matches first
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    # Check contains patterns
    for col in df.columns:
        col_lower = col.lower()

        if contains_all:
            if all(keyword.lower() in col_lower for keyword in contains_all):
                return col

        if contains_any:
            if any(keyword.lower() in col_lower for keyword in contains_any):
                return col

    return None


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect all required and optional columns.

    Returns:
        Dictionary mapping column types to detected column names
    """
    detected = {}

    # Contributor identifier (required)
    detected['contributor'] = detect_column(
        df,
        candidates=['Contributor', 'contributor', 'Author', 'author', 'Login', 'login'],
        contains_any=['name', 'author', 'contributor', 'login', 'email', 'id']
    )

    # Config commits (required)
    detected['config_commits'] = detect_column(
        df,
        candidates=['Config Commits', 'config_commits', 'configuration_commits'],
        contains_all=['config', 'commit']
    )

    # Non-config commits (optional)
    detected['non_config_commits'] = detect_column(
        df,
        candidates=['Non-Config Commits', 'non_config_commits', 'code_commits'],
        contains_any=['non', 'code']
    )

    # Config files/technologies (optional)
    detected['config_files'] = detect_column(
        df,
        candidates=['Config Files', 'config_files', 'files', 'technologies', 'artifacts'],
        contains_any=['config file', 'files', 'technology', 'tech', 'artifact', 'path']
    )

    return detected


def parse_file_list(value, delimiter_regex: str = r'[;,|]') -> List[Tuple[str, int]]:
    """
    Parse a config files column value into a list of (file_path, count) tuples.

    Handles:
    - Python list of tuples: "[('file1.yml', 2), ('file2.json', 3)]"
    - Python list strings: "['file1.yml', 'file2.json']" (count defaults to 1)
    - Delimited strings: "file1.yml;file2.json" (count defaults to 1)
    - Empty values: [] or ""

    Returns:
        List of (file_path, count) tuples
    """
    if pd.isna(value) or value == '' or value == '[]':
        return []

    # Try to parse as Python literal (list)
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    if not item:
                        continue
                    # Check if item is a tuple (file_path, count)
                    if isinstance(item, tuple) and len(item) == 2:
                        file_path, count = item
                        result.append((str(file_path), int(count)))
                    # Otherwise treat as just a file path with count=1
                    else:
                        result.append((str(item), 1))
                return result
        except (ValueError, SyntaxError):
            pass

    # Try delimiter-based splitting (count defaults to 1)
    if isinstance(value, str):
        files = re.split(delimiter_regex, value.strip())
        return [(f.strip().strip("'\""), 1) for f in files if f.strip()]

    return []


def extract_technologies(df: pd.DataFrame, config_files_col: str,
                        delimiter_regex: str) -> pd.DataFrame:
    """
    Extract technologies from config files and create per-contributor-technology data.

    Now uses actual file touch counts from the data to allocate commits to technologies.
    Files without recognized technology mapping are excluded from analysis.

    Returns:
        DataFrame with columns: contributor, technology, tech_commits
    """
    records = []

    for idx, row in df.iterrows():
        contributor = row.name  # Assume index is contributor ID
        config_commits = row['config_commits']

        if config_commits <= 0:
            continue

        file_tuples = parse_file_list(row[config_files_col], delimiter_regex)
        if not file_tuples:
            continue

        # Map files to technologies with their counts
        tech_counts = {}
        total_file_touches = 0

        for file_path, count in file_tuples:
            tech = get_technology(file_path)
            if tech:
                tech_counts[tech] = tech_counts.get(tech, 0) + count
                total_file_touches += count
            # Skip files without recognized technology mapping

        if not tech_counts or total_file_touches == 0:
            continue

        # Distribute config commits proportionally based on file touch counts
        for tech, touch_count in tech_counts.items():
            tech_commits = config_commits * (touch_count / total_file_touches)
            records.append({
                'contributor': contributor,
                'technology': tech,
                'tech_commits': tech_commits
            })

    return pd.DataFrame(records)


def extract_file_level_data(df: pd.DataFrame, config_files_col: str,
                            delimiter_regex: str) -> pd.DataFrame:
    """
    Extract file-level touch data for each contributor.

    Returns:
        DataFrame with columns: contributor, file_path, technology, touch_count
    """
    records = []

    for idx, row in df.iterrows():
        contributor = row.name  # Assume index is contributor ID
        config_commits = row['config_commits']

        if config_commits <= 0:
            continue

        file_tuples = parse_file_list(row[config_files_col], delimiter_regex)
        if not file_tuples:
            continue

        for file_path, count in file_tuples:
            tech = get_technology(file_path)
            # Include all files, even those without recognized technology
            records.append({
                'contributor': contributor,
                'file_path': file_path,
                'technology': tech if tech else 'unknown',
                'touch_count': count
            })

    return pd.DataFrame(records)


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute Gini coefficient for inequality measurement.

    Formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    where x is sorted in ascending order.

    Returns value in [0, 1] where 0 = perfect equality, 1 = perfect inequality.
    """
    if len(values) == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)
    total = sorted_values.sum()

    if total == 0:
        return 0.0

    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_values)) / (n * total) - (n + 1) / n

    return float(gini)


def compute_technology_metrics(tech_contrib_df: pd.DataFrame,
                               min_commits_k: int) -> pd.DataFrame:
    """
    Compute technology-centric metrics: ENC, TCS, orphaned, endangered flags.

    Args:
        tech_contrib_df: DataFrame with columns [contributor, technology, tech_commits]
        min_commits_k: Minimum commits threshold for orphaned/endangered classification

    Returns:
        DataFrame with technology metrics
    """
    metrics = []

    for tech, group in tech_contrib_df.groupby('technology'):
        total_commits = group['tech_commits'].sum()
        num_contributors = len(group)

        if total_commits == 0:
            continue

        # Compute share for each contributor in this technology
        shares = group['tech_commits'].values / total_commits

        # Sanity check: shares should sum to ~1
        assert abs(shares.sum() - 1.0) < 1e-6, f"Shares for {tech} sum to {shares.sum()}"

        # ENC (Effective Number of Contributors) = 1 / HHI
        # HHI = sum of squared shares
        hhi = np.sum(shares ** 2)
        enc = 1.0 / hhi if hhi > 0 else 0.0

        # Sanity check: ENC should be between 1 and num_contributors
        assert 1.0 <= enc <= num_contributors + 0.01, \
            f"ENC for {tech} is {enc}, expected [1, {num_contributors}]"

        # TCS (Top Contributor Share)
        tcs = float(shares.max())

        # Sanity check: TCS should be in [0, 1]
        assert 0.0 <= tcs <= 1.0, f"TCS for {tech} is {tcs}"

        # Orphaned: exactly 1 active contributor and meets commit threshold
        orphaned = (num_contributors == 1) and (total_commits >= min_commits_k)

        # Endangered: high concentration (TCS >= 0.80) and low diversity (ENC <= 1.5)
        endangered = (tcs >= 0.80) and (enc <= 1.5) and (total_commits >= min_commits_k)

        metrics.append({
            'technology': tech,
            'total_config_commits': round(total_commits, 2),
            'num_active_contributors': num_contributors,
            'enc': round(enc, 4),
            'tcs': round(tcs, 4),
            'orphaned': orphaned,
            'endangered': endangered
        })

    return pd.DataFrame(metrics)


def compute_tii(contributor_tech_commits: pd.Series) -> float:
    """
    Compute Technology Isolation Index (TII) using normalized entropy.

    TII measures how specialized a contributor is across technologies.
    - TII = 1.0: contributor works on exactly one technology (maximum isolation)
    - TII = 0.0: contributor distributes work evenly across many technologies

    Formula:
        H = -sum(p_t * log(p_t))  [Shannon entropy]
        TII = 1 - H / log(|T|)     [normalized entropy]

    Args:
        contributor_tech_commits: Series of tech_commits per technology for one contributor

    Returns:
        TII value in [0, 1]
    """
    total = contributor_tech_commits.sum()

    if total == 0 or len(contributor_tech_commits) == 0:
        return 0.0

    if len(contributor_tech_commits) == 1:
        return 1.0

    # Compute shares
    shares = contributor_tech_commits.values / total

    # Compute Shannon entropy (handle zero shares)
    nonzero_shares = shares[shares > 0]
    entropy = -np.sum(nonzero_shares * np.log(nonzero_shares))

    # Normalize by maximum possible entropy
    max_entropy = np.log(len(nonzero_shares))

    tii = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

    # Clamp to [0, 1] to handle floating point precision issues
    tii = max(0.0, min(1.0, tii))

    # Sanity check
    assert 0.0 <= tii <= 1.0, f"TII is {tii}, expected [0, 1]"

    return float(tii)


def compute_contributor_metrics(tech_contrib_df: pd.DataFrame,
                                contributor_col: str) -> pd.DataFrame:
    """
    Compute contributor-centric metrics: TII, number of technologies.

    Args:
        tech_contrib_df: DataFrame with columns [contributor, technology, tech_commits]
        contributor_col: Name of the contributor identifier column

    Returns:
        DataFrame with contributor metrics
    """
    metrics = []

    for contributor, group in tech_contrib_df.groupby('contributor'):
        total_commits = group['tech_commits'].sum()
        num_technologies = len(group)

        # Get list of technologies, sorted alphabetically
        technologies = sorted(group['technology'].unique().tolist())

        tii = compute_tii(group.set_index('technology')['tech_commits'])

        metrics.append({
            contributor_col: contributor,
            'config_commits': round(total_commits, 2),
            'num_technologies': num_technologies,
            'technologies': technologies,
            'tii': round(tii, 4)
        })

    return pd.DataFrame(metrics)


def compute_file_level_metrics(file_level_df: pd.DataFrame,
                                contributor_col: str) -> pd.DataFrame:
    """
    Compute file-level expertise/depth metrics for each contributor.

    Metrics:
    - total_config_files: Number of unique files touched
    - total_file_touches: Sum of all touch counts
    - avg_touches_per_file: Average engagement depth
    - max_file_touches: Highest touch count on any single file
    - touch_concentration: Gini coefficient on file touches (inequality measure)

    Args:
        file_level_df: DataFrame with columns [contributor, file_path, technology, touch_count]
        contributor_col: Name of the contributor identifier column

    Returns:
        DataFrame with file-level metrics per contributor
    """
    metrics = []

    for contributor, group in file_level_df.groupby('contributor'):
        total_files = len(group)
        total_touches = group['touch_count'].sum()
        avg_touches = total_touches / total_files if total_files > 0 else 0.0
        max_touches = group['touch_count'].max()

        # Compute touch concentration (Gini on file touches)
        touch_counts = group['touch_count'].values
        touch_gini = compute_gini_coefficient(touch_counts)

        metrics.append({
            contributor_col: contributor,
            'total_config_files': total_files,
            'total_file_touches': int(total_touches),
            'avg_touches_per_file': round(avg_touches, 2),
            'max_file_touches': int(max_touches),
            'touch_concentration': round(touch_gini, 4)
        })

    return pd.DataFrame(metrics)


def compute_kdp(tech_contrib_df: pd.DataFrame,
               contributor_tii: pd.DataFrame) -> pd.Series:
    """
    Compute Knowledge Diffusion Potential (KDP) per technology.

    KDP measures whether non-top contributors are generalists who could absorb knowledge.

    Formula:
        - Identify top contributor (largest share) for each technology
        - KDP_t = mean(1 - TII_i) for all i != top(t) with p_{i,t} > 0
        - If technology has <= 1 active contributor, KDP_t = 0

    Args:
        tech_contrib_df: DataFrame with [contributor, technology, tech_commits]
        contributor_tii: DataFrame with [contributor, tii]

    Returns:
        Series mapping technology -> KDP value
    """
    kdp_values = {}
    tii_map = contributor_tii.set_index(contributor_tii.columns[0])['tii'].to_dict()

    for tech, group in tech_contrib_df.groupby('technology'):
        if len(group) <= 1:
            kdp_values[tech] = 0.0
            continue

        # Find top contributor
        top_contributor = group.loc[group['tech_commits'].idxmax(), 'contributor']

        # Get non-top contributors
        non_top = group[group['contributor'] != top_contributor]

        if len(non_top) == 0:
            kdp_values[tech] = 0.0
            continue

        # Compute mean (1 - TII) for non-top contributors
        tii_values = [tii_map.get(c, 0.0) for c in non_top['contributor']]
        kdp = np.mean([1.0 - tii for tii in tii_values])

        kdp_values[tech] = round(float(kdp), 4)

    return pd.Series(kdp_values)


def process_single_file(input_file: Path, out_dir: Path, min_commits_k: int,
                       delimiter_regex: str, verbose: bool = True) -> bool:
    """
    Process a single contributor CSV file and generate metrics.

    Args:
        input_file: Path to input CSV
        out_dir: Output directory for results
        min_commits_k: Minimum commits threshold
        delimiter_regex: Regex for splitting list columns
        verbose: Whether to print detailed output

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read CSV
        if verbose:
            print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)

        # Auto-detect columns
        if verbose:
            print("Auto-detecting columns...")
        detected = auto_detect_columns(df)

        # Validate required columns
        if not detected['contributor']:
            if verbose:
                print("Error: Could not detect contributor identifier column", file=sys.stderr)
                print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            return False

        if not detected['config_commits']:
            if verbose:
                print("Error: Could not detect config commits column", file=sys.stderr)
                print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            return False

        if verbose:
            print(f"Detected columns:")
            for col_type, col_name in detected.items():
                status = col_name if col_name else "NOT FOUND"
                print(f"  {col_type}: {status}")

        # Prepare data
        df_clean = df.copy()
        df_clean = df_clean.rename(columns={
            detected['contributor']: 'contributor_id',
            detected['config_commits']: 'config_commits'
        })

        if detected['non_config_commits']:
            df_clean = df_clean.rename(columns={
                detected['non_config_commits']: 'non_config_commits'
            })

        # Ensure numeric columns
        df_clean['config_commits'] = pd.to_numeric(df_clean['config_commits'], errors='coerce').fillna(0)
        if 'non_config_commits' in df_clean.columns:
            df_clean['non_config_commits'] = pd.to_numeric(df_clean['non_config_commits'], errors='coerce').fillna(0)

        # Set index for contributor tracking
        df_clean['_idx'] = df_clean['contributor_id']
        df_clean = df_clean.set_index('_idx')

        # ========== A) Global Metrics ==========
        if verbose:
            print("\nComputing global inequality metrics...")

        config_commits = df_clean['config_commits'].values

        # Gini on all contributors (including zeros)
        gini_all = compute_gini_coefficient(config_commits)

        # Gini on active contributors only (config_commits > 0)
        active_commits = config_commits[config_commits > 0]
        gini_active = compute_gini_coefficient(active_commits)

        if verbose:
            print(f"  Global Gini (all contributors): {gini_all:.4f}")
            print(f"  Global Gini (active only): {gini_active:.4f}")

        # ========== B) Technology-Centric Metrics ==========
        tech_metrics_df = None
        contributor_metrics_df = None
        num_technologies = 0

        if detected['config_files']:
            if verbose:
                print("\nExtracting technologies from config files...")
            tech_contrib_df = extract_technologies(
                df_clean,
                detected['config_files'],
                delimiter_regex
            )

            # Extract file-level data for expertise metrics
            if verbose:
                print("Extracting file-level touch data...")
            file_level_df = extract_file_level_data(
                df_clean,
                detected['config_files'],
                delimiter_regex
            )

            if len(tech_contrib_df) > 0:
                num_technologies = tech_contrib_df['technology'].nunique()
                if verbose:
                    print(f"  Found {num_technologies} technologies across {len(tech_contrib_df)} contributor-technology pairs")

                if verbose:
                    print("\nComputing technology-centric metrics...")
                tech_metrics_df = compute_technology_metrics(tech_contrib_df, min_commits_k)

                if verbose:
                    print("\nComputing contributor-centric metrics...")
                contributor_metrics_df = compute_contributor_metrics(tech_contrib_df, 'contributor_id')

                # Compute file-level expertise metrics
                if verbose:
                    print("Computing file-level expertise metrics...")
                file_level_metrics_df = compute_file_level_metrics(file_level_df, 'contributor_id')

                # Add KDP to technology metrics
                if verbose:
                    print("\nComputing Knowledge Diffusion Potential (KDP)...")
                kdp_series = compute_kdp(tech_contrib_df, contributor_metrics_df)
                tech_metrics_df['kdp'] = tech_metrics_df['technology'].map(kdp_series).fillna(0.0).round(4)

                # Merge contributor metrics with original data
                contributor_metrics_full = df_clean[['contributor_id', 'config_commits']].reset_index(drop=True)
                if 'non_config_commits' in df_clean.columns:
                    contributor_metrics_full['non_config_commits'] = df_clean['non_config_commits'].values

                # Merge TII, num_technologies, and technologies
                contributor_metrics_full = contributor_metrics_full.merge(
                    contributor_metrics_df[['contributor_id', 'tii', 'num_technologies', 'technologies']],
                    on='contributor_id',
                    how='left'
                )

                # Merge file-level expertise metrics
                contributor_metrics_full = contributor_metrics_full.merge(
                    file_level_metrics_df[['contributor_id', 'total_config_files', 'total_file_touches',
                                           'avg_touches_per_file', 'max_file_touches', 'touch_concentration']],
                    on='contributor_id',
                    how='left'
                )

                # Fill NaN for contributors with no config files data
                contributor_metrics_full['tii'] = contributor_metrics_full['tii'].fillna(0.0).round(4)
                contributor_metrics_full['num_technologies'] = contributor_metrics_full['num_technologies'].fillna(0).astype(int)
                contributor_metrics_full['technologies'] = contributor_metrics_full['technologies'].apply(
                    lambda x: x if isinstance(x, list) else []
                )
                # Fill file-level metrics with 0 for inactive contributors
                contributor_metrics_full['total_config_files'] = contributor_metrics_full['total_config_files'].fillna(0).astype(int)
                contributor_metrics_full['total_file_touches'] = contributor_metrics_full['total_file_touches'].fillna(0).astype(int)
                contributor_metrics_full['avg_touches_per_file'] = contributor_metrics_full['avg_touches_per_file'].fillna(0.0).round(2)
                contributor_metrics_full['max_file_touches'] = contributor_metrics_full['max_file_touches'].fillna(0).astype(int)
                contributor_metrics_full['touch_concentration'] = contributor_metrics_full['touch_concentration'].fillna(0.0).round(4)
                contributor_metrics_full['config_commits'] = contributor_metrics_full['config_commits'].round(2)
                if 'non_config_commits' in contributor_metrics_full.columns:
                    contributor_metrics_full['non_config_commits'] = contributor_metrics_full['non_config_commits'].round(2)

                contributor_metrics_df = contributor_metrics_full
            else:
                if verbose:
                    print("  Warning: No technologies extracted from config files")
        else:
            if verbose:
                print("\nNo config files column detected - skipping technology-specific metrics")

        # ========== Summary Output ==========
        if verbose:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Global Gini Coefficient (all): {gini_all:.4f}")
            print(f"Global Gini Coefficient (active only): {gini_active:.4f}")
            print(f"Number of technologies: {num_technologies}")

            if tech_metrics_df is not None:
                num_orphaned = tech_metrics_df['orphaned'].sum()
                num_endangered = tech_metrics_df['endangered'].sum()

                print(f"Orphaned technologies: {num_orphaned}")
                print(f"Endangered technologies: {num_endangered}")

                print("\nTop 10 most concentrated technologies (lowest ENC):")
                top_concentrated = tech_metrics_df.nsmallest(10, 'enc')[
                    ['technology', 'enc', 'tcs', 'num_active_contributors', 'total_config_commits']
                ]
                print(top_concentrated.to_string(index=False))

        # ========== Write Outputs ==========
        out_dir.mkdir(parents=True, exist_ok=True)

        # Extract project name from input file
        # Expected format: projectname_contributors_merged.csv
        input_filename = input_file.stem  # filename without extension
        if '_contributors_merged' in input_filename:
            project_name = input_filename.replace('_contributors_merged', '')
        else:
            # Fallback: use full filename without extension
            project_name = input_filename

        if verbose:
            print(f"\nWriting outputs to {out_dir}...")
            print(f"Project name: {project_name}")

        # Write contributor metrics
        if contributor_metrics_df is not None:
            contrib_out = out_dir / f'{project_name}_contributors_metrics.csv'
            contributor_metrics_df.to_csv(contrib_out, index=False)
            if verbose:
                print(f"  Wrote {contrib_out.name}")
        else:
            # Write basic contributor metrics even without technology data
            basic_metrics = df_clean[['contributor_id', 'config_commits']].reset_index(drop=True)
            basic_metrics['config_commits'] = basic_metrics['config_commits'].round(2)
            if 'non_config_commits' in df_clean.columns:
                basic_metrics['non_config_commits'] = df_clean['non_config_commits'].round(2)
            basic_metrics['tii'] = 0.0
            basic_metrics['num_technologies'] = 0
            basic_metrics['technologies'] = [[] for _ in range(len(basic_metrics))]
            # Add placeholder file-level metrics
            basic_metrics['total_config_files'] = 0
            basic_metrics['total_file_touches'] = 0
            basic_metrics['avg_touches_per_file'] = 0.0
            basic_metrics['max_file_touches'] = 0
            basic_metrics['touch_concentration'] = 0.0

            contrib_out = out_dir / f'{project_name}_contributors_metrics.csv'
            basic_metrics.to_csv(contrib_out, index=False)
            if verbose:
                print(f"  Wrote {contrib_out.name} (basic metrics only)")

        # Write technology metrics
        if tech_metrics_df is not None:
            tech_out = out_dir / f'{project_name}_technologies_metrics.csv'
            tech_metrics_df.to_csv(tech_out, index=False)
            if verbose:
                print(f"  Wrote {tech_out.name}")

        # Write metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'project_name': project_name,
            'input_file': str(input_file),
            'detected_columns': detected,
            'assumptions': [
                "Config commits are distributed across technologies proportionally based on file touch counts",
                "Technologies are derived from file paths using mapping.py::get_technology()",
                "Files without recognized technology mapping are excluded from analysis",
                "TII = 0 for contributors with no config commits or no technologies",
                "TII = 1 for contributors working on exactly one technology",
                "KDP = 0 for technologies with <= 1 active contributor",
                "ENC = 0 skipped for technologies with zero total commits"
            ],
            'thresholds': {
                'min_commits_k': min_commits_k
            },
            'metrics': {
                'gini_all': round(float(gini_all), 4),
                'gini_active': round(float(gini_active), 4),
                'num_technologies': int(num_technologies),
                'num_contributors': int(len(df_clean)),
                'num_active_contributors': int((df_clean['config_commits'] > 0).sum())
            }
        }

        if tech_metrics_df is not None:
            orphaned_techs = tech_metrics_df[tech_metrics_df['orphaned'] == True]['technology'].tolist()
            endangered_techs = tech_metrics_df[tech_metrics_df['endangered'] == True]['technology'].tolist()

            metadata['metrics']['num_orphaned'] = len(orphaned_techs)
            metadata['metrics']['orphaned_technologies'] = orphaned_techs
            metadata['metrics']['num_endangered'] = len(endangered_techs)
            metadata['metrics']['endangered_technologies'] = endangered_techs

        metadata_out = out_dir / f'{project_name}_metadata.json'
        with open(metadata_out, 'w') as f:
            json.dump(metadata, f, indent=2)
        if verbose:
            print(f"  Wrote {metadata_out.name}")

        if verbose:
            print("\nDone!")

        return True

    except Exception as e:
        if verbose:
            print(f"Error processing {input_file}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compute configuration knowledge distribution metrics from contributor data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        default='../data/projects_contributors_merged/',
        help='Path to input CSV file or directory (default: ../data/projects_contributors_merged/)'
    )
    parser.add_argument(
        '--min_commits_k',
        type=int,
        default=5,
        help='Minimum commits threshold for orphaned/endangered classification (default: 5)'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='../data/projects_contributors_metrics',
        help='Output directory for metrics CSVs and metadata (default: ../data/project_contributor_metrics)'
    )
    parser.add_argument(
        '--delimiter_regex',
        type=str,
        default=r'[;,|]',
        help='Regex pattern for splitting list columns (default: [;,|])'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all *_contributors_merged.csv files in the input directory'
    )

    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)

    # Handle --all flag
    if args.all:
        input_path = Path(args.input)

        # Must be a directory
        if not input_path.is_dir():
            print(f"Error: --all requires input to be a directory, got: {input_path}", file=sys.stderr)
            sys.exit(1)

        # Find all *_contributors_merged.csv files
        csv_files = sorted(input_path.glob('*_contributors_merged.csv'))

        if not csv_files:
            print(f"Error: No *_contributors_merged.csv files found in {input_path}", file=sys.stderr)
            sys.exit(1)

        total_files = len(csv_files)
        print("=" * 70)
        print(f"Batch Processing: {total_files} projects")
        print("=" * 70)
        print(f"Input directory: {input_path}")
        print(f"Output directory: {out_dir}")
        print(f"Min commits threshold: {args.min_commits_k}")
        print("=" * 70)
        print()

        success_count = 0
        failed_count = 0
        failed_projects = []

        for idx, csv_file in enumerate(csv_files, 1):
            project_name = csv_file.stem.replace('_contributors_merged', '')
            print(f"[{idx}/{total_files}] Processing: {project_name}")

            success = process_single_file(
                csv_file,
                out_dir,
                args.min_commits_k,
                args.delimiter_regex,
                verbose=False  # Quiet mode for batch processing
            )

            if success:
                success_count += 1
                print(f"  ✓ Success")
            else:
                failed_count += 1
                failed_projects.append(project_name)
                print(f"  ✗ Failed")
            print()

        # Final summary
        print("=" * 70)
        print("Batch Processing Complete")
        print("=" * 70)
        print(f"Total projects: {total_files}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failed_count}")

        if failed_projects:
            print(f"\nFailed projects:")
            for proj in failed_projects:
                print(f"  - {proj}")

        print(f"\nOutput directory: {out_dir}")
        print("=" * 70)

        sys.exit(0 if failed_count == 0 else 1)

    # Single file mode
    input_path = Path(args.input)

    if input_path.is_dir():
        # Find first CSV in directory
        csv_files = list(input_path.glob('*.csv'))
        if not csv_files:
            print(f"Error: No CSV files found in {input_path}", file=sys.stderr)
            sys.exit(1)
        input_file = csv_files[0]
        print(f"Using first CSV file: {input_file.name}")
    else:
        input_file = input_path

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Process the single file
    success = process_single_file(
        input_file,
        out_dir,
        args.min_commits_k,
        args.delimiter_regex,
        verbose=True
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
