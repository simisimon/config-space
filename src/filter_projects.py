#!/usr/bin/env python3
"""
Filter projects based on activity and status.

Usage:
    # Filter with default settings (exclude archived, forks, commits before 2025-11-01)
    python filter_projects.py --input ../data/disney_projects_raw.csv

    # Custom minimum commit date
    python filter_projects.py --input ../data/disney_projects_raw.csv --min-date 2025-12-01

    # Include archived projects and forks
    python filter_projects.py --input ../data/disney_projects_raw.csv --include-archived --include-forks

    Output: {input_dir}/{prefix}_projects_final.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from datetime import datetime, timezone
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MIN_DATE = "2025-11-01"


def parse_date(date_str: str) -> datetime | None:
    """Parse ISO format date string."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def read_csv(path: str) -> List[Dict]:
    """Read CSV file into list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    """Write list of dicts to CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def filter_projects(
    rows: List[Dict],
    min_commit_date: datetime,
    exclude_archived: bool = True,
    exclude_forks: bool = True,
) -> List[Dict]:
    """Filter projects based on criteria."""
    filtered = []

    for row in rows:
        # Skip archived projects
        if exclude_archived and row.get("archived", "").lower() == "true":
            continue

        # Skip forks
        if exclude_forks and row.get("fork", "").lower() == "true":
            continue

        # Check latest commit date
        commit_date = parse_date(row.get("latest_commit_date", ""))
        if not commit_date or commit_date < min_commit_date:
            continue

        filtered.append(row)

    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter projects based on activity and status"
    )
    parser.add_argument("--input", required=True, help="Input CSV file (e.g., disney_projects_raw.csv)")
    parser.add_argument("--min-date", default=DEFAULT_MIN_DATE, help=f"Minimum commit date (default: {DEFAULT_MIN_DATE})")
    parser.add_argument("--include-archived", action="store_true", help="Include archived projects")
    parser.add_argument("--include-forks", action="store_true", help="Include forked projects")
    args = parser.parse_args()

    # Parse minimum date (make timezone-aware in UTC)
    min_commit_date = datetime.fromisoformat(args.min_date).replace(tzinfo=timezone.utc)
    logger.info(f"Filtering projects with commits after {args.min_date}")

    # Read input
    logger.info(f"Reading {args.input}")
    rows = read_csv(args.input)
    logger.info(f"Found {len(rows)} total projects")

    # Filter
    filtered = filter_projects(
        rows,
        min_commit_date=min_commit_date,
        exclude_archived=not args.include_archived,
        exclude_forks=not args.include_forks,
    )
    logger.info(f"Filtered to {len(filtered)} projects")

    # Derive output path: disney_projects_raw.csv -> disney_projects_final.csv
    input_basename = os.path.basename(args.input)
    output_basename = input_basename.replace("_raw.csv", "_final.csv")
    output_dir = os.path.dirname(args.input)
    output_path = os.path.join(output_dir, output_basename) if output_dir else output_basename

    # Write output
    if filtered:
        fieldnames = list(rows[0].keys())
        write_csv(output_path, filtered, fieldnames)
        logger.info(f"Wrote {len(filtered)} projects to {output_path}")
    else:
        logger.warning("No projects matched the filter criteria")

    # Summary
    logger.info(f"Summary: {len(rows)} -> {len(filtered)} projects ({len(rows) - len(filtered)} filtered out)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
