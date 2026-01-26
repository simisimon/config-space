#!/usr/bin/env python3
import argparse
import pandas as pd

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter projects based on activity and status"
    )
    parser.add_argument("--input", required=True, help="Input CSV file (e.g., disney_projects_raw.csv)")
    parser.add_argument("--output", required=True, help="Output file for commands")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    with open(args.output, "w", encoding="utf-8") as dest:
        for index, row in df.iterrows():
            dest.write(f"python3 /tmp/ssimon/config-space/experiments/analysis.py --url={row['html_url']} --name={row['name']} --commit={row['latest_commit_sha']}\n")


if __name__ == "__main__":
    raise SystemExit(main())
