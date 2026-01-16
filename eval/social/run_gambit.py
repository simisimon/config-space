
import pandas as pd
import argparse
import sys
import csv
csv.field_size_limit(100000000)
from collections import defaultdict
import gambit
import os
import ast


def parse_author_string(author_str):
    """Parse author string to extract name and email.

    Expected format: 'Name <email@domain.com>' or just 'Name'
    """
    if '<' in author_str and '>' in author_str:
        name = author_str[:author_str.index('<')].strip()
        email = author_str[author_str.index('<')+1:author_str.index('>')].strip()
    else:
        name = author_str.strip()
        email = ''
    return name, email


def is_bot(author_str):
    """Check if an author is a bot based on common patterns."""
    name, email = parse_author_string(author_str)
    name_lower = name.lower()
    email_lower = email.lower()

    # Common bot name patterns
    bot_name_patterns = [
        '[bot]',           # GitHub App bots like dependabot[bot]
        'dependabot',
        'renovate',
        'greenkeeper',
        'snyk-bot',
        'codecov',
        'github-actions',
        'semantic-release',
        'mergify',
        'allcontributors',
    ]

    for pattern in bot_name_patterns:
        if pattern in name_lower:
            return True

    # Bot-specific email patterns
    bot_email_patterns = [
        'bot@renovateapp.com',
        'bot@dependabot.com',
        'noreply@github.com',  # Used by GitHub Actions, not human noreply
    ]

    for pattern in bot_email_patterns:
        if pattern in email_lower:
            return True

    return False


def load_authors_from_file(filename):
    """Load authors from CSV file.

    Expects CSV with one author string per line.
    """
    authors_list = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty lines
                author_str = row[0].strip()
                if author_str:
                    authors_list.append({
                        'author': author_str
                    })
    return authors_list


def create_author_dataframe_gambit(authors_list):
    """Create a DataFrame in the format expected by gambit."""
    data = []
    for author_info in authors_list:
        author_str = author_info['author']
        name, email = parse_author_string(author_str)

        data.append({
            'alias_name': name,
            'alias_email': email
        })

    df = pd.DataFrame(data)
    return df


def run_gambit(authors_list):
    """Run gambit algorithm on a list of authors.

    Returns list of tuples: (original_author_string, unique_dealiased_id)
    """
    # Create DataFrame in expected format
    author_df = create_author_dataframe_gambit(authors_list)

    try:
        # Run the gambit disambiguation algorithm
        merging_df = gambit.disambiguate_aliases(author_df, method='gambit')

        # Create pairs of (before dealiasing, after dealiasing)
        results = []
        for idx, author_info in enumerate(authors_list):
            original_author = author_info['author']
            dealiased_id = merging_df.iloc[idx]['author_id']
            results.append((original_author, dealiased_id))

        return results, None
    except Exception as e:
        return None, str(e)


def merge_contributors(input_file, gambit_results):
    """Merge contributors based on gambit dealiasing results.

    Args:
        input_file: Original contributors CSV with all information
        gambit_results: List of tuples (original_author, dealiased_id)

    Returns:
        DataFrame with merged contributor information
    """
    # Read the original contributors file
    df_original = pd.read_csv(input_file)

    # Create DataFrame from gambit results
    df_gambit = pd.DataFrame(gambit_results, columns=['Original Author', 'Dealiased ID'])

    # Merge the dataframes on contributor/original author
    df_merged = pd.merge(
        df_original,
        df_gambit,
        left_on='Contributor',
        right_on='Original Author',
        how='inner'
    )

    # Group by Dealiased ID and aggregate
    def aggregate_config_files(series):
        """Combine all config files from multiple contributors and count occurrences.

        Returns a list of tuples: [(file_path, touch_count), ...]
        """
        file_counts = defaultdict(int)
        for item in series:
            if pd.notna(item) and item:
                try:
                    # Parse the string representation of list
                    files = ast.literal_eval(item) if isinstance(item, str) else item
                    if isinstance(files, list):
                        for file in files:
                            file_counts[file] += 1
                except:
                    pass
        # Return list of tuples (file, count)
        return str(list(file_counts.items()))

    def get_representative_name(series):
        """Get the most complete contributor name from the group."""
        # Prefer names with email addresses
        names_with_email = [name for name in series if '<' in name and '>' in name]
        if names_with_email:
            return names_with_email[0]
        return series.iloc[0]

    def get_all_aliases(series):
        """Get all unique aliases for a contributor."""
        # Return all unique names as a list
        return str(list(series.unique()))

    # Aggregate the data
    df_aggregated = df_merged.groupby('Dealiased ID').agg({
        'Contributor': get_representative_name,  # Pick representative name
        'Original Author': get_all_aliases,  # Collect all aliases
        'Config Commits': 'sum',
        'Non-Config Commits': 'sum',
        'Config Files': aggregate_config_files
    }).reset_index()

    # Rename the 'Original Author' column to 'Aliases'
    df_aggregated.rename(columns={'Original Author': 'Aliases'}, inplace=True)

    # Calculate average config files per commit
    # Config Files is now a list of tuples [(file, count), ...]
    # Count unique files, not total touches
    total_commits = df_aggregated['Config Commits'] + df_aggregated['Non-Config Commits']
    df_aggregated['Avg Config Files Per Commit'] = df_aggregated['Config Files'].apply(
        lambda x: len(ast.literal_eval(x)) if x and x != '[]' else 0
    ) / total_commits.replace(0, 1)  # Avoid division by zero

    # Reorder columns to include aliases
    df_aggregated = df_aggregated[[
        'Dealiased ID',
        'Contributor',
        'Aliases',
        'Config Commits',
        'Non-Config Commits',
        'Avg Config Files Per Commit',
        'Config Files'
    ]]

    return df_aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Run gambit dealiasing and merge contributors'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all files in the directory'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Process a specific file (e.g., "OpenAPI-Specification_contributors.csv")'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (only applies with --all)'
    )
    args = parser.parse_args()

    # Determine the base directory (support running from root or eval/)
    if os.path.exists("../data/projects_contributors"):
        base_dir = "../data"
    elif os.path.exists("../../data/projects_contributors"):
        base_dir = "../../data"
    else:
        print("Error: Could not find data/projects_contributors directory", file=sys.stderr)
        sys.exit(1)

    input_dir = f"{base_dir}/projects_contributors"
    gambit_dir = f"{base_dir}/projects_contributors_gambit"
    merged_dir = f"{base_dir}/projects_contributors_merged"

    # Determine which files to process
    if args.file:
        files = [args.file]
    elif args.all:
        files = [f for f in os.listdir(input_dir) if f.endswith('_contributors.csv')]
        if args.limit is not None and args.limit > 0:
            files = files[:args.limit]
    else:
        # Default: process just the first file
        files = [f for f in os.listdir(input_dir) if f.endswith('_contributors.csv')][:1]

    # Create output directories if they don't exist
    os.makedirs(gambit_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    print(f"\nProcessing {len(files)} file(s)...\n", file=sys.stderr)

    for idx, file in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Processing {file}...", file=sys.stderr)

        input_file = f"{input_dir}/{file}"
        gambit_output = f"{gambit_dir}/{file.split('.csv')[0]}_gambit.csv"
        merged_output = f"{merged_dir}/{file.split('.csv')[0]}_merged.csv"

        # Load authors from input file
        try:
            authors_list = load_authors_from_file(input_file)
            print(f"  Loaded {len(authors_list)} authors", file=sys.stderr)
        except Exception as e:
            print(f"Error loading file: {e}", file=sys.stderr)
            sys.exit(1)

        # Filter bots
        original_count = len(authors_list)
        authors_list = [a for a in authors_list if not is_bot(a['author'])]
        filtered_count = original_count - len(authors_list)
        if filtered_count > 0:
            print(f"  Filtered {filtered_count} bots, {len(authors_list)} authors remaining", file=sys.stderr)

        # Run gambit
        results, error = run_gambit(authors_list)

        if error:
            print(f"Error running gambit: {error}", file=sys.stderr)
            sys.exit(1)

        # Report statistics
        unique_outputs = len(set(dealiased_id for _, dealiased_id in results))
        print(f"  Gambit: {len(results)} authors -> {unique_outputs} unique ({len(results) - unique_outputs} merged)", file=sys.stderr)

        # Save gambit results to output file
        with open(gambit_output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Original Author", "Dealiased ID"])
            for original, dealiased_id in results:
                writer.writerow([original, dealiased_id])

        # Merge contributors with their information
        try:
            df_merged = merge_contributors(input_file, results)
            df_merged.to_csv(merged_output, index=False)

            print(f"  Final: {len(authors_list)} contributors -> {len(df_merged)} merged ({len(authors_list) - len(df_merged)} reduction)", file=sys.stderr)
            print(f"  Saved: {merged_output}\n", file=sys.stderr)
        except Exception as e:
            print(f"Error merging contributors: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)



if __name__ == '__main__':
    main()

