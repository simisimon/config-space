import pandas as pd
import pytest
import ast

# Code to be tested
def analyze_configurations(target_df, other_dfs):
    target_df['Set in other projects'] = 0
    target_df['Total occurrences'] = 0
    target_df['Changed globally'] = 0

    for index, row in target_df.iterrows():
        option = row['Option']

        for other_df in other_dfs:
            # Find all rows in other_df where the option matches
            matching_rows = other_df[other_df['Option'] == option]
            match_count = len(matching_rows)

            if match_count > 0:
                # Increment "Set in other projects" by 1 (project-level count)
                target_df.loc[index, 'Set in other projects'] += 1

                # Increment "Total occurrences" by the total count of matches
                target_df.loc[index, 'Total occurrences'] += match_count

                # Check each match for changes in values
                for _, match_row in matching_rows.iterrows():
                    # Parse the 'Values' column (convert from string to list if necessary)
                    raw_values = match_row['Values']
                    try:
                        values = ast.literal_eval(raw_values) if isinstance(raw_values, str) else raw_values
                    except (ValueError, SyntaxError):
                        values = [raw_values]  # Fall back to treating as a single value

                    # Ensure `values` is iterable
                    if not isinstance(values, (list, set, tuple)):
                        values = [values]

                    unique_values = set(values)
                    if len(unique_values) > 1:
                        # Increment "Changed globally" for each such occurrence
                        target_df.loc[index, 'Changed globally'] += 1

    return target_df

def test_analyze_configurations():
    # Create target_df
    target_df = pd.DataFrame({
        'Option': ['timeout', 'retries', 'delay'],
        'Values': ["['30']", "['5']", "['100']"]
    })

    # Create other_dfs
    other_df1 = pd.DataFrame({
        'Option': ['timeout', 'timeout', 'retries'],
        'Values': ["['30']", "['60']", "['5', '5']"]
    })

    other_df2 = pd.DataFrame({
        'Option': ['timeout', 'delay'],
        'Values': ["['30']", "['200']"]
    })

    other_dfs = [other_df1, other_df2]

    # Analyze configurations
    result_df = analyze_configurations(target_df.copy(), other_dfs)

    # Expected results
    expected = pd.DataFrame({
        'Option': ['timeout', 'retries', 'delay'],
        'Values': ["['30']", "['5']", "['100']"],
        'Set in other projects': [2, 1, 1],
        'Total occurrences': [3, 1, 1],
        'Changed globally': [2, 0, 0]
    })

    # Assert equality for each column
    pd.testing.assert_frame_equal(result_df, expected)

if __name__ == "__main__":
    pytest.main()