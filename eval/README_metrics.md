# Configuration Knowledge Distribution Metrics

This script (`compute_config_knowledge_metrics.py`) computes configuration-knowledge distribution metrics from contributor data.

## Overview

The script analyzes contributor activity and configuration file ownership to identify knowledge concentration risks and specialization patterns in software projects.

## Metrics Computed

### A) Global Inequality Context

#### 1. Gini Coefficient (gini_all)
**Definition:** Measures inequality in configuration commit distribution across **all contributors** (including those with 0 config commits).

**Range:** 0 to 1
- **0** = Perfect equality (everyone contributes equally)
- **1** = Perfect inequality (one person has all commits)

**Interpretation:**
- **< 0.5**: Low inequality - configuration work is well distributed
- **0.5 - 0.7**: Moderate inequality - some concentration but reasonable distribution
- **0.7 - 0.9**: High inequality - significant concentration of config knowledge
- **> 0.9**: Extreme inequality - severe bus factor risk

**Example:** Flask `gini_all = 0.991` means configuration knowledge is extremely concentrated relative to the total contributor base (818 contributors, but only 52 touch config).

#### 2. Gini Coefficient - Active Only (gini_active)
**Definition:** Measures inequality in configuration commit distribution among **only active contributors** (config_commits > 0).

**Range:** 0 to 1 (same as gini_all)

**Interpretation:**
- Lower than `gini_all` because it excludes non-config contributors
- High value indicates concentration even among those who work on configuration
- If `gini_all` is high but `gini_active` is moderate, it means few people touch config but those who do share work somewhat evenly
- If both are high, config work is dominated by a small elite even among config contributors

**Example:** Flask `gini_active = 0.858` means even among the 52 people who touch config, the work is highly concentrated (a few people dominate).

**Comparison:**
```
gini_all = 0.99, gini_active = 0.86  → Extreme concentration overall and among active contributors (Flask)
gini_all = 0.95, gini_active = 0.40  → Few touch config, but those who do share work evenly
```

---

### B) Technology-Centric Knowledge Concentration

#### 3. ENC (Effective Number of Contributors)
**Definition:** Inverse of the Herfindahl-Hirschman Index (HHI). Measures the "effective" number of contributors to a technology, accounting for unequal contributions.

**Formula:** `ENC = 1 / Σ(p_i²)` where `p_i` is contributor i's share of commits for that technology.

**Range:** 1 to N (where N = actual number of contributors)
- **1.0** = One person effectively owns the technology
- **Close to N** = Contributions are evenly distributed

**Interpretation:**
- **1.0 - 2.0**: Severe concentration - essentially one or two people
- **2.0 - 5.0**: Moderate concentration - small group dominates
- **> 5.0**: Good distribution - work is shared among many

**Example:**
```
Technology: appveyor
  - 7 actual contributors
  - ENC = 1.27
  - Interpretation: Despite 7 people touching it, it's effectively maintained by ~1-2 people
```

**Why it matters:** A technology with 10 contributors but ENC = 1.5 is riskier than one with 5 contributors and ENC = 4.5.

#### 4. TCS (Top Contributor Share)
**Definition:** The proportion of commits held by the single largest contributor to a technology.

**Range:** 0 to 1
- **0.33** = Top contributor has 33% of commits
- **1.0** = Top contributor has 100% of commits

**Interpretation:**
- **< 0.50**: Healthy - no single person dominates
- **0.50 - 0.70**: Moderate concentration - top contributor significant but not dominant
- **0.70 - 0.85**: High concentration - one person is the primary maintainer
- **> 0.85**: Extreme concentration - bus factor of 1

**Example:**
```
Technology: appveyor, TCS = 0.913
  → One person made 91.3% of commits to appveyor config
  → High bus factor risk
```

#### 5. Orphaned Technologies
**Definition:** Technologies with **exactly 1 active contributor** AND at least `min_commits_k` total commits.

**Value:** Boolean (True/False)

**Interpretation:**
- **True** = Critical bus factor risk - only one person knows this technology
- **False** = Multiple people have touched this technology

**Threshold:** Uses `--min_commits_k` (default: 5) to filter out trivial technologies.

**Example:**
```
Technology: pre-commit.yaml
  - orphaned = True
  - 1 contributor, 29.47 commits
  → If that person leaves, no one else knows this config
```

**Action:** Orphaned technologies should be prioritized for knowledge transfer.

#### 6. Endangered Technologies
**Definition:** Technologies meeting **all** of these criteria:
- TCS ≥ 0.80 (one person has 80%+ of commits)
- ENC ≤ 1.5 (effectively 1-2 people)
- Total commits ≥ `min_commits_k`

**Value:** Boolean (True/False)

**Interpretation:**
- **True** = High concentration risk - heavily dependent on one person
- **False** = Healthier distribution

**Example:**
```
Technology: azure pipelines
  - TCS = 0.853, ENC = 1.36
  - endangered = True
  → One person dominates, low diversity
```

**Difference from Orphaned:**
- Orphaned = literally only 1 person (most severe)
- Endangered = multiple people but extreme concentration (severe)

#### 7. KDP (Knowledge Diffusion Potential)
**Definition:** Measures whether **non-top contributors** are generalists who could absorb knowledge if the top contributor leaves.

**Formula:** `KDP = mean(1 - TII)` for all non-top contributors to a technology.

**Range:** 0 to 1
- **0** = Non-top contributors are all specialists (won't help)
- **1** = Non-top contributors are all generalists (can absorb knowledge)

**Interpretation:**
- **< 0.3**: Low potential - others are specialists, unlikely to take over
- **0.3 - 0.6**: Moderate potential - some generalists present
- **0.6 - 0.9**: High potential - many generalists who could learn
- **1.0**: Maximum potential - all others are pure generalists

**Example:**
```
Technology: tox
  - Top contributor has 29% share
  - KDP = 0.96
  → If top contributor leaves, many generalists can step up

Technology: appveyor
  - Top contributor has 91% share
  - KDP = 1.0
  → Non-top contributors are generalists BUT they have little experience (low share)
```

**Nuance:** High KDP is good for resilience BUT means little if non-top contributors have trivial contributions.

**Set to 0 when:** Technology has ≤ 1 contributor (no one to diffuse to).

---

### C) Contributor-Centric Specialization

#### 8. TII (Technology Isolation Index)
**Definition:** Measures how specialized a contributor is across configuration technologies using normalized entropy.

**Formula:**
```
H = -Σ(p_t × log(p_t))  [Shannon entropy]
TII = 1 - H / log(|T|)   [normalized]
```
where `p_t` is share of contributor's config commits to technology t, and |T| is number of technologies.

**Range:** 0 to 1
- **0** = Pure generalist (spreads work evenly across many technologies)
- **1** = Pure specialist (works on exactly one technology)

**Interpretation:**
- **0.0 - 0.3**: Generalist - works across many technologies
- **0.3 - 0.6**: Moderate specialization
- **0.6 - 0.9**: High specialization - focuses on few technologies
- **1.0**: Complete isolation - only touches one technology

**Special Cases:**
- TII = 0 if contributor has 0 config commits
- TII = 1.0 if contributor works on exactly 1 technology

**Example:**
```
Contributor A: 100 commits across 10 technologies evenly → TII ≈ 0.0 (generalist)
Contributor B: 100 commits, 95 to Docker, 5 to GitHub Actions → TII ≈ 0.8 (specialist)
Contributor C: 50 commits all to Travis CI → TII = 1.0 (isolated)
```

**Why it matters:**
- Generalists (low TII) provide resilience - can help with many technologies
- Specialists (high TII) are risky - deep knowledge but narrow
- A team of all specialists is fragile

#### 9. Number of Technologies (num_technologies)
**Definition:** Count of distinct configuration technologies a contributor has touched.

**Range:** 0 to N (where N = total technologies in project)

**Interpretation:**
- **0**: Contributor never touched config files
- **1-2**: Narrow focus - specialist
- **3-5**: Moderate breadth
- **> 5**: Broad involvement - generalist

**Correlation with TII:**
- High num_technologies + low TII = true generalist (works evenly across many)
- High num_technologies + high TII = specialist who dabbles (focuses on one, touches others minimally)
- Low num_technologies + high TII = pure specialist

**Example:**
```
Contributor: Maintainer A
  - num_technologies = 15
  - TII = 0.2
  → Works on 15 different config technologies, spreads effort evenly (valuable generalist)

Contributor: Maintainer B
  - num_technologies = 8
  - TII = 0.9
  → Touches 8 technologies but focuses 90% effort on one (specialist who dabbles)
```

---

## Interpreting Metrics Together

### High-Risk Patterns
1. **Bus Factor Risk:**
   - High gini_all + high gini_active
   - Many orphaned or endangered technologies
   - Low ENC, high TCS across technologies

2. **Knowledge Silos:**
   - High TII for key contributors
   - Low num_technologies per contributor
   - Low KDP for critical technologies

3. **Succession Risk:**
   - Orphaned technologies with TCS = 1.0
   - Top contributors with high TII (specialists)
   - Low KDP (no generalists to take over)

### Healthy Patterns
1. **Good Distribution:**
   - Moderate gini_all and gini_active (< 0.7)
   - High ENC (> 3.0) for critical technologies
   - TCS < 0.6 for most technologies

2. **Knowledge Resilience:**
   - Many contributors with low TII (generalists)
   - High num_technologies per active contributor
   - High KDP for important technologies

3. **Low Concentration:**
   - Few or no orphaned/endangered technologies
   - ENC close to actual contributor count
   - Flat distribution of TCS values

## Usage

### Basic Usage

```bash
python compute_config_knowledge_metrics.py --input <path/to/csv>
```

### With Custom Parameters

```bash
python compute_config_knowledge_metrics.py \
  --input ../data/projects_contributors_merged/flask_contributors_merged.csv \
  --min_commits_k 10 \
  --out_dir ../data/project_metrics \
  --delimiter_regex "[;,|]"
```

### Batch Processing Multiple Projects

#### Option 1: Using --all flag (Recommended)

Process all projects in a directory using the `--all` flag:

```bash
python compute_config_knowledge_metrics.py --all --input <directory> --out_dir <output> --min_commits_k <threshold>

# Example
python compute_config_knowledge_metrics.py --all --input ../data/projects_contributors_merged --out_dir ../data/projects_contributors_metrics --min_commits_k 5
```

Output:
```
======================================================================
Batch Processing: 100 projects
======================================================================
Input directory: ../data/projects_contributors_merged
Output directory: ../data/projects_contributors_metrics
Min commits threshold: 5
======================================================================

[1/100] Processing: flask
  ✓ Success

[2/100] Processing: bitcoin
  ✓ Success

...

======================================================================
Batch Processing Complete
======================================================================
Total projects: 100
Successful: 98
Failed: 2

Failed projects:
  - project-with-error
  - another-failed-project

Output directory: ../data/projects_contributors_metrics
======================================================================
```

#### Option 2: Using bash script

Alternatively, use the provided bash script:

```bash
./run_all_projects.sh [input_dir] [output_dir] [min_commits_k]

# Example
./run_all_projects.sh ../data/projects_contributors_merged ../data/project_contributor_metrics 5
```

Both methods process all `*_contributors_merged.csv` files and generate prefixed outputs for each project.

### Arguments

- `--input`: Path to CSV file or directory (default: `../data/projects_contributors_merged/`)
  - If directory without `--all`, uses first CSV file found
  - If directory with `--all`, processes all `*_contributors_merged.csv` files
- `--all`: Process all `*_contributors_merged.csv` files in the input directory (flag, no value needed)
- `--min_commits_k`: Minimum commits for orphaned/endangered classification (default: 5)
- `--out_dir`: Output directory (default: `../data/projects_contributors_metrics`)
- `--delimiter_regex`: Regex for splitting list columns (default: `[;,|]`)

## Input CSV Requirements

The script auto-detects columns with flexible naming:

### Required Columns
- **Contributor identifier**: Column containing names/IDs (auto-detects: "Contributor", "Author", "name", "login", etc.)
- **Config commits**: Numeric column (auto-detects: contains "config" AND "commit")

### Optional Columns
- **Non-config commits**: Numeric column (auto-detects: contains "non"/"code" AND "commit")
- **Config files**: List of files/technologies per contributor (auto-detects: contains "files", "technology", "path", etc.)
  - Can be Python list format: `['file1.yml', 'file2.json']`
  - Can be delimited: `file1.yml;file2.json;file3.toml`

## Output Files

All output files are prefixed with the project name extracted from the input filename.

**Naming Convention:**
- Input: `projectname_contributors_merged.csv`
- Outputs:
  - `projectname_contributors_metrics.csv`
  - `projectname_technologies_metrics.csv`
  - `projectname_metadata.json`

**Example:** For `flask_contributors_merged.csv`:
- `flask_contributors_metrics.csv`
- `flask_technologies_metrics.csv`
- `flask_metadata.json`

### 1. `<project>_contributors_metrics.csv`
- contributor_id
- config_commits (rounded to 2 decimals)
- non_config_commits (rounded to 2 decimals, if present in input)
- tii (Technology Isolation Index, rounded to 4 decimals)
- num_technologies (integer)
- technologies (list of technology names, alphabetically sorted)
  - Format: Python list representation, e.g., `['appveyor', 'github-action', 'tox']`
  - Empty list `[]` for contributors with no config commits or no recognized technologies

### 2. `<project>_technologies_metrics.csv`
- technology
- total_config_commits (rounded to 2 decimals)
- num_active_contributors (integer)
- enc (Effective Number of Contributors, rounded to 4 decimals)
- tcs (Top Contributor Share, rounded to 4 decimals)
- orphaned (boolean)
- endangered (boolean)
- kdp (Knowledge Diffusion Potential, rounded to 4 decimals)

### 3. `<project>_metadata.json`
- timestamp
- project_name
- input_file
- detected_columns
- assumptions
- thresholds
- summary metrics:
  - gini_all, gini_active (rounded to 4 decimals)
  - num_technologies, num_contributors, num_active_contributors
  - num_orphaned, orphaned_technologies (list of technology names)
  - num_endangered, endangered_technologies (list of technology names)

### Rounding Precision

All calculated values are rounded for readability:
- **Metric values** (ENC, TCS, TII, KDP, Gini): 4 decimal places
- **Commit counts** (total_config_commits, config_commits, non_config_commits): 2 decimal places
- **Counts** (num_contributors, num_technologies): integers (no rounding)

## Technology Detection

Technologies are extracted from file paths using `mapping.py::get_technology()`, which maps specific file patterns to technology names (e.g., `.travis.yml` → `travis`, `tox.ini` → `tox`, etc.).

**Important**: Files that don't match any pattern in the mapping (where `get_technology()` returns `None`) are **excluded** from the analysis. Only files with recognized technology mappings are included in metrics calculations.

### Key Assumptions

1. **Equal-split**: When only file lists are available (not per-technology commit counts), each contributor's config commits are divided evenly across technologies they touched.

2. **Exclusion of unmapped files**: Configuration files without a recognized technology mapping are excluded from all metrics calculations.

## Example Output

### Console Output

```
============================================================
SUMMARY
============================================================
Global Gini Coefficient (all): 0.9910
Global Gini Coefficient (active only): 0.8578
Number of technologies: 15
Orphaned technologies: 1
Endangered technologies: 3

Top 10 most concentrated technologies (lowest ENC):
                        technology      enc      tcs  num_active_contributors
                   pre-commit.yaml 1.000000 1.000000                        1
                          appveyor 1.269899 0.885011                        7
                   azure pipelines 1.364128 0.853282                        8
```

### Metadata JSON Example

```json
{
  "timestamp": "2026-01-13T11:08:45.939604",
  "project_name": "flask",
  "metrics": {
    "gini_all": 0.991,
    "gini_active": 0.8578,
    "num_technologies": 12,
    "num_contributors": 818,
    "num_active_contributors": 52,
    "num_orphaned": 1,
    "orphaned_technologies": [
      "pre-commit.yaml"
    ],
    "num_endangered": 3,
    "endangered_technologies": [
      "appveyor",
      "azure pipelines",
      "pre-commit.yaml"
    ]
  }
}
```

## Graceful Degradation

The script handles missing optional columns gracefully:
- If no config files column exists, only global Gini metrics are computed
- Missing non-config commits column is simply omitted from outputs
- All division-by-zero cases are handled deterministically

## Validation

The script includes built-in sanity checks:
- Shares sum to 1.0 per technology
- ENC within [1, num_active_contributors]
- TCS within [0, 1]
- TII within [0, 1]
