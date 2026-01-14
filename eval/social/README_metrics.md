# Configuration Work Distribution Metrics

This script (`compute_config_knowledge_metrics.py`) computes configuration-knowledge distribution metrics from contributor data.

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

### D) File-Level Expertise Metrics

These metrics leverage the actual file touch counts to measure contributor depth and engagement patterns with configuration files.

#### 10. Total Config Files (total_config_files)
**Definition:** Number of unique configuration files a contributor has touched.

**Range:** 0 to N (where N = total config files in project)

**Interpretation:**
- **0**: No config file contributions
- **1-20**: Narrow focus - works on specific subsystem
- **20-100**: Moderate breadth - touches multiple subsystems
- **> 100**: Very broad - involved across many config areas

**Why it matters:** More files touched suggests broader knowledge of configuration landscape.

#### 11. Total File Touches (total_file_touches)
**Definition:** Sum of all file touch counts across all configuration files.

**Range:** 0 to ∞

**Interpretation:**
- **Equal to total_config_files**: Each file touched exactly once (casual contributor)
- **Much higher than total_config_files**: Repeatedly touches same files (deep engagement)

**Example:**
```
Contributor A: 182 files, 341 touches → 341/182 = 1.87 avg
  → Revisits files, suggesting iterative development or maintenance

Contributor B: 21 files, 21 touches → 21/21 = 1.0 avg
  → One-time contributions, no repeated engagement
```

#### 12. Avg Touches Per File (avg_touches_per_file)
**Definition:** Average number of times a contributor touched each file they worked on.

**Formula:** `total_file_touches / total_config_files`

**Range:** 1.0 to ∞
- **1.0** = Touched each file exactly once
- **1.5-2.5** = Moderate revisitation
- **> 3.0** = Heavy revisitation - deep engagement

**Interpretation:**
- **1.0**: One-time contributor or broad coverage without depth
- **1.5-2.5**: Iterative development, bug fixes, refinements
- **> 3.0**: Deep expert - repeatedly maintains/improves same files

**Example:**
```
Contributor: Deep Expert
  - avg_touches_per_file = 2.5
  → Returns to files for maintenance, showing sustained ownership

Contributor: Casual Contributor
  - avg_touches_per_file = 1.0
  → Drive-by contributions, no sustained engagement
```

#### 13. Max File Touches (max_file_touches)
**Definition:** Highest touch count on any single configuration file.

**Range:** 0 to ∞

**Interpretation:**
- **1**: No file touched more than once
- **2-5**: Moderate expertise on specific files
- **> 10**: Deep expert on particular file(s)

**Why it matters:** Indicates peak expertise - files contributor knows intimately.

**Example:**
```
Contributor A: max_file_touches = 50
  → Has one or more files they're the primary maintainer of

Contributor B: max_file_touches = 1
  → No deep specialization on any single file
```

#### 14. Touch Concentration (touch_concentration)
**Definition:** Gini coefficient measuring inequality in file touch distribution. Shows whether contributor spreads touches evenly or concentrates on specific files.

**Range:** 0 to 1
- **0.0** = Perfect equality (touched all files same number of times)
- **0.1-0.3** = Low concentration (relatively even distribution)
- **0.3-0.6** = Moderate concentration (some focus files)
- **> 0.6** = High concentration (heavily focused on few files)

**Interpretation:**
- **Low (0.0-0.2)**: Generalist - spreads attention across many files evenly
- **Moderate (0.2-0.4)**: Balanced - has focus files but maintains others too
- **High (> 0.4)**: Specialist - deep expertise on few files, shallow on others

**Comparison with TII:**
- **touch_concentration** = inequality across **files** (within a contributor)
- **TII** = specialization across **technologies** (within a contributor)

**Example:**
```
Contributor A:
  - total_config_files = 100
  - total_file_touches = 150
  - touch_concentration = 0.05
  → Touched many files, mostly once or twice each (even distribution)

Contributor B:
  - total_config_files = 50
  - total_file_touches = 100
  - touch_concentration = 0.80
  → Most touches on just a few files (concentrated expertise)
```

---

## Calculation Details

This section documents the implementation details of how each metric is computed in `compute_config_knowledge_metrics.py`.

### A) Global Inequality Metrics - Implementation

#### Gini Coefficient Calculation (`compute_gini_coefficient`)
**Location:** compute_config_knowledge_metrics.py:189-211

**Algorithm:**
```python
# Sort values in ascending order
sorted_values = np.sort(values)
n = len(sorted_values)
total = sorted_values.sum()

# Compute Gini using standard formula
index = np.arange(1, n + 1)
gini = (2 * np.sum(index * sorted_values)) / (n * total) - (n + 1) / n
```

**Formula:** `G = (2 * Σ(i × xᵢ)) / (n × Σ(xᵢ)) - (n + 1) / n`

Where:
- `xᵢ` = sorted values in ascending order
- `i` = rank (1 to n)
- `n` = total number of values

**Special Cases:**
- Returns 0.0 if array is empty
- Returns 0.0 if sum of values is 0

**Computed Twice:**
1. **gini_all**: Applied to all contributors (including those with 0 config commits)
2. **gini_active**: Applied only to contributors where config_commits > 0

---

### B) Technology-Centric Metrics - Implementation

#### Technology Extraction (`extract_technologies`)
**Location:** compute_config_knowledge_metrics.py:156-203

**Process:**
1. Parse file list from config_files column using `parse_file_list()`
   - Handles Python list of tuples: `[('file1.yml', 2), ('file2.json', 3)]` where second element is touch count
   - Handles Python list strings: `['file1.yml', 'file2.json']` (defaults touch count to 1)
   - Handles delimited strings: `file1.yml;file2.json` (defaults touch count to 1)
2. Map each file path to technology using `mapping.py::get_technology()`
3. **Proportional distribution**: Distribute contributor's config commits proportionally based on file touch counts
   - Sum touch counts per technology: `tech_counts[tech] += count`
   - Calculate proportion: `tech_commits = config_commits * (tech_touch_count / total_touches)`
4. Create contributor-technology pairs with tech_commits

**Output:** DataFrame with columns `[contributor, technology, tech_commits]`

**Important:** Files that don't match any pattern in mapping.py (where `get_technology()` returns `None`) are excluded from analysis.

**Example:**
```
Contributor: Alice, config_commits = 100
Files touched:
  - .github/workflows/ci.yml (5 touches) → github-action
  - docker-compose.yml (10 touches) → docker compose
  - package.json (5 touches) → npm

Total touches = 20
Commits allocated:
  - github-action: 100 * (5/20) = 25 commits
  - docker compose: 100 * (10/20) = 50 commits
  - npm: 100 * (5/20) = 25 commits
```

#### Share Calculation (used by ENC and TCS)
**Location:** compute_config_knowledge_metrics.py:236

For each technology:
```python
shares = group['tech_commits'].values / total_commits
```

Where:
- `shares` = array of proportions for each contributor to this technology
- Guaranteed to sum to 1.0 (sanity check included)

#### ENC Calculation (`compute_technology_metrics`)
**Location:** compute_config_knowledge_metrics.py:241-248

**Algorithm:**
```python
# Compute Herfindahl-Hirschman Index (HHI)
hhi = np.sum(shares ** 2)

# ENC is inverse of HHI
enc = 1.0 / hhi if hhi > 0 else 0.0
```

**Formula:** `ENC = 1 / HHI = 1 / Σ(pᵢ²)`

Where:
- `pᵢ` = share of contributor i for this technology
- `HHI` = sum of squared shares

**Sanity Check:** `1.0 ≤ ENC ≤ num_contributors`

#### TCS Calculation (`compute_technology_metrics`)
**Location:** compute_config_knowledge_metrics.py:250-254

**Algorithm:**
```python
tcs = float(shares.max())
```

**Formula:** `TCS = max(p₁, p₂, ..., pₙ)`

Simply the largest share among all contributors to a technology.

**Sanity Check:** `0.0 ≤ TCS ≤ 1.0`

#### Orphaned Flag Calculation (`compute_technology_metrics`)
**Location:** compute_config_knowledge_metrics.py:256-257

**Algorithm:**
```python
orphaned = (num_contributors == 1) and (total_commits >= min_commits_k)
```

**Conditions (AND):**
1. Exactly 1 active contributor to this technology
2. Total commits meets minimum threshold (default: 5)

#### Endangered Flag Calculation (`compute_technology_metrics`)
**Location:** compute_config_knowledge_metrics.py:259-260

**Algorithm:**
```python
endangered = (tcs >= 0.80) and (enc <= 1.5) and (total_commits >= min_commits_k)
```

**Conditions (AND):**
1. Top contributor has ≥80% of commits (TCS ≥ 0.80)
2. Effective contributors ≤1.5 (ENC ≤ 1.5)
3. Total commits meets minimum threshold

#### KDP Calculation (`compute_kdp`)
**Location:** compute_config_knowledge_metrics.py:356-399

**Algorithm:**
```python
# For each technology:
# 1. Identify top contributor (max tech_commits)
top_contributor = group.loc[group['tech_commits'].idxmax(), 'contributor']

# 2. Get all non-top contributors
non_top = group[group['contributor'] != top_contributor]

# 3. Compute mean (1 - TII) for non-top contributors
tii_values = [tii_map.get(c, 0.0) for c in non_top['contributor']]
kdp = np.mean([1.0 - tii for tii in tii_values])
```

**Formula:** `KDP_t = mean(1 - TIIᵢ)` for all i ≠ top(t)

Where:
- `top(t)` = contributor with largest share for technology t
- `TIIᵢ` = Technology Isolation Index for contributor i

**Special Cases:**
- KDP = 0.0 if technology has ≤1 contributor (no non-top contributors)

---

### C) Contributor-Centric Metrics - Implementation

#### File-Level Extraction (`extract_file_level_data`)
**Location:** compute_config_knowledge_metrics.py:206-237

**Process:**
1. Parse file list from config_files column using `parse_file_list()`
2. For each file, extract: file_path, touch_count, and technology mapping
3. Create file-level records for each contributor

**Output:** DataFrame with columns `[contributor, file_path, technology, touch_count]`

**Note:** Unlike `extract_technologies`, this function includes files even if they don't map to a recognized technology (marked as 'unknown').

#### File-Level Metrics Calculation (`compute_file_level_metrics`)
**Location:** compute_config_knowledge_metrics.py:407-447

**Algorithm:**
```python
for contributor, group in file_level_df.groupby('contributor'):
    total_files = len(group)
    total_touches = group['touch_count'].sum()
    avg_touches = total_touches / total_files
    max_touches = group['touch_count'].max()

    # Compute touch concentration (Gini on file touches)
    touch_counts = group['touch_count'].values
    touch_gini = compute_gini_coefficient(touch_counts)
```

**Metrics Computed:**
1. **total_config_files**: Count of unique files (length of group)
2. **total_file_touches**: Sum of all touch counts
3. **avg_touches_per_file**: Mean of touch counts
4. **max_file_touches**: Maximum touch count
5. **touch_concentration**: Gini coefficient on file touch distribution

**Special Cases:**
- All metrics = 0 for contributors with no config file data
- avg_touches_per_file = 0.0 if total_files = 0

#### TII Calculation (`compute_tii`)
**Location:** compute_config_knowledge_metrics.py:308-370

**Algorithm:**
```python
# 1. Compute shares across technologies
shares = contributor_tech_commits.values / total

# 2. Compute Shannon entropy (skip zero shares)
nonzero_shares = shares[shares > 0]
entropy = -np.sum(nonzero_shares * np.log(nonzero_shares))

# 3. Normalize by maximum possible entropy
max_entropy = np.log(len(nonzero_shares))
tii = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
```

**Formula:**
```
H = -Σ(pₜ × ln(pₜ))           [Shannon entropy]
TII = 1 - H / ln(|T|)          [normalized]
```

Where:
- `pₜ` = share of contributor's config commits to technology t
- `|T|` = number of technologies contributor touched
- Natural logarithm (ln) is used

**Special Cases:**
- TII = 0.0 if total commits = 0
- TII = 0.0 if no technologies (empty array)
- TII = 1.0 if exactly 1 technology
- Result clamped to [0.0, 1.0] to handle floating point precision

**Interpretation:**
- TII = 0.0 → Perfect generalist (even distribution)
- TII = 1.0 → Perfect specialist (all work on one technology)

#### Number of Technologies Calculation
**Location:** compute_config_knowledge_metrics.py:338

**Algorithm:**
```python
num_technologies = len(group)
```

Simply counts distinct technologies in the contributor's tech_contrib_df group.

**Special Cases:**
- Set to 0 for contributors with no config commits or no recognized technologies

---

### Data Flow Summary

```
Input CSV (with file touch counts)
    ↓
Auto-detect columns (detect_column, auto_detect_columns)
    ↓
┌──────────────────────────────┬─────────────────────────────────┐
│ Parse file lists             │ Extract file-level data         │
│ → Extract technologies       │ (extract_file_level_data)       │
│ (extract_technologies)       │                                 │
│                              │                                 │
│ Proportional distribution:   │ Output: file-level DataFrame    │
│ commits ∝ touch counts       │ [contributor, file, technology, │
│                              │  touch_count]                   │
│ Output: tech-level DataFrame │                                 │
│ [contributor, technology,    │                                 │
│  tech_commits]               │                                 │
└──────────────────────────────┴─────────────────────────────────┘
    ↓                                        ↓
┌─────────────────────┬────────────────────┬──────────────────────┐
│  Global Metrics     │ Technology Metrics │ Contributor Metrics  │
├─────────────────────┼────────────────────┼──────────────────────┤
│ • Gini (all)        │ Per technology:    │ Per contributor:     │
│ • Gini (active)     │  • Compute shares  │  • TII               │
│                     │  • ENC (1/HHI)     │  • num_technologies  │
│                     │  • TCS (max share) │  • technologies list │
│                     │  • Orphaned flag   │  • File-level:       │
│                     │  • Endangered flag │    - total_files     │
│                     │  • KDP             │    - total_touches   │
│                     │                    │    - avg_touches     │
│                     │                    │    - max_touches     │
│                     │                    │    - touch_gini      │
└─────────────────────┴────────────────────┴──────────────────────┘
    ↓
Output: 3 files
    • <project>_contributors_metrics.csv (now includes file-level metrics)
    • <project>_technologies_metrics.csv
    • <project>_metadata.json
```

### Validation and Sanity Checks

The implementation includes assertions to ensure correctness (compute_config_knowledge_metrics.py:239, 247-248, 254, 317):

1. **Share sums**: `assert abs(shares.sum() - 1.0) < 1e-6` - shares per technology must sum to 1.0
2. **ENC bounds**: `assert 1.0 <= enc <= num_contributors + 0.01` - ENC must be between 1 and actual contributor count
3. **TCS bounds**: `assert 0.0 <= tcs <= 1.0` - TCS must be a valid proportion
4. **TII bounds**: `assert 0.0 <= tii <= 1.0` - TII must be between 0 and 1

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
   - Mix of depth (high avg_touches_per_file) and breadth (high total_config_files)

3. **Low Concentration:**
   - Few or no orphaned/endangered technologies
   - ENC close to actual contributor count
   - Flat distribution of TCS values

### File-Level Patterns

#### Deep Experts
- High `avg_touches_per_file` (> 2.0)
- High `max_file_touches` (> 10)
- Moderate to high `touch_concentration`
- **Risk:** Deep knowledge but potentially narrow
- **Value:** Sustained ownership and maintenance

#### Broad Generalists
- High `total_config_files` (> 50)
- Low `touch_concentration` (< 0.2)
- `avg_touches_per_file` ≈ 1.0
- **Risk:** Shallow knowledge, may not sustain
- **Value:** Wide system knowledge, can navigate codebase

#### Sustained Maintainers (Ideal)
- High `total_config_files` AND high `avg_touches_per_file`
- Moderate `touch_concentration` (0.2-0.4)
- High `max_file_touches` on critical files
- **Value:** Both breadth and depth, core maintainers

#### Casual Contributors
- Low `total_config_files` (< 20)
- `avg_touches_per_file` = 1.0
- `touch_concentration` = 0.0
- **Pattern:** One-time contributions, limited engagement

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

### Arguments

- `--input`: Path to CSV file or directory (default: `../data/projects_contributors_merged/`)
  - If directory without `--all`, uses first CSV file found
  - If directory with `--all`, processes all `*_contributors_merged.csv` files
- `--all`: Process all `*_contributors_merged.csv` files in the input directory (flag, no value needed)
- `--min_commits_k`: Minimum commits for orphaned/endangered classification (default: 5)
- `--out_dir`: Output directory (default: `../data/projects_contributors_metrics`)
- `--delimiter_regex`: Regex for splitting list columns (default: `[;,|]`)



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

---

## Presenting Metrics (`present_metrics.py`)

After computing metrics, use the `present_metrics.py` script to display results in a readable format.

### Features

The presentation script provides formatted views of:

1. **Global Metrics**: Project overview, Gini coefficients, orphaned/endangered counts
2. **Technology Metrics**:
   - Most concentrated technologies (lowest ENC)
   - Highest knowledge diffusion potential (KDP)
   - Most active technologies
   - Lists of orphaned/endangered technologies
3. **Contributor Metrics**:
   - Top contributors by commits
   - Specialists (high TII)
   - Generalists (low TII, many technologies)
   - Deep experts (high avg_touches_per_file)
   - Broad contributors (most files touched)
4. **Summary Statistics**: Mean, median, and standard deviation for all metrics

### Usage

#### Show All Metrics
```bash
python present_metrics.py <project_name>

# Example
python present_metrics.py 1Panel
```

#### Show Specific Sections
```bash
# Only contributor metrics
python present_metrics.py 1Panel --sections contributor

# Multiple sections
python present_metrics.py 1Panel --sections global technology

# With custom top N
python present_metrics.py 1Panel --top_n 5

# Custom metrics directory
python present_metrics.py 1Panel --metrics_dir /path/to/metrics
```

### Arguments

- `project_name` (required): Project name (e.g., "1Panel" for 1Panel_*_metrics.csv files)
- `--metrics_dir`: Directory containing metrics files (default: `../data/projects_contributors_metrics`)
- `--top_n`: Number of top items to show in rankings (default: 10)
- `--sections`: Sections to display - choices: `global`, `technology`, `contributor`, `summary`, `all` (default: `all`)

### Example Output

```
================================================================================
                              CONTRIBUTOR METRICS
================================================================================

--------------------------------------------------------------------------------
  Top 5 Contributors by Config Commits
--------------------------------------------------------------------------------
                                    contributor_id  config_commits  num_technologies    tii  total_config_files  total_file_touches  avg_touches_per_file
zhengkunwang223 <31820853+zhengkunwang223@users...>             347                15 0.1731                 182                 341                  1.87
                         ssongliu <songlius11@...>             282                15 0.1578                 168                 213                  1.27
               wanghe-fit2cloud <wanghe@fit2clo...>              25                11 0.1862                  91                 170                  1.87

--------------------------------------------------------------------------------
  Top 5 Deep Experts (Highest Avg Touches per File)
--------------------------------------------------------------------------------
                                    contributor_id  avg_touches_per_file  max_file_touches  total_config_files  total_file_touches  touch_concentration
               wanghe-fit2cloud <wanghe@fit2clo...>                  1.87                 3                  91                 170               0.1789
zhengkunwang223 <31820853+zhengkunwang223@users...>                  1.87                 5                 182                 341               0.3227
                         ssongliu <songlius11@...>                  1.27                 2                 168                 213               0.1547
```

The script automatically reads the three output files:
- `{project}_contributors_metrics.csv`
- `{project}_technologies_metrics.csv`
- `{project}_metadata.json`

