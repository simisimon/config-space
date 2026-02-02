"""
Streamlit app for exploring pre-computed config-space analysis results.

Run:  cd src && streamlit run app.py
"""

import ast
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"

ALL_COMPANIES = ["netflix", "uber", "disney", "airbnb", "google", "facebook"]

PAGES_TECHNOLOGICAL = [
    "Technology Landscape",
    "Option Usage",
    "Frequent Itemsets",
]

PAGES_CLUSTERING = [
    "Clustering: Technology Stack",
    "Clustering: Technology",
    "Clustering: Config Profile",
]

PAGES_SOCIAL = [
    "Participation Rate",
    "Gini Index",
    "Commit Share",
    "Technology Frequency",
]

# Copied from technology_composition.py (not imported because it triggers
# kaleido.get_chrome_sync() at module level).
FILE_TYPES = {
    "yaml": [
        "dependabot", "codecov", "buildkite", "ansible", "ansible playbook",
        "kubernetes", "docker compose", "github-action", "goreleaser", "mkdocs",
        "swiftlint", "sourcery", "circleci", "elasticsearch", "flutter",
        "mockery", "codeclimate", "heroku", "spring", "travis", "bandit",
        "amplify", "drone", "yaml", "buf", "github", "gitpod", "appveyor",
        "pnpm", "rubocop", "gitbook", "jitpack", "pre-commit", "snapscraft",
        "eslint", "markdownlint", "stylelint", "postcss", "mocha", "yarn",
        "golangci-lint", "jekyll", "codebeaver", "crowdin", "readthedocs",
        "graphql", "swagger", "chart testing", "codebuild", "lefthook",
        "hugoreleaser", "triagebot", "jazzy", "clomonitor", "prometheus",
        "helm", "cspell", "azure pipelines", "logstash", "verdaccio",
        "github issues", "github funding", "github config",
        "github codespaces", "ultralytics yolo", "tslint", "clusterfuzz",
        "jinja", "conda",
    ],
    "properties": [
        "alluxio", "spring", "kafka", "gradle", "cirrus", "gradle wrapper",
        "maven wrapper", "properties", "log4j",
    ],
    "json": [
        "angular", "eslint", "prettier", "lerna", "firebase", "renovate",
        "stripe", "tsconfig", "nodejs", "vercel", "npm", "cypress",
        "devcontainer", "deno", "cmake", "bower", "json", "babel", "turborepo",
        "vscode", "apify", "gocrazy", "jest", "markdownlint", "stylelint",
        "postcss", "mocha", "golangci-lint", "wrangler", "vcpkg", "changesets",
        "fuel", "knip", "tsdoc", "nodemon", "graphql", "swagger", "nixpacks",
        "lefthook", "bundlemon", "cspell", "biomejs", "oxc", "claude code",
        "cursor", "zed", "tslint", "jsdoc", "pyright",
    ],
    "xml": [
        "maven", "android", "hadoop common", "hadoop hbase", "hadoop hdfs",
        "mapreduce", "xml", "yarn", "log4j",
    ],
    "toml": [
        "cargo", "netlify", "poetry", "toml", "rustfmt", "flyio", "taplo",
        "cross", "cargo make", "stylua", "trunk", "rust", "clippy", "ruff",
        "typos", "golangci-lint", "jekyll", "wrangler", "graphql", "nixpacks",
        "lefthook", "deepsource", "git cliff", "reuse", "kodiak", "streamlit",
        "mdbook", "lychee", "cdbindgen", "triagebot", "taoskeeper", "cspell",
        "bun",
    ],
    "conf": ["mongodb", "nginx", "postgresql", "rabbitmq", "redis", "apache", "conf"],
    "ini": ["mysql", "php", "ini", "mypy", "tox", "grafana"],
    "cfg": ["zookeeper"],
    "python": ["django"],
    "other": ["docker"],
}

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_csv(path: str, nrows: int | None = None) -> pd.DataFrame | None:
    """Load a CSV with optional row limit. Returns None if file missing."""
    if not os.path.exists(path):
        return None
    kwargs = {}
    if nrows is not None:
        kwargs["nrows"] = nrows
    return pd.read_csv(path, **kwargs)


def run_script(script_path: str, args: list[str], cwd: str) -> bool:
    """Run a Python script in a subprocess. Returns True on success."""
    cmd = [sys.executable, script_path] + args
    with st.spinner(f"Running `{' '.join(cmd)}` …"):
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Script failed (exit {result.returncode})")
        if result.stderr:
            st.code(result.stderr, language="text")
        return False
    if result.stdout:
        st.code(result.stdout, language="text")
    return True


def show_or_generate(
    data_path: str,
    script_path: str,
    args: list[str],
    cwd: str,
    label: str = "Generate data",
) -> bool:
    """Check if *data_path* exists. If not, show a generate button.

    Returns True when data is present (possibly after generation).
    """
    if os.path.exists(data_path):
        return True
    st.info(f"Data not found: `{data_path}`")
    if st.button(label, key=f"gen_{data_path}"):
        run_script(script_path, args, cwd)
        # Rerun if the file appeared, even when the script reported errors
        if os.path.exists(data_path):
            st.cache_data.clear()
            st.rerun()
    return False


# ---------------------------------------------------------------------------
# Treemap builder (adapted from technology_composition.py ~30 lines)
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("-", " ").replace("_", " ").split())


def _build_concept_to_filetype() -> dict[str, str]:
    m = {}
    for ext, concepts in FILE_TYPES.items():
        for c in concepts:
            m[_norm(c)] = ext
    return m


_CONCEPT_TO_FILETYPE = _build_concept_to_filetype()


def build_treemap_figure(df_tech: pd.DataFrame):
    """Return a Plotly treemap figure from the technologies CSV."""
    tech_counts: dict[tuple[str, str], int] = {}
    for _, row in df_tech.iterrows():
        techs = row["technologies"]
        if isinstance(techs, str):
            techs = ast.literal_eval(techs)
        for raw in techs:
            normalized = " ".join(raw.lower().split("-")).strip()
            ft = _CONCEPT_TO_FILETYPE.get(normalized, "other")
            label = "other" if normalized == ft else normalized
            key = (ft, label)
            tech_counts[key] = tech_counts.get(key, 0) + 1

    rows = [
        {"File Type": ft, "Technology": concept, "Count": cnt}
        for (ft, concept), cnt in tech_counts.items()
    ]
    df_counts = pd.DataFrame(rows)
    df_counts["Label"] = df_counts["Technology"] + " (" + df_counts["Count"].astype(str) + ")"

    fig = px.treemap(
        df_counts,
        path=["File Type", "Label"],
        values="Count",
        color="File Type",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title=f"Technology Landscape Across {len(df_tech)} Projects",
    )
    fig.update_traces(
        root_color="lightgrey",
        textfont=dict(family="Arial Black, Arial Bold, sans-serif", size=16, color="black"),
        marker=dict(line=dict(width=2, color="white")),
    )
    fig.update_layout(width=1100, height=700, margin=dict(t=50, l=25, r=25, b=25))
    return fig


# ---------------------------------------------------------------------------
# Option-usage loader (safe import from plot_option_usage.py)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(SRC_DIR))
from plot_option_usage import load_option_usage  # noqa: E402


def _load_or_compute_option_usage(
    company: str, technology: str,
) -> tuple[dict[str, int], int, dict[str, dict[str, int]]]:
    """Return cached option-usage results or compute and cache them."""
    cache_path = DATA_DIR / company / "option_usage" / f"{technology}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        return data["option_counts"], data["total_projects"], data["option_values"]

    data_dir = str(DATA_DIR / company / "latest_commit")
    if not Path(data_dir).exists():
        return {}, 0, {}

    option_counts, total_projects, option_values = load_option_usage(
        data_dir, technology, collect_values=True,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({
            "option_counts": option_counts,
            "total_projects": total_projects,
            "option_values": option_values,
        }, f)

    return option_counts, total_projects, option_values


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------


def page_technology_landscape(company: str):
    st.header("Technology Landscape")

    is_all = company == "All Companies"
    data_company = "all_companies" if is_all else company

    tech_dir = DATA_DIR / data_company / "technological"
    tech_csv = str(tech_dir / "technologies.csv")
    stats_csv = str(tech_dir / "technology_statistics.csv")

    # Check data availability; offer generation if missing
    if is_all:
        gen_args = ["--all", "--refresh"]
        gen_label = "Generate technological data for all companies"
    else:
        gen_args = ["--input", company, "--refresh"]
        gen_label = f"Generate technological data for {company}"

    if not show_or_generate(
        data_path=tech_csv,
        script_path="technology_composition.py",
        args=gen_args,
        cwd=str(SRC_DIR / "technological"),
        label=gen_label,
    ):
        return

    # --- Interactive treemap ---
    df_tech = load_csv(tech_csv)
    if df_tech is not None and not df_tech.empty:
        fig = build_treemap_figure(df_tech)
        st.plotly_chart(fig, use_container_width=True)

    # --- Technology statistics table ---
    df_stats = load_csv(stats_csv)
    if df_stats is not None:
        st.subheader("Technology Statistics")
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # --- Static fallback images ---
    landscape_png = tech_dir / "technology_landscape.png"
    combos_png = tech_dir / "technology_combinations.png"
    with st.expander("Static images (pre-generated)"):
        if landscape_png.exists():
            st.image(str(landscape_png), caption="Technology Landscape")
        else:
            st.caption("technology_landscape.png not found")
        if combos_png.exists():
            st.image(str(combos_png), caption="Technology Combinations")
        else:
            st.caption("technology_combinations.png not found")


def page_option_usage(company: str):
    st.header("Option Usage")

    is_all = company == "All Companies"

    # Determine available technologies from technology_statistics.csv
    available_techs: list[str] = []
    if is_all:
        for c in ALL_COMPANIES:
            stats_path = DATA_DIR / c / "technological" / "technology_statistics.csv"
            df_s = load_csv(str(stats_path))
            if df_s is not None and "technology" in df_s.columns:
                available_techs.extend(df_s["technology"].tolist())
        available_techs = sorted(set(available_techs))
    else:
        stats_path = DATA_DIR / company / "technological" / "technology_statistics.csv"
        df_s = load_csv(str(stats_path))
        if df_s is not None and "technology" in df_s.columns:
            available_techs = sorted(df_s["technology"].tolist())

    if not available_techs:
        st.warning("No technology statistics found. Generate the Technology Landscape data first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        technology = st.selectbox("Technology", available_techs, index=0)
    with col2:
        top_n = st.slider("Top N options", min_value=5, max_value=200, value=30, step=5)

    if not technology:
        return

    # Company exclusion filter (All Companies mode)
    excluded_companies: set[str] = set()
    if is_all:
        excluded = st.multiselect(
            "Exclude companies",
            ALL_COMPANIES,
            default=[],
            help="Select companies to exclude from the statistics.",
        )
        excluded_companies = set(excluded)

    # Load option usage (per-company results are cached to disk)
    with st.spinner("Computing option usage …"):
        if is_all:
            combined_counts: dict[str, int] = defaultdict(int)
            combined_values: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            combined_total = 0
            for c in ALL_COMPANIES:
                if c in excluded_companies:
                    continue
                counts, total, values = _load_or_compute_option_usage(c, technology)
                combined_total += total
                for opt, cnt in counts.items():
                    combined_counts[opt] += cnt
                for opt, val_counts in values.items():
                    for val, cnt in val_counts.items():
                        combined_values[opt][val] += cnt
            option_counts = dict(combined_counts)
            option_values = {k: dict(v) for k, v in combined_values.items()}
            total_projects = combined_total
        else:
            option_counts, total_projects, option_values = _load_or_compute_option_usage(
                company, technology,
            )
            if not option_counts and total_projects == 0:
                st.warning(f"latest_commit directory not found for {company}.")
                return

    if total_projects == 0:
        st.info(f"No projects found using **{technology}**.")
        return

    st.metric("Projects using this technology", total_projects)

    # Sort and limit
    sorted_opts = sorted(option_counts.keys(), key=lambda o: option_counts[o], reverse=True)
    if top_n:
        sorted_opts = sorted_opts[:top_n]

    counts = [option_counts[o] for o in sorted_opts]
    percentages = [(c / total_projects) * 100 for c in counts]

    unique_values = [len(option_values.get(o, {})) for o in sorted_opts]

    df_bar = pd.DataFrame({
        "Option": sorted_opts,
        "# Projects": counts,
        "% Projects": percentages,
        "# Unique Values": unique_values,
    })

    label = company if not is_all else "all companies"
    fig = px.bar(
        df_bar,
        x="Option",
        y="% Projects",
        title=f"Option usage for '{technology}' (n={total_projects} projects, {label})",
        labels={"% Projects": "% of Projects"},
    )
    fig.update_layout(
        xaxis_tickangle=-90,
        height=600,
        xaxis_title="Configuration Option",
        yaxis_title="% of Projects Using This Option",
        yaxis_range=[0, 105],
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df_bar, use_container_width=True, hide_index=True)

    with st.expander("Option values"):
        selected_option = st.selectbox("Select an option", sorted_opts, key="opt_val_select")
        val_counts = option_values.get(selected_option, {})
        if val_counts:
            df_vals = pd.DataFrame(
                sorted(val_counts.items(), key=lambda x: x[1], reverse=True),
                columns=["Value", "Count"],
            )
            st.caption(f"{len(df_vals)} unique value(s), {df_vals['Count'].sum()} total occurrences")
            st.dataframe(df_vals, use_container_width=True, hide_index=True)
        else:
            st.info("No values recorded for this option.")


def page_frequent_itemsets(company: str):
    st.header("Frequent Itemsets")

    itemset_dir = DATA_DIR / company / "frequent_itemsets"

    tab_configs = [
        (
            "Value Equality",
            "frequent_itemsets_value_equality.csv",
            "value_equality",
        ),
        (
            "Co-occurrence",
            "frequent_itemsets_cooccurrence.csv",
            "cooccurrence",
        ),
        (
            "Structural N-grams",
            "frequent_structural_ngrams.csv",
            "structural",
        ),
    ]

    tabs = st.tabs([t[0] for t in tab_configs])

    for tab, (label, filename, mode) in zip(tabs, tab_configs):
        with tab:
            csv_path = str(itemset_dir / filename)

            if not show_or_generate(
                data_path=csv_path,
                script_path="compute_frequent_itemsets.py",
                args=["--input", company, "--mode", mode],
                cwd=str(SRC_DIR),
                label=f"Generate {label} data for {company}",
            ):
                continue

            # Row-limit control for large files (co-occurrence can be ~48MB)
            is_cooccurrence = mode == "cooccurrence"
            nrows = None
            if is_cooccurrence:
                nrows = st.slider(
                    "Max rows to load",
                    min_value=100,
                    max_value=50000,
                    value=1000,
                    step=100,
                    key=f"nrows_{company}_{mode}",
                )

            df = load_csv(csv_path, nrows=nrows)
            if df is None or df.empty:
                st.info("No data in this file.")
                continue

            # --- Filters ---
            col_f1, col_f2, col_f3 = st.columns(3)

            # Determine which size column exists
            size_col = "itemset_size" if "itemset_size" in df.columns else "ngram_size"
            support_col = "support"

            with col_f1:
                min_sup = st.slider(
                    "Min support",
                    min_value=0.0,
                    max_value=float(df[support_col].max()),
                    value=0.0,
                    step=0.01,
                    key=f"sup_{company}_{mode}",
                )
            with col_f2:
                size_range = st.slider(
                    f"{'Itemset' if size_col == 'itemset_size' else 'N-gram'} size",
                    min_value=int(df[size_col].min()),
                    max_value=int(df[size_col].max()),
                    value=(int(df[size_col].min()), int(df[size_col].max())),
                    key=f"size_{company}_{mode}",
                )
            with col_f3:
                text_search = st.text_input(
                    "Search options/n-grams",
                    key=f"search_{company}_{mode}",
                )

            mask = (df[support_col] >= min_sup) & (
                df[size_col].between(size_range[0], size_range[1])
            )
            text_col = "options" if "options" in df.columns else "ngram"
            if text_search:
                mask = mask & df[text_col].str.contains(text_search, case=False, na=False)

            df_filtered = df[mask]
            st.caption(f"Showing {len(df_filtered)} of {len(df)} rows")
            st.dataframe(df_filtered, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Clustering page renderers
# ---------------------------------------------------------------------------

CLUSTERING_SCRIPT_DIR = str(SRC_DIR / "clustering")


def _detect_methods(directory: Path, prefix: str, suffix: str) -> list[str]:
    """Scan for available clustering methods from file names like {prefix}_{method}{suffix}."""
    methods = []
    for f in sorted(directory.glob(f"{prefix}_*{suffix}")):
        name = f.stem  # e.g. ecosystems_project_assignments_agglomerative
        method = name.replace(f"{prefix}_", "", 1)
        if method.endswith(suffix.replace(".csv", "").replace(".png", "")):
            method = method[: -len(suffix.replace(".csv", "").replace(".png", ""))]
        methods.append(method)
    return sorted(set(methods)) if methods else ["agglomerative"]


def page_technology_stack_clustering(company: str):
    st.header("Clustering: Technology Stack")
    st.caption("Clusters projects into technology ecosystems based on which technologies they use.")

    cluster_dir = DATA_DIR / company / "clustering" / "technology_stack"
    assignments_pattern = "ecosystems_project_assignments"

    # Detect available methods
    methods = _detect_methods(cluster_dir, assignments_pattern, ".csv")
    method = st.selectbox("Clustering method", methods, key="stack_method")

    assignments_csv = str(cluster_dir / f"ecosystems_project_assignments_{method}.csv")
    summary_csv = str(cluster_dir / f"ecosystems_cluster_summary_{method}.csv")
    tech_matrix_csv = str(cluster_dir / f"ecosystems_tech_matrix_{method}.csv")
    entropy_csv = str(cluster_dir / f"ecosystems_entropy_{method}.csv")
    stability_csv = str(cluster_dir / "ecosystems_stability_agglomerative.csv")

    if not show_or_generate(
        data_path=assignments_csv,
        script_path="cluster_technology_stack.py",
        args=["--input", company, "--method", method],
        cwd=CLUSTERING_SCRIPT_DIR,
        label=f"Generate technology stack clustering for {company}",
    ):
        return

    # --- Project assignments ---
    df_assign = load_csv(assignments_csv)
    if df_assign is not None and not df_assign.empty:
        st.subheader("Project Assignments")
        st.dataframe(df_assign, use_container_width=True, hide_index=True)

        # Ecosystem size bar chart
        eco_counts = df_assign["ecosystem"].value_counts().sort_index()
        fig = px.bar(
            x=eco_counts.index.astype(str),
            y=eco_counts.values,
            labels={"x": "Ecosystem", "y": "# Projects"},
            title="Projects per Ecosystem",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Cluster summary ---
    df_summary = load_csv(summary_csv)
    if df_summary is not None:
        st.subheader("Cluster Summary")
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # --- Entropy ---
    df_entropy = load_csv(entropy_csv)
    if df_entropy is not None:
        st.subheader("Ecosystem Entropy")
        st.dataframe(df_entropy, use_container_width=True, hide_index=True)

    # --- Stability (agglomerative only) ---
    df_stab = load_csv(stability_csv)
    if df_stab is not None and method == "agglomerative":
        st.subheader("Stability (ARI by k)")
        fig_stab = px.line(
            df_stab, x="k", y="median_ari",
            title="Cluster Stability: Median ARI by k",
            markers=True,
        )
        fig_stab.update_layout(height=400)
        st.plotly_chart(fig_stab, use_container_width=True)

    # --- Static images ---
    image_suffixes = ["embedding", "heatmap", "mds", "tsne", "sizes"]
    images = [
        (cluster_dir / f"ecosystems_{s}_{method}.png", s)
        for s in image_suffixes
    ]
    existing_images = [(p, s) for p, s in images if p.exists()]
    if existing_images:
        with st.expander("Static plots"):
            for img_path, label in existing_images:
                st.image(str(img_path), caption=label)


def page_technology_clustering(company: str):
    st.header("Clustering: Technology")
    st.caption("Clusters projects by how they configure a single technology.")

    cluster_dir = DATA_DIR / company / "clustering" / "technologies"

    # Detect available technologies from file names like {tech}_cluster_assignments_{method}.csv
    tech_files = sorted(cluster_dir.glob("*_cluster_assignments_*.csv")) if cluster_dir.exists() else []
    available_techs = sorted({
        f.name.rsplit("_cluster_assignments_", 1)[0]
        for f in tech_files
    })

    if not available_techs and not cluster_dir.exists():
        st.info(f"No clustering data found for {company}.")
        if st.button("Generate clustering for a technology", key="gen_tech_cluster"):
            st.info("Use the CLI: `python cluster_technologies.py --input {company} --technology <tech>`")
        return

    if not available_techs:
        st.info("No clustered technologies found.")
        return

    technology = st.selectbox("Technology", available_techs, key="clust_tech")

    # Detect methods for this technology
    methods = _detect_methods(
        cluster_dir, f"{technology}_cluster_assignments", ".csv"
    )
    method = st.selectbox("Clustering method", methods, key="tech_clust_method")

    assignments_csv = str(cluster_dir / f"{technology}_cluster_assignments_{method}.csv")
    summary_csv = str(cluster_dir / f"{technology}_cluster_summary_{method}.csv")
    config_matrix_csv = str(cluster_dir / f"{technology}_config_matrix_{method}.csv")
    entropy_csv = str(cluster_dir / f"{technology}_cluster_entropy_{method}.csv")
    k_sweep_csv = str(cluster_dir / f"{technology}_k_sweep_{method}.csv")

    # --- Project assignments ---
    df_assign = load_csv(assignments_csv)
    if df_assign is not None and not df_assign.empty:
        st.subheader("Project Assignments")
        st.dataframe(df_assign, use_container_width=True, hide_index=True)

        cluster_col = "cluster"
        cluster_counts = df_assign[cluster_col].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            labels={"x": "Cluster", "y": "# Projects"},
            title=f"Projects per Cluster ({technology})",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Cluster summary ---
    df_summary = load_csv(summary_csv)
    if df_summary is not None:
        st.subheader("Cluster Summary")
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # --- Entropy ---
    df_entropy = load_csv(entropy_csv)
    if df_entropy is not None:
        st.subheader("Cluster Entropy")
        st.dataframe(df_entropy, use_container_width=True, hide_index=True)

    # --- k sweep ---
    df_sweep = load_csv(k_sweep_csv)
    if df_sweep is not None:
        st.subheader("Parameter Sweep")
        y_col = "silhouette_score" if "silhouette_score" in df_sweep.columns else df_sweep.columns[1]
        fig_sweep = px.line(
            df_sweep, x=df_sweep.columns[0], y=y_col,
            title="Silhouette Score by k",
            markers=True,
        )
        fig_sweep.update_layout(height=400)
        st.plotly_chart(fig_sweep, use_container_width=True)

    # --- Static images ---
    images = [
        (cluster_dir / f"{technology}_pca_{method}.png", "PCA"),
        (cluster_dir / f"{technology}_heatmap_{method}.png", "Heatmap"),
    ]
    existing_images = [(p, s) for p, s in images if p.exists()]
    if existing_images:
        with st.expander("Static plots"):
            for img_path, label in existing_images:
                st.image(str(img_path), caption=label)


def page_config_profile_clustering(company: str):
    st.header("Config Profile Clustering")
    st.caption("Clusters projects within an ecosystem by their configuration option-value pairs.")

    cluster_dir = DATA_DIR / company / "clustering" / "technology_stack_config"

    # Detect available ecosystems from file names like cluster_{N}_config_project_clusters.csv
    eco_files = sorted(cluster_dir.glob("cluster_*_config_project_clusters.csv")) if cluster_dir.exists() else []
    available_ecos = sorted({
        int(f.name.split("_")[1])
        for f in eco_files
    })

    if not available_ecos:
        st.info(f"No config profile clustering data found for {company}.")
        return

    ecosystem = st.selectbox(
        "Ecosystem",
        available_ecos,
        format_func=lambda e: f"Ecosystem {e}",
        key="config_eco",
    )

    assignments_csv = str(cluster_dir / f"cluster_{ecosystem}_config_project_clusters.csv")
    summary_csv = str(cluster_dir / f"cluster_{ecosystem}_config_cluster_summary.csv")
    entropy_csv = str(cluster_dir / f"cluster_{ecosystem}_config_entropy.csv")
    stability_csv = str(cluster_dir / f"cluster_{ecosystem}_config_stability.csv")

    # --- Project assignments ---
    df_assign = load_csv(assignments_csv)
    if df_assign is not None and not df_assign.empty:
        st.subheader("Project Assignments")
        st.dataframe(df_assign, use_container_width=True, hide_index=True)

        cluster_col = "config_cluster"
        cluster_counts = df_assign[cluster_col].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            labels={"x": "Config Cluster", "y": "# Projects"},
            title=f"Projects per Config Cluster (Ecosystem {ecosystem})",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # --- Cluster summary ---
    df_summary = load_csv(summary_csv)
    if df_summary is not None:
        st.subheader("Cluster Summary")
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # --- Entropy ---
    df_entropy = load_csv(entropy_csv)
    if df_entropy is not None:
        st.subheader("Config Cluster Entropy")
        st.dataframe(df_entropy, use_container_width=True, hide_index=True)

    # --- Stability ---
    df_stab = load_csv(stability_csv)
    if df_stab is not None:
        st.subheader("Stability (ARI by k)")
        fig_stab = px.line(
            df_stab, x="k", y="median_ari",
            title=f"Cluster Stability: Median ARI by k (Ecosystem {ecosystem})",
            markers=True,
        )
        fig_stab.update_layout(height=400)
        st.plotly_chart(fig_stab, use_container_width=True)

    # --- Combined entropy across all ecosystems ---
    all_entropy_csv = str(cluster_dir / "cluster_all_config_entropy.csv")
    df_all_entropy = load_csv(all_entropy_csv)
    if df_all_entropy is not None:
        with st.expander("Combined entropy across all ecosystems"):
            st.dataframe(df_all_entropy, use_container_width=True, hide_index=True)

    # --- Static embedding image ---
    embedding_png = cluster_dir / f"cluster_{ecosystem}_config_embedding.png"
    if embedding_png.exists():
        with st.expander("Static plots"):
            st.image(str(embedding_png), caption=f"PCA Embedding (Ecosystem {ecosystem})")


# ---------------------------------------------------------------------------
# Social analysis page renderers
# ---------------------------------------------------------------------------

SOCIAL_SCRIPT_DIR = str(SRC_DIR / "social")


def page_participation_rate(company: str):
    st.header("Configuration Participation Rate")

    social_dir = DATA_DIR / company / "social"
    csv_path = str(social_dir / "participation_rate_results.csv")

    if not show_or_generate(
        data_path=csv_path,
        script_path="compute_participation_rate.py",
        args=["--input", company, "--all"],
        cwd=SOCIAL_SCRIPT_DIR,
        label=f"Generate participation rate data for {company}",
    ):
        return

    df = load_csv(csv_path)
    if df is None or df.empty:
        st.info("No participation rate data available.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart: participation rate per project
    df_sorted = df.sort_values("participation_rate", ascending=False)
    fig = px.bar(
        df_sorted,
        x="project_name",
        y="participation_rate",
        title="Configuration Participation Rate per Project",
        labels={"project_name": "Project", "participation_rate": "Participation Rate"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500, yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: total vs active contributors
    if "total_contributors" in df.columns and "active_contributors" in df.columns:
        fig2 = px.scatter(
            df,
            x="total_contributors",
            y="active_contributors",
            hover_data=["project_name"],
            title="Total vs Active Contributors",
            labels={
                "total_contributors": "Total Contributors",
                "active_contributors": "Active Config Contributors",
            },
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)


def page_gini_index(company: str):
    st.header("Gini Index")

    social_dir = DATA_DIR / company / "social"
    csv_path = str(social_dir / "gini_results.csv")

    if not show_or_generate(
        data_path=csv_path,
        script_path="compute_gini_index.py",
        args=["--input", company, "--all"],
        cwd=SOCIAL_SCRIPT_DIR,
        label=f"Generate Gini index data for {company}",
    ):
        return

    df = load_csv(csv_path)
    if df is None or df.empty:
        st.info("No Gini index data available.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Bar chart: gini_all and gini_active side by side
    df_sorted = df.sort_values("gini_all", ascending=False)
    df_melted = df_sorted.melt(
        id_vars="project_name",
        value_vars=["gini_all", "gini_active"],
        var_name="Variant",
        value_name="Gini",
    )
    fig = px.bar(
        df_melted,
        x="project_name",
        y="Gini",
        color="Variant",
        barmode="group",
        title="Gini Coefficient per Project",
        labels={"project_name": "Project", "Gini": "Gini Coefficient"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=500, yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # Scatter: gini_all vs gini_active
    fig2 = px.scatter(
        df,
        x="gini_all",
        y="gini_active",
        hover_data=["project_name"],
        title="Gini (All Contributors) vs Gini (Active Only)",
        labels={"gini_all": "Gini (All)", "gini_active": "Gini (Active)"},
    )
    fig2.update_layout(height=450)
    st.plotly_chart(fig2, use_container_width=True)


def page_commit_share(company: str):
    st.header("Configuration Commit Share")

    social_dir = DATA_DIR / company / "social"
    csv_path = str(social_dir / "contributor_shares.csv")

    if not show_or_generate(
        data_path=csv_path,
        script_path="compute_contributor_shares.py",
        args=["--input", company, "--all"],
        cwd=SOCIAL_SCRIPT_DIR,
        label=f"Generate commit share data for {company}",
    ):
        return

    df = load_csv(csv_path)
    if df is None or df.empty:
        st.info("No commit share data available.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Grouped bar: top1, top3, top5 share per project
    share_cols = [c for c in ["top1_share", "top3_share", "top5_share"] if c in df.columns]
    if share_cols:
        df_sorted = df.sort_values(share_cols[0], ascending=False)
        df_melted = df_sorted.melt(
            id_vars="project_name",
            value_vars=share_cols,
            var_name="Top-K",
            value_name="Share (%)",
        )
        fig = px.bar(
            df_melted,
            x="project_name",
            y="Share (%)",
            color="Top-K",
            barmode="group",
            title="Top-K Contributor Commit Share per Project",
            labels={"project_name": "Project"},
        )
        fig.update_layout(xaxis_tickangle=-45, height=500, yaxis_range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

    # Scatter: num contributors vs total commits
    if "num_config_contributors" in df.columns and "total_commits" in df.columns:
        fig2 = px.scatter(
            df,
            x="num_config_contributors",
            y="total_commits",
            hover_data=["project_name"],
            title="Config Contributors vs Total Config Commits",
            labels={
                "num_config_contributors": "Config Contributors",
                "total_commits": "Total Config Commits",
            },
        )
        fig2.update_layout(height=450)
        st.plotly_chart(fig2, use_container_width=True)


def page_technology_frequency(company: str):
    st.header("Technology Frequency per Contributor")

    social_dir = DATA_DIR / company / "social"
    csv_path = str(social_dir / "contributor_technology_frequency_all.csv")

    if not show_or_generate(
        data_path=csv_path,
        script_path="compute_contributor_technology_frequency.py",
        args=["--input", company, "--all"],
        cwd=SOCIAL_SCRIPT_DIR,
        label=f"Generate technology frequency data for {company}",
    ):
        return

    df = load_csv(csv_path)
    if df is None or df.empty:
        st.info("No technology frequency data available.")
        return

    # Get list of projects in the data
    projects = sorted(df["Project"].unique()) if "Project" in df.columns else []

    if projects:
        selected_project = st.selectbox("Project", projects, key="techfreq_project")
        df_proj = df[df["Project"] == selected_project].drop(columns=["Project"])
    else:
        df_proj = df

    # Drop contributor column for numeric aggregation
    contrib_col = "Contributor" if "Contributor" in df_proj.columns else None
    if contrib_col:
        df_display = df_proj.set_index(contrib_col)
    else:
        df_display = df_proj

    # Drop columns that are all zero
    df_display = df_display.loc[:, (df_display != 0).any(axis=0)]

    st.dataframe(df_proj, use_container_width=True, hide_index=True)

    # Heatmap of contributor x technology
    if not df_display.empty and len(df_display.columns) > 0:
        fig = px.imshow(
            df_display,
            aspect="auto",
            color_continuous_scale="Blues",
            title=f"Contributor–Technology Frequency ({selected_project if projects else company})",
            labels=dict(x="Technology", y="Contributor", color="Touches"),
        )
        fig.update_layout(height=max(400, len(df_display) * 22))
        st.plotly_chart(fig, use_container_width=True)

    # Aggregate: total touches per technology across all contributors
    if contrib_col and contrib_col in df_proj.columns:
        tech_totals = df_proj.drop(columns=[contrib_col]).sum().sort_values(ascending=False)
    else:
        tech_totals = df_proj.sum(numeric_only=True).sort_values(ascending=False)
    tech_totals = tech_totals[tech_totals > 0]

    if not tech_totals.empty:
        df_totals = pd.DataFrame({"Technology": tech_totals.index, "Total Touches": tech_totals.values})
        fig2 = px.bar(
            df_totals,
            x="Technology",
            y="Total Touches",
            title="Total File Touches per Technology (all contributors)",
        )
        fig2.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(page_title="Configuration Space of Open-Source Software Projects", layout="wide")
    st.title("Configuration Space of Open-Source Software Projects")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        company = st.selectbox(
            "Company",
            ALL_COMPANIES + ["All Companies"],
            index=0,
        )

        st.markdown("---")
        page = st.radio("Page", PAGES_TECHNOLOGICAL + PAGES_CLUSTERING + PAGES_SOCIAL)

    # --- Dispatch ---
    per_company_only = {
        "Frequent Itemsets",
        "Clustering: Technology",
        "Clustering: Config Profile",
        "Participation Rate", "Gini Index", "Commit Share",
        "Technology Frequency",
    }
    if company == "All Companies" and page in per_company_only:
        st.info(f"**{page}** is per-company. Please select a specific company.")
    elif page == "Technology Landscape":
        page_technology_landscape(company)
    elif page == "Option Usage":
        page_option_usage(company)
    elif page == "Frequent Itemsets":
        page_frequent_itemsets(company)
    elif page == "Clustering: Technology Stack":
        page_technology_stack_clustering(company)
    elif page == "Cluserting: Technology":
        page_technology_clustering(company)
    elif page == "Cluserting: Config Profile":
        page_config_profile_clustering(company)
    elif page == "Participation Rate":
        page_participation_rate(company)
    elif page == "Gini Index":
        page_gini_index(company)
    elif page == "Commit Share":
        page_commit_share(company)
    elif page == "Technology Frequency":
        page_technology_frequency(company)


if __name__ == "__main__":
    main()
