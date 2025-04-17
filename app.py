import streamlit as st
import glob
import json
import pandas as pd

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disables the limit

from src.evolution import plot_option_evolution, plot_technology_evolution, plot_artifact_evolution
from src.technologies import show_options_per_technology, get_options_per_project
from src.change_frequency import show_change_frequency_options
from src.responsibility import get_contributors_stats, plot_contributors_and_files

def get_projects():
    project_files = glob.glob("data/microservice_projects/*.json")
    return [project_file.split("/")[-1].split(".json")[0] for project_file in project_files]


def load_commit_data(project: str):
    with open(f"data/microservice_projects/{project}.json", "r") as src:
        return json.load(src)

def load_option_data_latest(project: str):
    return pd.read_csv(f"data/options/{project}_options_latest.csv")

def load_option_data_internal(project: str):
    return pd.read_csv(f"data/options/{project}_options_internal.csv")


def show_software_evolution(project: str):
    data = load_commit_data(project=project)

    st.subheader("Evolution of Configuration Options")
    fig1 = plot_option_evolution(data)
    st.pyplot(fig1)

    st.subheader("Evolution of Technologies")
    fig2 = plot_technology_evolution(data)
    st.pyplot(fig2)

    st.subheader("Evolution of Configuration Files")
    fig3 = plot_artifact_evolution(data)
    st.pyplot(fig3)

def show_configurability_of_technologies(project: str):
    property_files = glob.glob("data/technology/*.properties")
    df_option = load_option_data_latest(project=project)

    st.subheader("Configurability of Technologies")
    fig4 = show_options_per_technology(technology_files=property_files)
    st.pyplot(fig4)

    st.subheader(f"Relative VS Total Options Set")
    df1 = get_options_per_project(technology_files=property_files, df_option=df_option)
    st.dataframe(df1)

def show_change_frequency(project: str):
    df_option = load_option_data_internal(project=project)
    st.subheader("Change Frequency of Options")
    fig5 = show_change_frequency_options(df_options=df_option)
    st.pyplot(fig5)

def show_configuration_responsibility(project: str):
    project_data = load_commit_data(project=project)
    df_contriburtors, df_changed_files = get_contributors_stats(project_data=project_data)

    st.subheader("Contributors")
    st.dataframe(df_contriburtors)
    st.subheader("Contributors and Changed Files")
    fig6 = plot_contributors_and_files(df_changed_files=df_changed_files)
    st.pyplot(fig6)

st.set_page_config(layout="wide", page_title="Exploring the Configuration Complexity in Open-Source Software Projects")


st.sidebar.title("Studied Software Projects")
project = st.sidebar.selectbox(
    label="Select a project",
    options=[name for name in get_projects()]
)


st.title(f"Configuration Analysis for: {project}", )


with st.expander("📈 Software Evolution", expanded=False):
    show_software_evolution(project=project)

with st.expander("⚙️ Configurability of Technologies", expanded=False):
    show_configurability_of_technologies(project=project)

with st.expander("🔢 Default Values", expanded=False):
    pass

with st.expander("📊 Change Frequency of Options", expanded=False):
    show_change_frequency(project=project)

with st.expander("🔁 Co-Evolutionary Changes", expanded=False):
    pass

with st.expander("🚨 Introduction of Misconfiguration", expanded=False):
    pass

with st.expander("🧑‍💻 Configuration Responsibilities", expanded=False):
    show_configuration_responsibility(project=project)