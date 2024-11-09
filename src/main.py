import tempfile
import shutil
import git
import os
import pandas as pd
from cfgnet.network.nodes import OptionNode, ArtifactNode, ValueNode
from cfgnet.network.network import Network
from cfgnet.network.network_configuration import NetworkConfiguration





def analyze(repo_name: str, repo_url: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory at {temp_dir}")
        print(f"Cloning repository {repo_url} into {temp_dir}")
        repo = git.Repo.clone_from(repo_url, temp_dir)
        temp_repo_path = repo.working_tree_dir

        latest_commit = repo.head.commit

        network_config = NetworkConfiguration(
            project_root_abs=temp_repo_path,
            enable_all_conflicts=True,
            enable_internal_links=True,
            enable_static_blacklist=True,
            system_level=False
        )

        network = Network.init_network(cfg=network_config)

        option_nodes = network.get_nodes(OptionNode)
        
        option_nodes = [node for node in option_nodes if node.prevalue_node]

        concepts = set(node.concept_name for node in network.get_nodes(ArtifactNode))

        print("Latest Commit: ", str(latest_commit))
        print("Len option nodes: ", len(option_nodes))
        print("Concepts: ", concepts)

        for node in network.get_nodes(ValueNode):
            print(node)

def main():


    project_file = "../data/projects.csv"
    df = pd.read_csv(project_file)

    test_repo_name = "cardboard"
    test_repo_url = "https://github.com/CardboardPowered/cardboard"

    analyze(repo_name=test_repo_name, repo_url=test_repo_url)

    return

    for index, row in df.iterrows()[0]:
        print("Index: ", index)
        print("Full name", row["full_name"])

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Created temporary directory at {temp_dir}")
            repo_url = row["url"]  # Replace with the actual repo URL
            print(f"Cloning repository {repo_url} into {temp_dir}")
            repo = git.Repo.clone_from(repo_url, temp_dir)

            temp_repo_path = os.path.join(temp_dir, row["name"])

            # Step 3: Extract the latest commit
            latest_commit = repo.head.commit
            print(f"Latest commit: {latest_commit.hexsha}")
            print(f"Author: {latest_commit.author.name}")
            print(f"Date: {latest_commit.committed_datetime}")
            print(f"Message: {latest_commit.message}")

            # Step 4: Perform analysis on the repository
            analyze_repository(temp_dir)

if __name__ == "__main__":
    main()
