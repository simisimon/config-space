import argparse
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def load_project_files(limit: int | None = None):
    project_files = glob.glob("../data/projects_last_commit/*.json")
    logger.info(f"Found {len(project_files)} project files")
    if limit:
        project_files = project_files[:limit]
        logger.info(f"Using {len(project_files)} project files")
    return project_files


def main(limit: int | None = None):
    project_files = load_project_files(limit)
    # TODO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of projects to process")
    args = parser.parse_args()
    main(args.limit)