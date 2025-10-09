import json
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import itertools
import glob
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


FILE_TYPES = {
    "yaml": ["ansible", "ansible playbook", "kubernetes", "docker compose", "github action", "circleci", "elasticsearch", "flutter", "heroku", "spring", "travis", "yaml"],
    "properties": ["alluxio", "spring", "kafka", "gradle", "gradle wrapper", "maven wrapper", "properties"],
    "json": ["angular", "tsconfig", "nodejs", "cypress", "json"],
    "xml": ["maven", "android", "hadoop common", "hadoop hbase", "hadoop hdfs", "mapreduce", "yarn", "xml"],
    "toml": ["cargo", "netlify", "poetry", "toml"],
    "conf": ["mongodb", "nginx", "postgresql", "rabbitmq", "redis", "apache", "conf"],
    "ini": ["mysql", "php", "ini"],
    "cfg": ["zookeeper"],
    "other": ["docker", "django"]
}

OUT_DIR = "../data/results"

class ConfigurationSpaceClusterer:
    def __init__(self):
        self.projects_data = []
        self.scaler = StandardScaler()
        
    def load_project_data(self, file_paths: List[str]):
        """Load configuration data from multiple project JSON files"""
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
                # Extract latest commit data
                latest_commit = project_data.get('latest_commit_data', {})
                if latest_commit:
                    self.projects_data.append({
                        'project_name': project_data['project_name'],
                        'latest_commit': latest_commit
                    })
    
    def extract_features(self) -> pd.DataFrame:
        """Extract features from configuration spaces for clustering"""
        features = []
        
        for project in self.projects_data:
            project_name = project['project_name']
            commit_data = project['latest_commit']['network_data']
            
            # Technology stack features
            concepts = set(commit_data.get('concepts', []))
            
            # Option-value combinations (like the simple script)
            option_value_pairs = []
            for file_data in commit_data.get('config_file_data', []):
                for pair in file_data.get('pairs', []):
                    option = pair.get('option', '')
                    value = pair.get('value', '')
                    if option and value:
                        option_value_pairs.append(f"{option}={value}")
            
            features.append({
                'project_name': project_name,
                'concepts': list(concepts),
                'option_value_pairs': option_value_pairs,
            })
        
        return pd.DataFrame(features)
    
    def extract_technology_features(self, technology_name: str) -> pd.DataFrame:
        """Extract features from configuration spaces for a given technologyfor clustering"""
        features = []
        
        for project in self.projects_data:
            project_name = project['project_name']
            commit_data = project['latest_commit']['network_data']
            
            # Technology stack features
            concepts = set(commit_data.get('concepts', []))
            
            # Option-value combinations (like the simple script)
            option_value_pairs = []
            for file_data in commit_data.get('config_file_data', []):
                if file_data.get('concept') == technology_name:
                    for pair in file_data.get('pairs', []):
                        option = pair.get('option', '')
                        value = pair.get('value', '')
                        if option and value:
                            option_value_pairs.append(f"{option}={value}")
            
            features.append({
                'project_name': project_name,
                'concept': list(concepts),
                'option_value_pairs': option_value_pairs,
            })
        
        return pd.DataFrame(features)

    def cluster_technology_ecosystems(self, technology_name: str, n_clusters: int = 3) -> Dict:
        tech_df = self.extract_technology_features(technology_name)
        if tech_df.empty:
            raise ValueError(f"No relevant config found for technology: {technology_name}")
        return self.cluster_option_values(tech_df, n_clusters, technology_name)
    
    def cluster_technology_stacks(self, features_df: pd.DataFrame, n_clusters: int = 3):
        """Cluster projects based on technology stacks only"""
        
        # Create technology similarity matrix
        all_concepts = set()
        for concepts in features_df['concepts']:
            all_concepts.update(concepts)
        
        tech_matrix = []
        for _, row in features_df.iterrows():
            tech_vector = [1 if concept in row['concepts'] else 0 
                          for concept in sorted(all_concepts)]
            tech_matrix.append(tech_vector)
        
        tech_features = np.array(tech_matrix)
        tech_similarity = cosine_similarity(tech_features)
        
        # K-means clustering on technology vectors
        svd = TruncatedSVD(n_components=2, random_state=42)
        reduced_tech = svd.fit_transform(tech_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(reduced_tech)
                
        # DBSCAN clustering on technology vectors
        distance_matrix = cosine_distances(tech_features)
        dbscan = DBSCAN(eps=0.3, min_samples=2, metric='precomputed')
        dbscan_labels = dbscan.fit_predict(distance_matrix)
        
        project_names = features_df['project_name'].tolist()
        
        return {
            'type': 'technology_stack',
            'features': tech_features,
            'similarity_matrix': tech_similarity,
            'feature_names': sorted(all_concepts),
            'kmeans': {
                'labels': kmeans_labels,
                'projects': project_names,
                'clusters': self._group_by_clusters(project_names, kmeans_labels)
            },
            'dbscan': {
                'labels': dbscan_labels,
                'projects': project_names,
                'clusters': self._group_by_clusters(project_names, dbscan_labels)
            }
        }
    
    def cluster_option_values(self, features_df: pd.DataFrame, n_clusters: int = 3, technology_name: str = None):
        """Cluster projects based on option-value combinations"""
        
        # Create TF-IDF vectors from option-value pairs
        project_texts = []
        for _, row in features_df.iterrows():
            project_texts.append(' '.join(row['option_value_pairs']))
        
        # TF-IDF vectorization
        tfidf = TfidfVectorizer(max_features=200, stop_words=None)
        tfidf_matrix = tfidf.fit_transform(project_texts)
        option_features = tfidf_matrix.toarray()
        option_similarity = cosine_similarity(tfidf_matrix)
        
        # K-means clustering on option-value vectors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(option_features)
        
        # DBSCAN clustering on option-value vectors
        dbscan = DBSCAN(eps=0.7, min_samples=1, metric='cosine')
        dbscan_labels = dbscan.fit_predict(option_features)
        
        project_names = features_df['project_name'].tolist()
        
        return {
            'type': 'option_value',
            'features': option_features,
            'similarity_matrix': option_similarity,
            'feature_names': tfidf.get_feature_names_out(),
            'kmeans': {
                'labels': kmeans_labels,
                'projects': project_names,
                'clusters': self._group_by_clusters(project_names, kmeans_labels)
            },
            'dbscan': {
                'labels': dbscan_labels,
                'projects': project_names,
                'clusters': self._group_by_clusters(project_names, dbscan_labels)
            }
        }
    
    def cluster_combined_features(self, features_df: pd.DataFrame, n_clusters: int = 3):
        """Cluster projects based on combined tech stack and option-value pairs"""
        
        # Prepare tech stack vectors (binary one-hot)
        all_concepts = sorted(set(concept for concepts in features_df['concepts'] for concept in concepts))
        tech_matrix = []
        for _, row in features_df.iterrows():
            tech_vector = [1 if concept in row['concepts'] else 0 for concept in all_concepts]
            tech_matrix.append(tech_vector)
        tech_features = np.array(tech_matrix)

        # Prepare option-value vectors (TF-IDF)
        project_texts = [' '.join(row['option_value_pairs']) for _, row in features_df.iterrows()]
        tfidf = TfidfVectorizer(max_features=200)
        option_features = tfidf.fit_transform(project_texts)

        # Combine both 
        combined_features = hstack([tech_features, option_features])  # sparse format

        # Dimensionality reduction
        svd = TruncatedSVD(n_components=20, random_state=42)
        reduced_features = svd.fit_transform(combined_features)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(reduced_features)

        # DBSCAN clustering
        distance_matrix = cosine_distances(combined_features)
        dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
        dbscan_labels = dbscan.fit_predict(distance_matrix)

        project_names = features_df['project_name'].tolist()

        return {
            'type': 'combined',
            'features': combined_features,
            'similarity_matrix': cosine_similarity(combined_features),
            'feature_names': all_concepts + list(tfidf.get_feature_names_out()),
            'kmeans': {
                'labels': kmeans_labels,
                'projects': project_names,
                'clusters': self._group_by_clusters(project_names, kmeans_labels)
            },
            'dbscan': {
                'labels': dbscan_labels,
                'projects': project_names,
                'clusters': self._group_by_clusters(project_names, dbscan_labels)
            }
        }


    def _group_by_clusters(self, project_names: List[str], labels: np.ndarray) -> Dict:
        """Group projects by cluster labels"""
        clusters = defaultdict(list)
        for project, label in zip(project_names, labels):
            clusters[label].append(project)
        return dict(clusters)
    
    def analyze_technology_overlaps(self, features_df: pd.DataFrame, clustering_results: Dict) -> Dict:
        """Analyze technology overlaps between clusters"""
        
        overlaps = {}
        
        for method in ['kmeans', 'dbscan']:
            method_overlaps = {}
            clusters = clustering_results[method]['clusters']
            
            for cluster_id, projects in clusters.items():
                if cluster_id == -1:  # Skip noise cluster in DBSCAN
                    continue
                    
                # Find common technologies
                cluster_projects = features_df[features_df['project_name'].isin(projects)]
                
                common_concepts = set(cluster_projects.iloc[0]['concepts'])
                for _, project in cluster_projects.iterrows():
                    common_concepts &= set(project['concepts'])
                
                # Count all technologies in cluster
                all_concepts = []
                for _, project in cluster_projects.iterrows():
                    all_concepts.extend(project['concepts'])
                
                concept_counts = Counter(all_concepts)
                
                method_overlaps[cluster_id] = {
                    'projects': projects,
                    'common_technologies': list(common_concepts),
                    'all_technologies': dict(concept_counts),
                    'size': len(projects)
                }
            
            overlaps[method] = method_overlaps
        
        return overlaps
    
    def analyze_option_value_overlaps(self, features_df: pd.DataFrame, clustering_results: Dict) -> Dict:
        """Analyze option-value overlaps between clusters"""
        
        overlaps = {}
        
        for method in ['kmeans', 'dbscan']:
            method_overlaps = {}
            clusters = clustering_results[method]['clusters']
            
            for cluster_id, projects in clusters.items():
                if cluster_id == -1:  # Skip noise cluster in DBSCAN
                    continue
                    
                # Find common option-value pairs
                cluster_projects = features_df[features_df['project_name'].isin(projects)]
                
                all_pairs = []
                for _, project in cluster_projects.iterrows():
                    all_pairs.extend(project['option_value_pairs'])
                
                pair_counts = Counter(all_pairs)
                frequent_pairs = {pair: count for pair, count in pair_counts.items() 
                                if count > len(projects) * 0.5}  # Appears in >50% of projects
                
                method_overlaps[cluster_id] = {
                    'projects': projects,
                    'frequent_option_values': frequent_pairs,
                    'all_option_values': dict(pair_counts.most_common(10)),
                    'size': len(projects)
                }
            
            overlaps[method] = method_overlaps
        
        return overlaps
    
    def analyze_combined_overlaps(self, features_df: pd.DataFrame, clustering_results: Dict) -> Dict:
        """Analyze overlaps of both technologies and option-values in combined clusters"""
        
        overlaps = {}
        
        for method in ['kmeans', 'dbscan']:
            method_overlaps = {}
            clusters = clustering_results[method]['clusters']
            
            for cluster_id, projects in clusters.items():
                if cluster_id == -1:
                    continue  # skip noise
                
                cluster_projects = features_df[features_df['project_name'].isin(projects)]

                # === Technology Overlap ===
                common_concepts = set(cluster_projects.iloc[0]['concepts'])
                all_concepts = []
                for _, project in cluster_projects.iterrows():
                    common_concepts &= set(project['concepts'])
                    all_concepts.extend(project['concepts'])
                concept_counts = Counter(all_concepts)

                # === Option-Value Overlap ===
                all_pairs = []
                for _, project in cluster_projects.iterrows():
                    all_pairs.extend(project['option_value_pairs'])
                pair_counts = Counter(all_pairs)
                frequent_pairs = {pair: count for pair, count in pair_counts.items()
                                if count > len(projects) * 0.5}

                method_overlaps[cluster_id] = {
                    'projects': projects,
                    'common_technologies': list(common_concepts),
                    'all_technologies': dict(concept_counts),
                    'frequent_option_values': frequent_pairs,
                    'all_option_values': dict(pair_counts.most_common(10)),
                    'size': len(projects)
                }

            overlaps[method] = method_overlaps

        return overlaps

    
    def visualize_clusters(self, clustering_results: Dict, title_suffix: str = ""):
        """Visualize clustering results"""
        
        features = clustering_results['features']
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(features)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # K-means visualization
        axes[0].scatter(pca_features[:, 0], pca_features[:, 1], 
                       c=clustering_results['kmeans']['labels'], 
                       cmap='viridis', s=100)
        axes[0].set_title(f'K-means Clustering {title_suffix}')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add project labels
        project_names = clustering_results['kmeans']['projects']
        for i, project in enumerate(project_names):
            axes[0].annotate(project, (pca_features[i, 0], pca_features[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # DBSCAN visualization
        axes[1].scatter(pca_features[:, 0], pca_features[:, 1], 
                       c=clustering_results['dbscan']['labels'], 
                       cmap='viridis', s=100)
        axes[1].set_title(f'DBSCAN Clustering {title_suffix}')
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Add project labels
        for i, project in enumerate(project_names):
            axes[1].annotate(project, (pca_features[i, 0], pca_features[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        fig.show()
        return fig
    
    def generate_technology_report(self, features_df: pd.DataFrame, 
                                 clustering_results: Dict, 
                                 overlaps: Dict) -> str:
        """Generate a technology clustering report"""
        
        report = "Technology Stack Clustering Report\n"
        report += "=" * 50 + "\n\n"
        
        for method in ['kmeans', 'dbscan']:
            report += f"{method.upper()} Technology Clustering Results:\n"
            report += "-" * 40 + "\n"
            
            clusters = clustering_results[method]['clusters']
            method_overlaps = overlaps[method]
            
            for cluster_id, projects in clusters.items():
                if cluster_id == -1:
                    report += f"Noise/Outliers: {projects}\n\n"
                    continue
                
                overlap_info = method_overlaps.get(cluster_id, {})
                
                report += f"Cluster {cluster_id} ({len(projects)} projects):\n"
                report += f"  Projects: {', '.join(projects)}\n"
                report += f"  Common Technologies: {', '.join(overlap_info.get('common_technologies', []))}\n"
                
                all_techs = overlap_info.get('all_technologies', {})
                if all_techs:
                    report += f"  All Technologies: {dict(list(all_techs.items())[:5])}\n"
                report += "\n"
            
            report += "\n"
        
        return report
    
    def generate_option_value_report(self, features_df: pd.DataFrame, 
                                   clustering_results: Dict, 
                                   overlaps: Dict) -> str:
        """Generate an option-value clustering report"""
        
        report = "Option-Value Configuration Clustering Report\n"
        report += "=" * 50 + "\n\n"
        
        for method in ['kmeans', 'dbscan']:
            report += f"{method.upper()} Option-Value Clustering Results:\n"
            report += "-" * 45 + "\n"
            
            clusters = clustering_results[method]['clusters']
            method_overlaps = overlaps[method]
            
            for cluster_id, projects in clusters.items():
                if cluster_id == -1:
                    report += f"Noise/Outliers: {projects}\n\n"
                    continue
                
                overlap_info = method_overlaps.get(cluster_id, {})
                
                report += f"Cluster {cluster_id} ({len(projects)} projects):\n"
                report += f"  Projects: {', '.join(projects)}\n"
                
                frequent_pairs = overlap_info.get('frequent_option_values', {})
                if frequent_pairs:
                    report += f"  Frequent Option-Value Pairs:\n"
                    for pair, count in list(frequent_pairs.items())[:5]:
                        report += f"    - {pair} (appears in {count} projects)\n"
                
                all_pairs = overlap_info.get('all_option_values', {})
                if all_pairs:
                    report += f"  Top Option-Value Pairs:\n"
                    for pair, count in list(all_pairs.items())[:3]:
                        report += f"    - {pair} ({count} times)\n"
                report += "\n"
            
            report += "\n"
        
        return report
    
    def generate_combined_report(self, features_df: pd.DataFrame, clustering_results: Dict, overlaps: Dict) -> str:
        """Generate report for combined technology + option-value clustering"""
        
        report = "Combined Configuration Space Clustering Report\n"
        report += "=" * 60 + "\n\n"
        
        for method in ['kmeans', 'dbscan']:
            report += f"{method.upper()} Combined Clustering Results:\n"
            report += "-" * 50 + "\n"
            
            clusters = clustering_results[method]['clusters']
            method_overlaps = overlaps[method]
            
            for cluster_id, projects in clusters.items():
                if cluster_id == -1:
                    report += f"Noise/Outliers: {projects}\n\n"
                    continue
                
                overlap_info = method_overlaps.get(cluster_id, {})
                report += f"Cluster {cluster_id} ({len(projects)} projects):\n"
                report += f"  Projects: {', '.join(projects)}\n"
                
                if overlap_info.get('common_technologies'):
                    report += f"  Common Technologies: {', '.join(overlap_info['common_technologies'])}\n"
                if overlap_info.get('frequent_option_values'):
                    report += "  Frequent Option-Value Pairs:\n"
                    for pair, count in list(overlap_info['frequent_option_values'].items())[:5]:
                        report += f"    - {pair} (appears in {count} projects)\n"
                report += "\n"

            report += "\n"
        
        return report


# Example usage functions
def cluster_technology_stacks(project_files: List[str], n_clusters: int = 3):
    """Cluster projects based on technology stacks only"""
    
    clusterer = ConfigurationSpaceClusterer()
    clusterer.load_project_data(project_files)
    
    features_df = clusterer.extract_features()
    
    # Technology stack clustering
    tech_results = clusterer.cluster_technology_stacks(features_df, n_clusters)
    tech_overlaps = clusterer.analyze_technology_overlaps(features_df, tech_results)
    
    # Generate visualizations and report
    tech_fig = clusterer.visualize_clusters(tech_results, "- Technology Stacks")
    tech_report = clusterer.generate_technology_report(features_df, tech_results, tech_overlaps)

    # Save figure
    os.makedirs(OUT_DIR, exist_ok=True)
    tech_fig.savefig(os.path.join(OUT_DIR, "technology_stacks_clustering.png"), dpi=200, bbox_inches="tight")
    
    return {
        'features': features_df,
        'clustering_results': tech_results,
        'overlaps': tech_overlaps,
        'visualization': tech_fig,
        'report': tech_report
    }

def cluster_option_values(project_files: List[str], n_clusters: int = 3):
    """Cluster projects based on option-value combinations only"""
    
    clusterer = ConfigurationSpaceClusterer()
    clusterer.load_project_data(project_files)
    
    features_df = clusterer.extract_features()
    
    # Option-value clustering
    option_results = clusterer.cluster_option_values(features_df, n_clusters)
    option_overlaps = clusterer.analyze_option_value_overlaps(features_df, option_results)
    
    # Generate visualizations and report
    option_fig = clusterer.visualize_clusters(option_results, "- Option-Value Combinations")
    option_report = clusterer.generate_option_value_report(features_df, option_results, option_overlaps)

    # Save figure
    os.makedirs(OUT_DIR, exist_ok=True)
    option_fig.savefig(os.path.join(OUT_DIR, "option_values_clustering.png"), dpi=200, bbox_inches="tight")
    
    return {
        'features': features_df,
        'clustering_results': option_results,
        'overlaps': option_overlaps,
        'visualization': option_fig,
        'report': option_report
    }

def cluster_combined_features(project_files: List[str], n_clusters: int = 3):
    clusterer = ConfigurationSpaceClusterer()
    clusterer.load_project_data(project_files)

    features_df = clusterer.extract_features()

    combined_results = clusterer.cluster_combined_features(features_df, n_clusters)
    overlaps = clusterer.analyze_combined_overlaps(features_df, combined_results)
    fig = clusterer.visualize_clusters(combined_results, "- Combined Features")
    report = clusterer.generate_combined_report(features_df, combined_results, overlaps)

    # Save figure
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "combined_features_clustering.png"), dpi=200, bbox_inches="tight")

    return {
        'features': features_df,
        'clustering_results': combined_results,
        'overlaps': overlaps,
        'visualization': fig,
        'report': report
    }


def cluster_technology_ecosystem(project_files: List[str], technology: str, n_clusters: int = 3):
    clusterer = ConfigurationSpaceClusterer()
    clusterer.load_project_data(project_files)

    tech_df = clusterer.extract_technology_features(technology)
    if tech_df.empty:
        raise ValueError(f"No relevant config found for technology: {technology}")

    clustering_result = clusterer.cluster_option_values(tech_df, n_clusters)
    overlaps = clusterer.analyze_option_value_overlaps(tech_df, clustering_result)
    report = clusterer.generate_option_value_report(tech_df, clustering_result, overlaps)
    fig = clusterer.visualize_clusters(clustering_result, f"- Ecosystem: {technology}")

    # Save figure
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"ecosystem_{technology}_clustering.png"), dpi=200, bbox_inches="tight")

    return {
        'features': tech_df,
        'clustering_results': clustering_result,
        'overlaps': overlaps,
        'visualization': fig,
        'report': report
    }


def run_all_clustering_approaches(project_files: List[str], n_clusters: int = 3):
    """Run both technology stack and option-value clustering"""
    
    logger.info("Running Technology Stack Clustering...")
    tech_stack_results = cluster_technology_stacks(project_files, n_clusters)
    
    #logger.info("\nRunning Option-Value Clustering...")
    #option_results = cluster_option_values(project_files, n_clusters)

    #logger.info("\nRunning Combined Clustering...")
    #combined_results = cluster_combined_features(project_files, n_clusters)

    #logger.info("\nRunning Technology Ecosystem Clustering...")
    #tech_ecosystem_results = cluster_technology_ecosystem(project_files, "docker", n_clusters)
    
    return {
        'techn_stack': tech_stack_results,
    #    'option_value': option_results,
    #    'combined': combined_results,
        #'tech_ecosystem': tech_ecosystem_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Limit number of projects to process")
    args = parser.parse_args()
    project_files = glob.glob("../data/projects_last_commit/*.json")

    if args.limit:
        project_files = project_files[:args.limit]

    # Run both clustering approaches
    results = run_all_clustering_approaches(project_files, n_clusters=5)

    print("Technology Stack Clustering:")
    print(results['techn_stack']['report'])

    #print("\nOption-Value Clustering:")
    #print(results['option_value']['report']) 

    #print("\nCombined Clustering:")
    #print(results['combined']['report'])

    #print("\nTechnology Ecosystem Clustering:")
    #print(results['tech_ecosystem']['report'])