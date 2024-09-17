#!/usr/bin/python3.5

# script name: clustering_taxonomic_profiles.py
# developed by: Nefeli Venetsianou
# description: 
    # Cluster taxonomic profiles, using similarity scores as distances between points.
        # Hierarchical  
            # Scatter with PCA
            # Heatmap / Clustermap
            # Dendrogram
# framework: CCMRI
# last update: 17/09/2024

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import plotly.figure_factory as ff
import plotly.colors as pc
import logging
logging.basicConfig(level=logging.INFO)

#import sys
#print(sys.version)

def load_data(input_file):
    try:
        df = pd.read_csv(input_file, sep="\t")
        # Get the unique sample names
        samples = pd.concat([df["Sample1"], df["Sample2"]]).unique()
        # Initialize the symmetric matrix
        distance_matrix = pd.DataFrame(np.ones((len(samples), len(samples))), index=samples, columns=samples)
        # Fill the matrix with distances from the pairwise data
        for _, row in df.iterrows():
            sample1 = row["Sample1"]
            sample2 = row["Sample2"]
            distance = row["Distance"]
            distance_matrix.loc[sample1, sample2] = distance
            distance_matrix.loc[sample2, sample1] = distance  # Make it symmetric
        # Verify the matrix
        logging.info("Distance matrix loaded and filled successfully:\n{}.".format(distance_matrix.head()))
        return distance_matrix
    except Exception as e:
        logging.error("Error in loading and preprocessing data: {}".format(e))
        raise

def downsample_data(df, sample_size=1000):
    try:
        if len(df) > sample_size:
            sample_idx = random.sample(list(df.index), sample_size)
            downsampled_df = df.loc[sample_idx, sample_idx]
            logging.info("Data downloaded to shape: {}".format(downsampled_df.shape))
        else:
            downsampled_df = df
            logging.info("Data is already small, skipping downsampling.")
        return downsampled_df
    except Exception as e:
        logging.error("Error during downsampling.".format(e))
        raise

def perform_hierarchical_clustering(df, n_clusters, output_dir, input_file):
    try:
        # Convert the distance matrix to a condensed format for linkage function
        # The condensed matrix is required for hierarchical clustering.
        condensed_distances = df.values[np.triu_indices(len(df), k=1)]
        # Perform hierarchical clustering
        Z = linkage(df, method="average")
        labels = fcluster(Z, t = n_clusters, criterion="maxclust")
        logging.info("Hierarchical clustering completed with {} clusters".format(n_clusters))
        # Create the interactive dendrogram
        fig = ff.create_dendrogram(Z, labels=df.index, orientation='top')
        # Get unique cluster labels
        unique_clusters = np.unique(labels)
        # Generate a list of colors (you can define your own or use a predefined color palette)
        colors = pc.qualitative.Plotly  # Plotly's built-in qualitative color scale
        color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
        # Color the branches of the dendrogram
        for i, d in enumerate(fig["data"]):
            sample_label = df.index[i]
            cluster_label = labels[i]  # Get cluster label for the current sample
            fig["data"][i]["line"]["color"] = color_map[cluster_label]  # Assign color based on cluster
        fig.update_layout(width=1200, height=800, title="Interactive Dendrogram", xaxis_title="Samples", yaxis_title="Cluster Distance")
        # Save the interactive plot to an HTML file
        output_path = os.path.join(output_dir, "interactive_dendrogram_{}_{}.html".format(n_clusters, os.path.basename(input_file)[:-4]))
        fig.write_html(output_path)
        logging.info("Interactive dendrogram saved to: {}".format(output_path))
        """
        # Plot dendrogram
        plt.figure(figsize=(10,7))
        dendrogram(Z, labels=df.index, leaf_rotation=90, leaf_font_size=10)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.tight_layout()
        # Construct output file path
        output_path = os.path.join(output_dir, "nC{}_{}.png".format( n_clusters, os.path.basename(input_file)[:-4]))
        plt.savefig(output_path)
        logging.info("Hierarchical clustering performed and dendrogram plotted successfully and saved to: {}".format(output_path))
        """
        return Z
    except Exception as e:
        logging.error("Error during hierarchical clustering: {}".format(e))
        raise

def main():
    input_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics"
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/clustering"
    input_file = os.path.join(input_dir, "c_distances_filtered_v5.0_LSU_ge_filtered.tsv")

    try:
        # Step 1: Load and preprocess the data
        df = load_data(input_file)
        
        # Step 2: Downsample the data (e.g., select 1,000 samples)
        sample_size = 1000
        downsampled_matrix = downsample_data(df, sample_size=sample_size)
        
        # Step 3: Perform hierarchical clustering on the downsampled matrix
        n_clusters = 5
        Z = perform_hierarchical_clustering(downsampled_matrix, n_clusters, output_dir, input_file)
    
    except Exception as e:
        logging.error("Error in main: {}".format(e))
        raise

if __name__ == "__main__":
    main()        

"""
def load_and_preprocess_data(input_file):
    try: 
        df = pd.read_csv(input_file, sep="\t")
        logging.info("Input file {} loaded correctly, with shape {}.".format(os.path.basename(input_file), df.shape))
        print(df.isnull().sum())
        # Pivot table to get matrix with samples as both rows and columns 
        df_pivot = df.pivot(index="Sample1", columns="Sample2", values="Distance")
        #print(df.head(10))
        #print(df_pivot.head(10))
        logging.info("Data preprocessed to shape {}.\n Head df_pivot: {}".format(df_pivot.shape, df_pivot.head(n=10)))
        df_pivot = df_pivot.combine_first(df_pivot.T)
        return df_pivot
    except Exception as e:
        logging.error("Error in loading and preprocessing data: {}".format(e))
        raise

def downsample_data(df, sample_size=1000):
    try:
        if len(df) > sample_size:
            sample_idx = random.sample(list(df.index), sample_size)
            downsampled_df = df.loc[sample_idx, sample_idx]
            logging.info("Data downloaded to shape: {}".format(downsampled_df.shape))
        else:
            downsampled_df = df
            logging.info("Data is already small, skipping downsampling.")
        return downsampled_df
    except Exception as e:
        logging.error("Error during downsampling.".format(e))
        raise

def perform_hierarchical_clustering(df, n_clusters, output_dir, input_file):
    try:
        # Perform hierarchical clustering
        Z = linkage(df, method="average")
        labels = fcluster(Z, t = n_clusters, criterion="maxclust")
        logging.info("Hierarchical clustering completed with {} clusters".format(n_clusters))
        # Plot dendrogram
        plt.figure(figsize=(10,7))
        dendrogram(Z)
        plt.title("Hierarchical Clustering Dendrogram")
        # Construct output file path
        output_path = os.path.join(output_dir, "nC{}_{}.png".format( n_clusters, os.path.basename(input_file)[:-4]))
        plt.savefig(output_path)
        logging.info("Hierarchical clustering performed and dendrogram plotted successfully and saved to: {}".format(output_path))
        return Z
    except Exception as e:
        logging.error("Error during hierarchical clustering: {}".format(e))
        raise

def main():
    input_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics"
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/clustering"
    input_file = os.path.join(input_dir, "c_distances_filtered_v5.0_LSU_ge_filtered.tsv")

    try:
        # Step 1: Load and preprocess the data
        df = load_and_preprocess_data(input_file)
        
        # Step 2: Downsample the data (e.g., select 1,000 samples)
        sample_size = 1000
        downsampled_matrix = downsample_data(df, sample_size=sample_size)
        
        # Step 3: Perform hierarchical clustering on the downsampled matrix
        n_clusters = 10
        Z = perform_hierarchical_clustering(downsampled_matrix, n_clusters, output_dir, input_file)
    
    except Exception as e:
        logging.error("Error in main: {}".format(e))
        raise

if __name__ == "__main__":
    main()
"""



""" # TRIAL # 
def cluster_data(df, n_clusters):
    try:
        # Apply hierarchical clustering with precomputed affinity matrix (similarities between data points)
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
        # Convert similarity score to distance
        labels = clustering.fit_predict(1 - df)
        logging.info("Hierarchical clustering completed with {} clusters.".format(n_clusters))
        return df, clustering, labels
    except Exception as e: 
        logging.error("Error in clustering: {}".format(e))
        raise

def count_samples_per_cluster(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique_labels, counts))
    
def visualize_clusters_heatmap(df, labels, output_dir, input_file, n_clusters):
    try:
        sorted_idx = np.argsort(labels)
        sorted_df = df.iloc[sorted_idx, sorted_idx]
        plt.figure(figsize=(10,8))
        ax = sns.heatmap(sorted_df, cmap="viridis", cbar=True, annot=False)
        #sns.heatmap(sorted_df, cmap="viridis")
        # Add cluster borders
        cluster_boundaries = np.where(np.diff(labels[sorted_idx]) != 0)[0] + 1
        for boundary in cluster_boundaries:
            ax.axhline(y=boundary - 0.5, color='red', linestyle='--', linewidth=2)
            ax.axvline(x=boundary - 0.5, color='red', linestyle='--', linewidth=2)
        plt.title("Hierarchical Clustering Heatmap")
        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        # Construct output file path
        output_path = os.path.join(output_dir, "nC{}_heatmap_{}.png".format(n_clusters, os.path.basename(input_file)[:-4]))
        plt.savefig(output_path)
        logging.info("Clustering plot successfully saved to: {}".format(output_path))
    except Exception as e:
        logging.error("Error in visualizing clusters: {}".format(e))
        raise

def visualize_clusters_clustermap(df, labels, output_dir, input_file, n_clusters):
    try:
        # Sort the DataFrame based on cluster labels for better visualization
        sorted_idx = np.argsort(labels)
        sorted_df = df.iloc[sorted_idx, sorted_idx]
        # Create a clustermap
        clustermap = sns.clustermap(sorted_df, cmap="viridis", figsize=(15, 12))
        # Add titles and labels directly using the `matplotlib` Axes object
        clustermap.ax_heatmap.set_title('Hierarchical Clustering Heatmap with Dendrogram', fontsize=16)
        clustermap.ax_heatmap.set_xlabel("Sample Index")
        clustermap.ax_heatmap.set_ylabel("Sample Index")
        # Adjust the layout
        plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=90)
        plt.setp(clustermap.ax_heatmap.get_yticklabels(), rotation=0)
        # Save the figure
        output_path = os.path.join(output_dir, "nC{}_clustermap_{}.png".format(n_clusters, os.path.basename(input_file)[:-4]))
        plt.savefig(output_path)
        plt.close()  # Close the plot to avoid display issues
        logging.info("Clustermap plot successfully saved to: {}".format(output_path))
    except Exception as e:
        logging.error("Error in visualizing clustermap: {}".format(e))
        raise

def visualize_clusters_scatter_pca(df, labels, output_dir, input_file, n_clusters):
    try:
        # Dimensionality reduction using PCA
        pca = PCA(n_components=2)
        df_reduced = pca.fit_transform(df)
        # Print the explained variance data 
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        logging.info("Explained variance by component: {}".format(explained_variance)) # variance explained by each component
        logging.info("Cumulative explained variance: {}".format(cumulative_variance)) # variance explained by the first "i" principal components
        # Apply count samples per cluster function
        cluster_counts = count_samples_per_cluster(labels)
        # Plot clusters
        plt.figure(figsize=(10,8))
        u_labels = np.unique(labels)
        for i in u_labels:
            plt.scatter(df_reduced[labels == i, 0], df_reduced[labels == i, 1], label = "Cluster {}: {} Samples".format(i, cluster_counts[i]))
        plt.title("Hierarchical Clustering")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        # Construct output file path
        output_path = os.path.join(output_dir, "nC{}_scatter_pca_plot_{}.png".format(n_clusters, os.path.basename(input_file)[:-4]))
        plt.savefig(output_path)
        logging.info("Clustering plot successfully saved to: {}".format(output_path))
    except Exception as e:
        logging.error("Error in visualizing clusters: {}".format(e))
        raise

def cluster_dendrogram_data(df, n_clusters):
    try:
        # Apply hierarchical clustering with precomputed affinity matrix (similarities between data points)
        # Convert similarity score to distance
        distance_matrix = 1 - df
        # Perform hierarchical clustering
        Z = linkage(distance_matrix, method='average') # based on average distance
        # Assign cluster labels
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        logging.info("Hierarchical clustering completed with {} clusters.".format(n_clusters))
        return df, Z, labels
    except Exception as e: 
        logging.error("Error in clustering: {}".format(e))
        raise
    
def plot_dendrogram(Z, **kwargs):
    try:
        plt.figure(figsize=(10, 8))
        dendrogram(Z, **kwargs)
    except Exception as e:
        logging.error("Error in plotting dendrogram: {}".format(e))
        raise
    
def visualize_dendrogram(Z, output_dir, input_file, n_clusters):
    try:
        plot_dendrogram(Z, truncate_mode='level', p=5)
        plt.title("Hierarchical Clustering Dendrogram")
        # Definition: point = actual observations, index = simple numerical identifiers assigned to data points or clusters
        plt.xlabel("Number of points in node (or index of point if no parenthesis).") 
        # Distance (dissimilarity) between clusters. Height of lines = distance > higher line == less similar clusters that have been merged  
        plt.ylabel("Distance")
        plt.tight_layout()
        # Construct output file path
        output_path = os.path.join(output_dir, "nC{}_dendrogram_{}.png".format(n_clusters, os.path.basename(input_file)[:-4]))
        plt.savefig(output_path)
        logging.info("Dendrogram plot successfully saved to: {}".format(output_path))
    except Exception as e:
        logging.error("Error in visualizing dendrogram: {}".format(e))
        raise  
    
def main():
    input_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics"
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/clustering"
    input_file = os.path.join(input_dir, "c_similarity_scores_filtered_v5.0_LSU_ge_filtered.tsv")
    try:
        df = load_and_preprocess_data(input_file)
        n_clusters = 5
        # Define data to be clustered
        df_clustered, clustering, labels = cluster_data(df, n_clusters)
        # Cluster PCA scatter plot
        visualize_clusters_scatter_pca(df_clustered, labels, output_dir, input_file, n_clusters)
        # Cluster clustermap plot
        visualize_clusters_clustermap(df_clustered, labels, output_dir, input_file, n_clusters)
        # Count samples per cluster 
        cluster_counts = count_samples_per_cluster(labels)
        for cluster, count in cluster_counts.items():
            logging.info("Cluster {} : {} Samples".format(cluster, count))
        # Cluster data in dendrogram
        df_dendro_clustered, Z, dendro_labels = cluster_dendrogram_data(df, n_clusters)
        visualize_dendrogram(Z, output_dir, input_file, n_clusters)
    except Exception as e:
        logging.error("Error in main; {}".format(e))
        raise
    
if __name__ ==  "__main__":
    main()   
"""