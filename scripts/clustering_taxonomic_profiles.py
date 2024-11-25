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
# last update: 18/10/2024

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
#from sklearn.cluster import AgglomerativeClustering
#from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import plotly.figure_factory as ff
import plotly.colors as pc
from sklearn.metrics import pairwise_distances
#from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import cophenet
from sklearn.metrics import confusion_matrix
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import logging
logging.basicConfig(level=logging.INFO)

#import sys
#print(sys.version)

def load_data(input_file, biome_file):
    try:
        df = pd.read_csv(input_file, sep="\t")
        print(df)
        # Load biome information
        biome_df = pd.read_csv(biome_file, sep="\t")
        # Preprocess biome info to keep only the first 3 elements 
        biome_df["biome_info"] = biome_df["biome_info"].apply(lambda x: ":".join(x.split(":")[:2]) if pd.notna(x) else "Unknown")
        # Create dictionary to map sample_id with biome_info
        biome_dict = dict(zip(biome_df["sample_id"], biome_df["biome_info"]))
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
        # Add biome info to the distance matrix
        distance_matrix["biome_info"] = distance_matrix.index.map(biome_dict).fillna("Unknown")
        # Count occurrences of each biome
        biome_counts = distance_matrix["biome_info"].value_counts()
        # Print the unique biomes and their counts
        logging.info("Unique biomes and their counts:")
        for biome, count in biome_counts.items():
            logging.info("{}: {}".format(biome, count))
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
            downsampled_df = df.loc[sample_idx, :]
            logging.info("Data downloaded to shape: {}".format(downsampled_df.shape))
        else:
            downsampled_df = df
            logging.info("Data is already small, skipping downsampling.")
        return downsampled_df
    except Exception as e:
        logging.error("Error during downsampling.".format(e))
        raise

def perform_hierarchical_clustering(df, output_dir, input_file, k_values):
    try:
        logging.info("DataFrame: {}".format(df.head()))
        
        for k in k_values:
            logging.info("Performing hierarchical clustering for k={}".format(k))

            # Perform hierarchical clustering, excluding biome info
            Z = linkage(df.drop(columns="biome_info"), method="average")  
            logging.info("Hierarchical clustering completed")
            
            # Extract cluster labels for k clusters
            labels = fcluster(Z, t=k, criterion="maxclust")
            df['cluster'] = labels  # Add cluster labels to DataFrame

            # Plot dendrogram - NON-interactive
            plt.figure(figsize=(10, 7))
            # Create a mapping from sample index to sample names
            sample_names = df.index.tolist()
            # Replace sample names with dots in the dendrogram labels
            dendro_labels = ['.' for _ in sample_names]
            dendrogram(Z, labels=dendro_labels, leaf_rotation=90, leaf_font_size=10)
            # Color dendrogram labels based on biome_info
            unique_biomes = df["biome_info"].unique()
            colors = sns.color_palette("hsv", len(unique_biomes))
            biome_color_map = dict(zip(unique_biomes, colors))
            ax = plt.gca()
            xlbls = ax.get_xmajorticklabels()
            for idx, lbl in enumerate(xlbls):
                # Access the original sample name using the index
                sample_name = sample_names[idx]
                # Set the color based on the biome information of the original sample
                lbl.set_color(biome_color_map[df.loc[sample_name, "biome_info"]])
            plt.title("Hierarchical Clustering Dendrogram with Biome Color Coding (k={})".format(k))
            # Create a legend for biomes
            handles = [plt.Line2D([0], [0], color=biome_color_map[biome], lw=4) for biome in unique_biomes]
            plt.legend(handles, unique_biomes, title="Biomes", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
            output_path = os.path.join(output_dir, "non-interactive_dendrogram_k{}_{}.png".format(k, os.path.basename(input_file)[:-4]))
            plt.savefig(output_path)
            plt.close()  # Close the plot to avoid memory issues
            logging.info("Dendrogram saved to: {}".format(output_path))

            # Create a cross-tabulation table of counts per biome and cluster
            contingency_table = pd.crosstab(df['biome_info'], df['cluster'])
            # Log the contingency table for reference
            logging.info("Contingency table (biome vs. cluster):\n{}".format(contingency_table))
            # Visualize as a heatmap
            plt.figure(figsize=(10, 7))
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
            plt.xlabel('Cluster')
            plt.ylabel('Biome')
            plt.title('Sample Counts per Biome and Cluster')
            # Use tight layout to fit everything nicely
            plt.tight_layout()
            heatmap_output_path = os.path.join(output_dir, "biome_cluster_heatmap_k{}_{}.png".format(k, os.path.basename(input_file)[:-4]))
            plt.savefig(heatmap_output_path)
            logging.info("Heatmap saved to: {}".format(heatmap_output_path))

            # Compute confusion matrix and Adjusted Rand Index (ARI)
            true_labels = df['biome_info']
            pred_labels = df['cluster']
            # Convert true_labels to numeric labels
            label_encoder = LabelEncoder()
            true_labels_numeric = label_encoder.fit_transform(true_labels)
            """
            # Confusion Matrix
            cm = confusion_matrix(true_labels_numeric, pred_labels)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(pred_labels), yticklabels=np.unique(true_labels_numeric))
            plt.xlabel('Predicted Clusters')
            plt.ylabel('True Biomes')
            plt.title('Confusion Matrix: Clustering vs Biomes (k={})'.format(k))
            confusion_matrix_output_path = os.path.join(output_dir, "confusion_matrix_k{}_{}.png".format(k, os.path.basename(input_file)[21:-4]))
            plt.savefig(confusion_matrix_output_path)
            plt.close()  # Close the plot to avoid memory issues
            logging.info("Confusion Matrix saved to: {}".format(confusion_matrix_output_path))
            """
            # Adjusted Rand Index
            ari_score = adjusted_rand_score(true_labels_numeric, pred_labels)
            logging.info("Adjusted Rand Index for k={}: {}".format(k, ari_score))

    except Exception as e:
        logging.error("Error during hierarchical clustering: {}".format(e))
        raise

def compute_cluster_centroids(downsampled_matrix, labels, n_clusters): # Centroid is the mean point of all sampled in a cluster.
    try:
        # Initialize dictionary
        centroids = {}
        for cluster_id in range(1, n_clusters + 1):
            # Extract samples that belong to the current cluster
            cluster_samples = downsampled_matrix[labels == cluster_id]
            # Compute centroid by averaging accross rows 
            centroid = np.mean(cluster_samples, axis=0)
            centroids[cluster_id] = centroid
        return centroids
    except Exception as e:
        logging.error("Error computing cluster centroid {}.".format(e))
        raise

def assign_samples_to_clusters(new_samples_matrix, centroids, metric):
    try:
        # Convert the centroids dictionary to a matrix (n_clusters, n_features)
        centroids_matrix = np.array(list(centroids.values()))
        """
        # Calculate pairwise distances between new samples and centroids 
        #Rows represent the new samples that need to be assigned to clusters.
        #Columns represent the centroids of the clusters.
        #The value in a cell [i, j] is the distance between the new sample i and the centroid of cluster j.
        #Distances calculated based on metric used for the calculation of distances of input file.
        """
        distances = pairwise_distances(new_samples_matrix, centroids_matrix, metric=metric)
        # Assign each new sample to the closest centroid (smallest distance)
        # Argmin function finds the smallest distance (nearest centroid) for each row (axis = 1).
        cluster_assignments = np.argmin(distances, axis=1) + 1  # +1 to adjust for 1-based cluster indexing, since arrays start from 0, while clusters from 1.
        return cluster_assignments
    except Exception as e:
        logging.error("Error assigning new samples to clusters {}.".format(e))
        raise

def load_remaining_samples(original_matrix, downsampled_matrix):
    """
    Extract the remaining samples from the original matrix that were not used in downsampling.
    Ensure that the columns of the remaining samples matrix are the same as the downsampled samples matrix.
    """
    try:
        # Get the indices (sample names) of the remaining samples not in the downsampled data
        remaining_samples = original_matrix.index.difference(downsampled_matrix.index)
        # Extract the remaining samples from the original distance matrix
        # Ensure that the columns are aligned with the downsampled samples
        remaining_samples_matrix = original_matrix.loc[remaining_samples, downsampled_matrix.index]
        logging.info("Remaining samples matrix ({} samples) extracted successfully.".format(len(remaining_samples)))
        return remaining_samples_matrix
    except Exception as e:
        logging.error("Error loading remaining samples: {}".format(e))
        raise

"""
Assess Clustering Results on the downsampled data & Determine optimal cluster number
2 computational approaches: i. elbow method, ii. silhouette analysis 


def plot_elbow_method(df, output_dir, input_file):
    # Initialize list to store sum of squared distances (SSD) 
    # SSD represents how far the data points in a cluster are from the cluster's center (the centroid)
    distortions = []
    # Define range of cluster numbers
    K = range(1, 21)  # Try more clusters if needed
    # Perform hierarchical clustering
    Z = linkage(df, method="average")
    # Calculate cophenetic distances
    #coph_dists = cophenet(Z)  # The cophenetic distance is the height at which two points or clusters are merged in the dendrogram.
    for k in K:
        labels = fcluster(Z, t=k, criterion="maxclust")
        # Compute the sum of squared distances for each cluster by using actual distances
        ssd = 0
        for label in np.unique(labels):
            # Retrieve all points in the current cluster
            cluster_points = df[labels == label]
            # Pairwise distances within the cluster
            pairwise_dists = pairwise_distances(cluster_points)
            # Sum of squared distances within the cluster
            ssd += np.sum(np.square(pairwise_dists))
        distortions.append(ssd)

    # Plot the elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances (SSD)')
    plt.title('Elbow Method for Optimal K')
    output_path = os.path.join(output_dir, "elbow_2_plot_{}.png".format(os.path.basename(input_file)[:-4]))
    plt.savefig(output_path)
    logging.info("Elbow method plot saved to: {}".format(output_path))

def silhouette_analysis(df, output_dir, input_file, metric):
    silhouette_scores = []
    K = range(2, 21)  # At least 2 clusters

    Z = linkage(df, method="average")

    for k in K:
        labels = fcluster(Z, t=k, criterion="maxclust")
        score = silhouette_score(df, labels, metric=metric)
        silhouette_scores.append(score)

    # Plot the silhouette scores
    plt.figure(figsize=(8, 6))
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal K')
    output_path = os.path.join(output_dir, "silhouette_2_plot_{}.png".format(os.path.basename(input_file)[:-4]))
    plt.savefig(output_path)
    plt.show()
    logging.info("Silhouette analysis plot saved to: {}".format(output_path))
"""

def main():
    #input_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics"
    #output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/clustering"
    
     ### METAGENOMICS - RUNS ###
    input_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/normalized_counts/runs_files/metagenomics/similarity_metrics"
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/normalized_counts/runs_files/metagenomics/clustering/2b"
    
    biome_dir = "/ccmri/similarity_metrics/data/raw_data/studies_samples/biome_info"
    #ass_plot_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/clustering/assessment"
    #input_file = os.path.join(input_dir, "c_distances_filtered_v4.1_LSU_ge_filtered.tsv")
    input_file = os.path.join(input_dir, "c_distances_mtgmc_runs_ra_v5.0_LSU_ge_filtered.tsv") # metagenomics input file
    biome_file = os.path.join(biome_dir, "study-sample-biome_v5.0_LSU.tsv") 
    
    try:
        # Step 1: Load and preprocess the data
        df = load_data(input_file, biome_file)
        
        # Step 2: Downsample the data (e.g., select 1,000 samples)
        sample_size = 1000
        downsampled_matrix = downsample_data(df, sample_size=sample_size)
        
        # Step 3: Apply clustering evaluation on the downsampled data matrix
        #plot_elbow_method(downsampled_matrix, ass_plot_output_dir, input_file)
        #silhouette_analysis(downsampled_matrix, ass_plot_output_dir, input_file, metric="cosine")
        
        # Step 4: Perform hierarchical clustering on the downsampled matrix
        k_values = range(5, 41, 5) # k values from 5 to 40 with step 5
        Z = perform_hierarchical_clustering(downsampled_matrix, output_dir, input_file, k_values)
        """
        # Step 5: Compute cluster centroid
        centroids = compute_cluster_centroids(downsampled_matrix.values, labels, n_clusters)
        
        # Step 6: Load new samples 
        remaining_samples_matrix = load_remaining_samples(df, downsampled_matrix)
        
        # Step 7: Assign new samples to the nearest cluster centroid using pairwise distances
        cluster_assignments = assign_samples_to_clusters(remaining_samples_matrix, centroids, metric="cosine")
        
        # Save the cluster assignments
        output_path = os.path.join(output_dir, "new_sample_cluster_assignments.tsv")
        pd.DataFrame({"Sample": remaining_samples_matrix.index, "Cluster": cluster_assignments}).to_csv(output_path, sep='\t', index=False)
        logging.info("Cluster assignments for remaining samples saved to: {}".format(output_path))
        """
    except Exception as e:
        logging.error("Error in main: {}".format(e))
        raise

if __name__ == "__main__":
    main()        