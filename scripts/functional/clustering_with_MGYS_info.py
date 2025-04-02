#!/usr/bin/python3.5

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import logging
logging.basicConfig(level=logging.INFO)
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio

def load_data(input_file, biome_file):
    try:
        df = pd.read_csv(input_file, sep="\t")
        biome_df = pd.read_csv(biome_file, sep="\t")

        # Create mapping from sample_id to study_id
        sample_to_study = dict(zip(biome_df["sample_id"], biome_df["study_id"]))

        # Replace Run1 and Run2 with their corresponding study_id
        df["Run1"] = df["Run1"].map(sample_to_study).fillna(df["Run1"])
        df["Run2"] = df["Run2"].map(sample_to_study).fillna(df["Run2"])

        # Create biome info column
        biome_df["biome_info"] = biome_df["biome_info"].apply(
            lambda x: ":".join(x.split(":")[:3]) if pd.notna(x) else "Unknown"
        )
        biome_dict = dict(zip(biome_df["study_id"], biome_df["biome_info"]))  # Map study_id to biome_info

        samples = pd.concat([df["Run1"], df["Run2"]]).unique()
        distance_matrix = pd.DataFrame(
            np.ones((len(samples), len(samples))), index=samples, columns=samples
        )

        for _, row in df.iterrows():
            run1, run2, distance = row["Run1"], row["Run2"], row["Distance"]
            distance_matrix.loc[run1, run2] = distance
            distance_matrix.loc[run2, run1] = distance

        # Map biome info based on study_id (now used instead of sample_id)
        distance_matrix["biome_info"] = distance_matrix.index.map(biome_dict).fillna("Unknown")

        # Drop rows where biome_info is "Unknown"
        distance_matrix = distance_matrix[distance_matrix["biome_info"] != "Unknown"]

        logging.info(f"Distance matrix loaded and filtered. Shape: {distance_matrix.shape}")
        return distance_matrix

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def downsample_data(df, sample_size=1000):
    try:
        if len(df) > sample_size:
            sample_idx = random.sample(list(df.index), sample_size)
            downsampled_df = df.loc[sample_idx, :]
            logging.info("Data downsampled to shape: {}".format(downsampled_df.shape))
        else:
            downsampled_df = df
            logging.info("Data is small, skipping downsampling.")
        return downsampled_df
    except Exception as e:
        logging.error("Error during downsampling: {}".format(e))
        raise

def perform_hierarchical_clustering(df, output_dir, input_file, k_values, stage="downsampled"):
    try:
        # Ensure output directory exists
        stage_output_dir = os.path.join(output_dir, stage)
        os.makedirs(stage_output_dir, exist_ok=True)

        # Perform hierarchical clustering, excluding non-numeric columns
        numeric_df = df.drop(columns=["biome_info"], errors="ignore")
        Z = linkage(numeric_df, method="average")  # Perform the hierarchical clustering

        # ---- Plot Static Dendrogram ----
        plt.figure(figsize=(10, 7))
        sample_names = df.index.tolist()
        dendro_labels = ['.' for _ in sample_names]  # Replace sample names with dots
        dendrogram(Z, labels=dendro_labels, leaf_rotation=90, leaf_font_size=10)

        # Color dendrogram labels based on biome_info
        unique_biomes = df["biome_info"].unique()
        colors = sns.color_palette("hsv", len(unique_biomes))
        biome_color_map = dict(zip(unique_biomes, colors))
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        for idx, lbl in enumerate(xlbls):
            sample_name = sample_names[idx]
            lbl.set_color(biome_color_map[df.loc[sample_name, "biome_info"]])

        plt.title("Hierarchical Clustering Dendrogram")
        handles = [plt.Line2D([0], [0], color=biome_color_map[biome], lw=4) for biome in unique_biomes]
        plt.legend(handles, unique_biomes, title="Biomes", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        dendrogram_path = os.path.join(stage_output_dir, f"dendrogram_{os.path.basename(input_file)[:-4]}.png")
        plt.savefig(dendrogram_path)
        plt.close()
        logging.info(f"Dendrogram saved to: {dendrogram_path}")

        for k in k_values:
            logging.info(f"Performing hierarchical clustering for k={k}")

            # Extract cluster labels for k clusters
            labels = fcluster(Z, t=k, criterion="maxclust")
            df['cluster'] = labels  # Add cluster labels to DataFrame

            # ---- Heatmap of Biome vs Cluster ----
            contingency_table = pd.crosstab(df['biome_info'], df['cluster'])
            logging.info(f"Contingency table:\n{contingency_table}")

            plt.figure(figsize=(10, 7))
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
            plt.xlabel('Cluster')
            plt.ylabel('Biome')
            plt.title(f'Run Counts per Biome and Cluster (k={k})')
            plt.tight_layout()
            heatmap_path = os.path.join(stage_output_dir, f"heatmap_k{k}_{os.path.basename(input_file)[:-4]}.png")
            plt.savefig(heatmap_path)
            plt.close()
            logging.info(f"Heatmap saved to: {heatmap_path}")

            # ---- Adjusted Rand Index (ARI) ----
            true_labels = df['biome_info']
            label_encoder = LabelEncoder()
            true_labels_numeric = label_encoder.fit_transform(true_labels)
            ari_score = adjusted_rand_score(true_labels_numeric, labels)
            logging.info(f"Adjusted Rand Index for k={k}: {ari_score}")
    
    except Exception as e:
        logging.error(f"Error during hierarchical clustering: {e}")
        raise
    return df

def compute_cluster_centroids(df):
    try:
        # Drop non-numeric columns before calculatinf centroids
        numeric_df = df.drop(columns=["biome_info"], errors="ignore")
        centroids = numeric_df.groupby("cluster").mean().to_dict(orient="index")
        return centroids
    except Exception as e:
        logging.error("Error computing centroids: {}".format(e))
        raise

def assign_samples_to_clusters(new_samples_matrix, centroids, metric="cosine"):
    try:
        # Convert centroids to a DataFrame
        centroids_matrix = pd.DataFrame.from_dict(centroids, orient="index")
        # Ensure new_samples_matrix is a DataFrame and contains the same columns as centroids
        common_columns = new_samples_matrix.columns.intersection(centroids_matrix.columns)
        new_samples_matrix = new_samples_matrix[common_columns]
        centroids_matrix = centroids_matrix[common_columns]
        # Calculate pairwise distances
        distances = pairwise_distances(new_samples_matrix, centroids_matrix, metric=metric)
        # Return cluster assignment based on the minimum distance
        return np.argmin(distances, axis=1) + 1
    except Exception as e:
        logging.error("Error assigning clusters: {}".format(e))
        raise


def load_remaining_samples(original_matrix, downsampled_matrix):
    try:
        remaining_samples = original_matrix.index.difference(downsampled_matrix.index)
        remaining_samples_matrix = original_matrix.loc[remaining_samples, downsampled_matrix.index]
        logging.info(f"Remaining samples loaded: {len(remaining_samples)}")
        return remaining_samples_matrix
    except Exception as e:
        logging.error("Error loading remaining samples: {}".format(e))
        raise

def main():
    input_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics"
    output_dir = os.path.join(input_dir, "clustering_MGYS")
    input_file = os.path.join(input_dir, "c_distances_batches_normalized_f_aggregated_0.75_0.01_lf_v4.1_super_table.tsv")
    biome_file = "/ccmri/similarity_metrics/data/functional/raw_data/biome_info/merged_biome_v4.1.tsv"
        
    try:
        df = load_data(input_file, biome_file)
        downsampled_matrix = downsample_data(df, sample_size=1000)
        k_values = [2] + list(range(5, 41, 5))
        
        # Perform clustering on downsampled data
        df_clustered = perform_hierarchical_clustering(downsampled_matrix, output_dir, input_file, k_values, stage="downsampled")

        # After assigning remaining samples
        centroids = compute_cluster_centroids(df_clustered)
        remaining_samples_matrix = load_remaining_samples(df, downsampled_matrix)
        cluster_assignments = assign_samples_to_clusters(remaining_samples_matrix, centroids)

        # Save the final cluster assignments for the remaining samples
        df_with_assignments = df.copy()
        df_with_assignments.loc[remaining_samples_matrix.index, 'cluster'] = cluster_assignments
        df_with_assignments = perform_hierarchical_clustering(df_with_assignments, output_dir, input_file, k_values, stage="assigned")

        output_path = os.path.join(output_dir, "new_sample_cluster_assignments.tsv")
        pd.DataFrame({"Sample": remaining_samples_matrix.index, "Cluster": cluster_assignments}).to_csv(output_path, sep="\t", index=False)

        logging.info(f"Cluster assignments saved: {output_path}")

    except Exception as e:
        logging.error("Error in main: {}".format(e))
        raise

if __name__ == "__main__":
    main()
