import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances, silhouette_score, confusion_matrix, adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)

def load_data(input_file, biome_file):
    try:
        df = pd.read_csv(input_file, sep="\t")
        biome_df = pd.read_csv(biome_file, sep="\t")
        biome_df["biome_info"] = biome_df["biome_info"].apply(lambda x: ":".join(x.split(":")[:3]) if pd.notna(x) else "Unknown")
        biome_dict = dict(zip(biome_df["sample_id"], biome_df["biome_info"]))
        samples = pd.concat([df["Sample1"], df["Sample2"]]).unique()
        distance_matrix = pd.DataFrame(np.ones((len(samples), len(samples))), index=samples, columns=samples)
        for _, row in df.iterrows():
            sample1, sample2, distance = row["Sample1"], row["Sample2"], row["Distance"]
            distance_matrix.loc[sample1, sample2] = distance
            distance_matrix.loc[sample2, sample1] = distance
        distance_matrix["biome_info"] = distance_matrix.index.map(biome_dict).fillna("Unknown")
        logging.info("Loaded distance matrix successfully:\n{}".format(distance_matrix.head()))
        return distance_matrix
    except Exception as e:
        logging.error("Error loading data: {}".format(e))
        raise

def downsample_data(df, sample_size=1000):
    try:
        if len(df) > sample_size:
            sample_idx = random.sample(list(df.index), sample_size)
            downsampled_df = df.loc[sample_idx, :]
            logging.info("Downsampled data to shape: {}".format(downsampled_df.shape))
        else:
            downsampled_df = df
            logging.info("Data is already small, skipping downsampling.")
        return downsampled_df
    except Exception as e:
        logging.error("Error during downsampling: {}".format(e))
        raise
"""
def perform_hierarchical_clustering(df, output_dir, input_file, k_values):
    try:
        for k in k_values:
            Z = linkage(df.drop(columns="biome_info"), method="ward")
            labels = fcluster(Z, t=k, criterion="maxclust")
            df['cluster'] = labels
            
            plt.figure(figsize=(10, 7))
            sample_names = df.index.tolist()
            dendro_labels = ['.' for _ in sample_names]
            dendrogram(Z, labels=dendro_labels, leaf_rotation=90, leaf_font_size=10)
            unique_biomes = df["biome_info"].unique()
            colors = sns.color_palette("hsv", len(unique_biomes))
            biome_color_map = dict(zip(unique_biomes, colors))
            ax = plt.gca()
            xlbls = ax.get_xmajorticklabels()
            for idx, lbl in enumerate(xlbls):
                sample_name = sample_names[idx]
                lbl.set_color(biome_color_map[df.loc[sample_name, "biome_info"]])
            plt.title(f"Dendrogram with Biome Color Coding (k={k})")
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"com_dendrogram_k{k}_{os.path.basename(input_file)}.png")
            plt.savefig(output_path)
            plt.close()

            contingency_table = pd.crosstab(df['biome_info'], df['cluster'])
            plt.figure(figsize=(10, 7))
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
            plt.title(f"Biome vs Cluster (k={k})")
            heatmap_output_path = os.path.join(output_dir, f"com_heatmap_k{k}_{os.path.basename(input_file)}.png")
            plt.savefig(heatmap_output_path)
            plt.close()

            true_labels = df['biome_info']
            pred_labels = df['cluster']
            label_encoder = LabelEncoder()
            true_labels_numeric = label_encoder.fit_transform(true_labels)
            ari_score = adjusted_rand_score(true_labels_numeric, pred_labels)
            logging.info("Adjusted Rand Index for k={}: {}".format(k, ari_score))
    except Exception as e:
        logging.error("Error in hierarchical clustering: {}".format(e))
        raise
"""

def perform_dbscan_clustering(df, output_dir, input_file, metric='braycurtis', eps=0.5, min_samples=5):
    try:
        # Drop the biome info column for clustering
        features = df.drop(columns=["biome_info"]).values

        # Apply DBSCAN with the specified distance metric
        dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = dbscan_model.fit_predict(features)
        df['cluster'] = labels  # Add cluster labels to DataFrame

        # Count the number of clusters (excluding noise points labeled as -1)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logging.info(f"DBSCAN found {num_clusters} clusters (excluding noise).")

        # Visualize clusters with PCA scatter plot
        plot_pca_clusters(df, labels, output_dir, input_file)

        # Create a contingency table (biome vs. cluster)
        contingency_table = pd.crosstab(df['biome_info'], df['cluster'])
        logging.info(f"Contingency table (biome vs. cluster):\n{contingency_table}")

        # Plot the contingency table as a heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('DBSCAN: Sample Counts per Biome and Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Biome')
        heatmap_output_path = os.path.join(output_dir, f"dbscan_biome_cluster_heatmap_{os.path.basename(input_file)[:-4]}.png")
        plt.tight_layout()
        plt.savefig(heatmap_output_path)
        logging.info(f"Heatmap saved to: {heatmap_output_path}")
        plt.close()

        # Calculate and log the Adjusted Rand Index (ARI)
        true_labels = df['biome_info']
        label_encoder = LabelEncoder()
        true_labels_numeric = label_encoder.fit_transform(true_labels)
        ari_score = adjusted_rand_score(true_labels_numeric, labels)
        logging.info(f"Adjusted Rand Index (ARI) for DBSCAN: {ari_score}")

        # Calculate silhouette score if there are more than one cluster
        if num_clusters > 1:
            silhouette = silhouette_score(features, labels, metric=metric)
            logging.info(f"Silhouette Score for DBSCAN: {silhouette}")

        return df

    except Exception as e:
        logging.error(f"Error during DBSCAN clustering: {e}")
        raise

def plot_pca_clusters(df, labels, output_dir, input_file):
    """ Plot PCA scatter plot for visualizing clusters """
    from sklearn.decomposition import PCA

    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.drop(columns=["biome_info", "cluster"]))
    
    plt.figure(figsize=(10, 7))
    unique_clusters = np.unique(labels)
    colors = sns.color_palette('hsv', len(unique_clusters))
    for cluster, color in zip(unique_clusters, colors):
        if cluster == -1:
            color = 'grey'  # Noise points
        cluster_indices = labels == cluster
        plt.scatter(pca_result[cluster_indices, 0], pca_result[cluster_indices, 1], 
                    label=f'Cluster {cluster}', alpha=0.7, color=color, edgecolors='k')
    
    plt.title('DBSCAN Clustering with PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    output_path = os.path.join(output_dir, f"dbscan_pca_{os.path.basename(input_file)[:-4]}.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"PCA scatter plot saved to: {output_path}")

# Modify your main function to include the DBSCAN step
def main():
    input_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics"
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/clustering"
    biome_dir = "/ccmri/similarity_metrics/data/raw_data/studies_samples/biome_info"
    input_file = os.path.join(input_dir, "c_distances_filtered_v5.0_LSU_ge_filtered.tsv")
    biome_file = os.path.join(biome_dir, "study-sample-biome_v5.0_LSU.tsv")
    
    try:
        # Load and preprocess the data
        df = load_data(input_file, biome_file)
        # Downsample the data if necessary
        sample_size = 1000
        downsampled_matrix = downsample_data(df, sample_size=sample_size)
        # Perform DBSCAN clustering
        perform_dbscan_clustering(downsampled_matrix, output_dir, input_file, metric='braycurtis', eps=0.3, min_samples=5)
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
