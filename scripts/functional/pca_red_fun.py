# script name: pca_red_fun.py
# developed by: Nefeli Venetsianou
# description: 
# Apply dimensionality reduction in functional profiles data using PCA and incorporate biome information.
# framework: CCMRI
# last update: 18/03/2024

import os 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
import logging
logging.basicConfig(level=logging.INFO)

def load_and_process_data(input_file, biome_file):
    try:
        # Read abundance data
        df = pd.read_csv(input_file, sep="\t")
        abundance_matrix = df.pivot_table(index='Run', columns='GO', values='Count', fill_value=0)
        # Read biome data
        biome_df = pd.read_csv(biome_file, sep="\t")
        biome_df["biome_info"] = biome_df["biome_info"].apply(lambda x: ":".join(x.split(":")[:3]) if pd.notna(x) else "Unknown")
        biome_dict = dict(zip(biome_df["sample_id"], biome_df["biome_info"]))
        # Add biome info to the abundance matrix
        abundance_matrix["biome_info"] = abundance_matrix.index.map(biome_dict).fillna("Unknown")
        # Log unique biome counts
        biome_counts = abundance_matrix["biome_info"].value_counts()
        logging.info("Unique biomes and their counts:")
        for biome, count in biome_counts.items():
            logging.info("{}: {}".format(biome, count))
        return abundance_matrix
    except Exception as e:
        logging.error("Error in loading and preprocessing data: {}".format(e))
        raise

def normalize_data(abundance_matrix):
    try:
        # Scale the abundance matrix
        scaler = MinMaxScaler()
        scaled_abundance = scaler.fit_transform(abundance_matrix.drop(columns="biome_info"))
        return scaled_abundance
    except Exception as e:
        logging.error("Error in normalizing data: {}".format(e))
        raise

### HIERARCHICAL CLUSTERING ###
def perform_hierarchical_clustering(reduced_features, abundance_matrix, output_dir, input_file, n_pca, max_distance=10):
    try:
        Z = linkage(reduced_features, method="ward")
        dendrogram_labels = abundance_matrix.index  # Use actual sample IDs 
        plt.figure(figsize=(10, 7))
        dendrogram(Z, labels=dendrogram_labels)  # Ensure labels are set to sample names
        plt.title("Dendrogram for Hierarchical Clustering")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        # Color-coding by biome info
        unique_biomes = abundance_matrix["biome_info"].unique()
        colors = sns.color_palette("hsv", len(unique_biomes))
        biome_color_map = dict(zip(unique_biomes, colors))
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        for lbl in xlbls:
            lbl.set_color(biome_color_map[abundance_matrix.loc[lbl.get_text(), "biome_info"]])
        handles = [plt.Line2D([0], [0], color=biome_color_map[biome], lw=4) for biome in unique_biomes] # create list of handles that represent each biome
        plt.legend(handles, unique_biomes, title="Biomes", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        dendrogram_output = os.path.join(output_dir, f"dendrogram_{n_pca}_pca_biome_{os.path.basename(input_file)[:-4]}.png")
        plt.savefig(dendrogram_output)
        logging.info("Dendrogram saved at {}".format(dendrogram_output))
        # Assign cluster labels to each sample point
        hierarchical_labels = fcluster(Z, max_distance, criterion="distance")
        # Convert hierarchical labels to DataFrame to align with original samples
        cluster_df = pd.DataFrame({"Cluster": hierarchical_labels}, index=abundance_matrix.index)
        return hierarchical_labels
    except Exception as e:
        logging.error("Error in hierarchical clustering: {}".format(e))
        raise

### 2D PCA Plot ###
def visualize_clusters_2d(reduced_features, labels, abundance_matrix, input_file, output_dir, n_pca):
    try:
        pca = PCA(n_components=2)
        reduced_2d = pca.fit_transform(reduced_features)
        # Define a set of marker shapes (common matplotlib markers)
        shapes = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', '8', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
        # Ensure that we have enough markers by cycling through them if needed
        unique_biomes = abundance_matrix["biome_info"].unique()
        shape_cycle = itertools.cycle(shapes)
        shape_map = {biome: next(shape_cycle) for biome in unique_biomes}
        plt.figure(figsize=(10, 12))
        # Plot each point with the corresponding shape for its biome
        for i, biome in enumerate(abundance_matrix["biome_info"]):
            plt.scatter(reduced_2d[i, 0], reduced_2d[i, 1], marker=shape_map[biome], label=biome, edgecolor='black')
        # Add a legend with shape representations for each biome
        legend_elements = [Line2D([0], [0], marker=shape, color='w', markerfacecolor='black', markersize=8, label=biome) 
                           for biome, shape in shape_map.items()]
        plt.legend(handles=legend_elements, title="Biomes", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.title("2D Clustering with Biome Information (Shapes)")
        plt.tight_layout()
        cluster_output = os.path.join(output_dir, f"clusters2d_{n_pca}_pca_biome_shapes_{os.path.basename(input_file)[:-4]}.png")
        plt.savefig(cluster_output)
        logging.info("2D cluster visualization saved at {}".format(cluster_output))
    except Exception as e:
        logging.error("Error in 2D Clusters Visualization: {}".format(e))
        raise

def plot_biome_cluster_heatmap(abundance_matrix, hierarchical_labels, output_dir, input_file, n_pca):
    try:
        # Convert hierarchical cluster labels into a DataFrame
        cluster_df = pd.DataFrame({"Cluster": hierarchical_labels}, index=abundance_matrix.index)
        cluster_df["Biome"] = abundance_matrix["biome_info"]

        # Create a contingency table (counts of biome occurrences in each cluster)
        cluster_biome_counts = cluster_df.groupby(["Cluster", "Biome"]).size().unstack(fill_value=0)

        # Normalize counts by the total number of samples (optional)
        cluster_biome_norm = cluster_biome_counts.div(cluster_biome_counts.sum(axis=1), axis=0)

        # Sort clusters for better visualization
        cluster_biome_norm = cluster_biome_norm.sort_index()

        # Plot heatmap
        plt.figure(figsize=(20, 15))
        sns.heatmap(cluster_biome_norm, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
        plt.xlabel("Biome")
        plt.ylabel("Cluster")
        plt.title(f"Biome Distribution Across Clusters ({n_pca} PCA Components)")

        # Save the heatmap
        heatmap_output = os.path.join(output_dir, f"biome_cluster_heatmap_{n_pca}_pca_{os.path.basename(input_file)[:-4]}.png")
        plt.savefig(heatmap_output, bbox_inches="tight")
        logging.info(f"Biome-Cluster heatmap saved at {heatmap_output}")

        plt.show()  # Show the plot
    except Exception as e:
        logging.error(f"Error in plotting heatmap: {e}")
        raise

def main():
    parent_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/final_CCMRI_GO-slim"
    input_file = os.path.join(parent_dir, "f_aggregated_0.75_0.01_lf_v5.0_super_table.tsv")
    biome_dir = "/ccmri/similarity_metrics/data/functional/raw_data/biome_info" 
    biome_file = os.path.join(biome_dir, "merged_biome_v5.0.tsv")
    output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/dimensionality_reduction"
    heatmaps_output_dir = os.path.join(output_dir, "heatmaps")
    
    # STEP 1: LOAD AND PROCESS DATA (with biome info)
    abundance_matrix = load_and_process_data(input_file, biome_file)
    # STEP 2: NORMALIZE DATA
    scaled_abundance = normalize_data(abundance_matrix)

    # STEP 3: APPLY PCA FOR DIMENSIONALITY REDUCTION
    n_pca = 2 # Adjust the number of components
    pca = PCA(n_components=n_pca)
    reduced_features = pca.fit_transform(scaled_abundance)
    logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    # STEP 4: CLUSTERING (with biome info)
    hierarchical_labels = perform_hierarchical_clustering(reduced_features, abundance_matrix, output_dir, input_file, n_pca)
    logging.info("No of Clusters labels from hierarchical clustering = {}".format(len(set(hierarchical_labels))))

    # Calculate ARI
    true_labels = abundance_matrix["biome_info"]
    label_encoder = LabelEncoder()
    true_labels_encoded = label_encoder.fit_transform(true_labels)
    ari = adjusted_rand_score(true_labels_encoded, hierarchical_labels)
    logging.info("ARI for PCA dimensionality reduction: {}".format(ari))

    # STEP 5: VISUALIZATION (with biome info)
    visualize_clusters_2d(reduced_features, hierarchical_labels, abundance_matrix, input_file, output_dir, n_pca)
    
    # STEP 6: Composition Heatmap Plots
    plot_biome_cluster_heatmap(abundance_matrix, hierarchical_labels, heatmaps_output_dir, input_file, n_pca)

if __name__ == "__main__":
    main()
