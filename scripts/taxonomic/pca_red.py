# script name: pca_red.py
# developed by: Nefeli Venetsianou
# description: 
# Apply dimensionality reduction using PCA and incorporate biome information.
# Clustering 2D, Dendrogram and Heatmap Composition matrix plots are included.
# Principal Components can be skipped as well, be defining n_skip_components, to skip first or first few PC.
# framework: CCMRI
# last update: 09/04/2024

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
        abundance_matrix = df.pivot_table(index='Sample', columns='Taxa', values='Count', fill_value=0)
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
def perform_hierarchical_clustering(reduced_features, abundance_matrix, output_dir, input_file, latent_dim, max_distance=10):
    try:
        Z = linkage(reduced_features, method="ward")
        sample_ids = abundance_matrix.index.tolist()
        plt.figure(figsize=(10, 7))
        # Show dots as labels
        dendro = dendrogram(Z, labels=['.'] * len(sample_ids))
        # Map label index to sample ID
        dendro_order = dendro['leaves']
        ordered_sample_ids = [sample_ids[i] for i in dendro_order]
        # Color labels (dots) based on biome
        unique_biomes = abundance_matrix["biome_info"].unique()
        colors = sns.color_palette("hsv", len(unique_biomes))
        biome_color_map = dict(zip(unique_biomes, colors))
        ax = plt.gca()
        xtick_labels = ax.get_xmajorticklabels()
        for i, lbl in enumerate(xtick_labels):
            sample_id = ordered_sample_ids[i]
            biome = abundance_matrix.loc[sample_id, "biome_info"]
            lbl.set_color(biome_color_map[biome])
        # Add biome legend
        handles = [Line2D([0], [0], color=biome_color_map[biome], lw=4) for biome in unique_biomes]
        plt.legend(handles, unique_biomes, title="Biomes", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.title("Dendrogram for Hierarchical Clustering")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.tight_layout()
        dendrogram_output = os.path.join(output_dir, "dendrogram_5_pca_biome_{}.png".format(os.path.basename(input_file)[:-4]))
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
def visualize_clusters_2d(reduced_features, labels, abundance_matrix, input_file, output_dir):
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
        cluster_output = os.path.join(output_dir, "clusters2d_5_pca_biome_shapes_{}.png".format(os.path.basename(input_file)[:-4]))
        plt.savefig(cluster_output)
        logging.info("2D cluster visualization saved at {}".format(cluster_output))
    except Exception as e:
        logging.error("Error in 2D Clusters Visualization: {}".format(e))
        raise

def main():
    parent_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/"
    input_file = os.path.join(parent_dir, "v5.0_LSU_ge_filtered.tsv")
    biome_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info" 
    biome_file = os.path.join(biome_dir, "study-sample-biome_v5.0_LSU.tsv")
    output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/dimensionality_reduction/PCA/3b/PC1_2_skipped"
    
    # STEP 1: LOAD AND PROCESS DATA (with biome info)
    abundance_matrix = load_and_process_data(input_file, biome_file)
    # STEP 2: NORMALIZE DATA
    scaled_abundance = normalize_data(abundance_matrix)

    # STEP 3: APPLY PCA FOR DIMENSIONALITY REDUCTION
    pca = PCA(n_components=5)
    reduced_features_all = pca.fit_transform(scaled_abundance)
    # Skip the first N components
    n_skip_components = 2
    reduced_features = reduced_features_all[:, n_skip_components:]
    retained_variance = sum(pca.explained_variance_ratio_[n_skip_components:])
    logging.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    logging.info(f"Retained variance after skipping first {n_skip_components} components: {retained_variance:.4f}")

    # STEP 4: CLUSTERING (with biome info)
    hierarchical_labels = perform_hierarchical_clustering(reduced_features, abundance_matrix, output_dir, input_file, latent_dim=None)
    logging.info("No of Clusters labels from hierarchical clustering = {}".format(len(set(hierarchical_labels))))

    # Calculate ARI
    true_labels = abundance_matrix["biome_info"]
    label_encoder = LabelEncoder()
    true_labels_encoded = label_encoder.fit_transform(true_labels)
    ari = adjusted_rand_score(true_labels_encoded, hierarchical_labels)
    logging.info("ARI for PCA dimensionality reduction: {}".format(ari))

    # STEP 5: HEATMAP — Biome vs Cluster Counts
    cluster_df = pd.DataFrame({"Cluster": hierarchical_labels}, index=abundance_matrix.index)
    cluster_df["biome_info"] = abundance_matrix["biome_info"]

    # Cross-tabulate: rows = biomes, columns = clusters
    contingency_table = pd.crosstab(cluster_df["biome_info"], cluster_df["Cluster"])

    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel("Cluster")
    plt.ylabel("Biome")
    plt.title("Sample Counts per Biome and Cluster (PCA)")
    plt.tight_layout()
    heatmap_output_path = os.path.join(output_dir, "biome_cluster_heatmap_pca_5_{}.png".format(os.path.basename(input_file)[:-4]))
    plt.savefig(heatmap_output_path)
    plt.close()
    logging.info("Biome-cluster heatmap saved at {}".format(heatmap_output_path))

    # STEP 6: VISUALIZATION (with biome info)
    visualize_clusters_2d(reduced_features, hierarchical_labels, abundance_matrix, input_file, output_dir)

if __name__ == "__main__":
    main()