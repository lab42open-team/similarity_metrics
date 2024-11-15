# script name: dimensionality_reduction.py
# developed by: Nefeli Venetsianou
# description: 
    # Apply dimensionality reduction in data using autoencoders and incorporate biome information.
# framework: CCMRI
# last update: 21/10/2024

import os 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
import itertools
from matplotlib.lines import Line2D
import logging
logging.basicConfig(level=logging.INFO)

def load_and_process_data(input_file, biome_file):
    try:
        # Read abundance data
        df = pd.read_csv(input_file, sep="\t")
        abundance_matrix = df.pivot_table(index='Sample', columns='Taxa', values='Count', fill_value=0)
        # Read biome data
        biome_df = pd.read_csv(biome_file, sep="\t")
        biome_df["biome_info"] = biome_df["biome_info"].apply(lambda x: ":".join(x.split(":")[:2]) if pd.notna(x) else "Unknown")
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

def build_autoencoder(input_shape):
    try:
        input_layer = Input(shape=(input_shape,))
        enc = Dense(64)(input_layer)
        enc = LeakyReLU()(enc)
        enc = Dense(32)(enc)
        enc = LeakyReLU()(enc)
        latent_space = Dense(16, activation="tanh")(enc)
        dec = Dense(32)(latent_space)
        dec = LeakyReLU()(dec)
        dec = Dense(64)(dec)
        dec = LeakyReLU()(dec)
        output_layer = Dense(input_shape, activation="sigmoid")(dec)
        autoencoder = Model(input_layer, output_layer)
        autoencoder.compile(optimizer="adam", loss="mse", metrics=["mse"])
        return autoencoder
    except Exception as e:
        logging.error("Error in building autoencoder: {}".format(e))
        raise

def train_autoencoder(autoencoder, scaled_data, epochs=50, batch_size=32, validation_split=0.2):
    try:
        autoencoder.fit(scaled_data, scaled_data, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    except Exception as e:
        logging.error("Error in training autoencoder: {}".format(e))
        raise

def extract_latent_features(autoencoder, scaled_data):
    try:
        encoder = Model(autoencoder.input, autoencoder.layers[-5].output)
        reduced_features = encoder.predict(scaled_data)
        return reduced_features
    except Exception as e:
        logging.error("Error in extracting features from latent space: {}".format(e))
        raise

def perform_hierarchical_clustering(reduced_features, abundance_matrix, output_dir, input_file, max_distance=10):
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
        dendrogram_output = os.path.join(output_dir, "dendrogram_biome_2b_{}.png".format(os.path.basename(input_file)[:-4]))
        plt.savefig(dendrogram_output)
        logging.info("Dendrogram saved at {}".format(dendrogram_output))
        # Assign cluster labels to each sample point
        hierarchical_labels = fcluster(Z, max_distance, criterion="distance")
        # Convert hierarchical labels to DataFrame to align with original samples
        cluster_df = pd.DataFrame({"Cluster":hierarchical_labels}, index=abundance_matrix.index)
        return hierarchical_labels
    except Exception as e:
        logging.error("Error in hierarchical clustering: {}".format(e))
        raise

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
        cluster_output = os.path.join(output_dir, "clusters_2d_biome_shapes_{}.png".format(os.path.basename(input_file)[:-4]))
        plt.savefig(cluster_output)
        logging.info("2D cluster visualization saved at {}".format(cluster_output))
    except Exception as e:
        logging.error("Error in 2D Clusters Visualization: {}".format(e))
        raise

def main():
    parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/"
    input_file = os.path.join(parent_dir, "v5.0_LSU_ge_filtered.tsv")
    biome_dir = "/ccmri/similarity_metrics/data/raw_data/studies_samples/biome_info" 
    biome_file = os.path.join(biome_dir, "study-sample-biome_v5.0_LSU.tsv")
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/dimensionality_reduction"
    
    # STEP 1: LOAD AND PROCESS DATA (with biome info)
    abundance_matrix = load_and_process_data(input_file, biome_file)

    # STEP 2: NORMALIZE DATA
    scaled_abundance = normalize_data(abundance_matrix)
    
    # STEP 3: BUILD AND TRAIN AUTOENCODER
    autoencoder = build_autoencoder(scaled_abundance.shape[1])
    train_autoencoder(autoencoder, scaled_abundance)
    
    # STEP 4: EXTRACT FEATURES FROM LATENT SPACE
    reduced_features = extract_latent_features(autoencoder, scaled_abundance)
    
    # STEP 5: CLUSTERING (with biome info)
    hierarchical_labels = perform_hierarchical_clustering(reduced_features, abundance_matrix, output_dir, input_file)
    logging.info("No of Clusters labels from hierarchical clustering = {}".format(len(set(hierarchical_labels))))
    
    # STEP 6: VISUALIZATION (with biome info)
    visualize_clusters_2d(reduced_features, hierarchical_labels, abundance_matrix, input_file, output_dir)

if __name__ == "__main__":
    main()
