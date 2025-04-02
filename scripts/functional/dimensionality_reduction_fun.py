import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

# Load Data
def load_data(input_file, biome_file):
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

        # Return both abundance data and biome labels
        return abundance_matrix.drop(columns=['biome_info']), abundance_matrix['biome_info']
    
    except Exception as e:
        logging.error("Error in loading and preprocessing data: {}".format(e))
        raise


# Normalize Data
def normalize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data).astype(np.float32), scaler


# Build Autoencoder Model
def build_autoencoder(input_dim, latent_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(latent_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    encoder = Model(input_layer, encoded)  # Extract encoder for feature reduction
    return autoencoder, encoder

# Train Autoencoder
def train_autoencoder(autoencoder, data, epochs=50, batch_size=32):
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

# Apply Clustering & Evaluate
def cluster_and_evaluate(embeddings, true_labels):
    clustering = AgglomerativeClustering(n_clusters=len(set(true_labels)))
    cluster_labels = clustering.fit_predict(embeddings)
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    return cluster_labels, ari_score

# Main Execution
def main():
    file_path = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/final_CCMRI_GO-slim/f_aggregated_0.75_0.01_lf_v5.0_super_table.tsv"
    metadata_path = "/ccmri/similarity_metrics/data/functional/raw_data/biome_info/merged_biome_v5.0.tsv"
    latent_dims = [4, 8, 16, 32]  # Multiple latent space sizes
    
    data, metadata = load_data(file_path, metadata_path)
    normalized_data, scaler = normalize_data(data)
    normalized_data = np.array(normalized_data)  # Ensure it's a NumPy array

    for latent_dim in latent_dims:
        print(f"Training Autoencoder with latent dimension: {latent_dim}")
        autoencoder, encoder = build_autoencoder(normalized_data.shape[1], latent_dim)
        train_autoencoder(autoencoder, normalized_data)
        
        latent_features = encoder.predict(normalized_data)
        cluster_labels, ari_score = cluster_and_evaluate(latent_features, true_labels)
        print(f"Latent Dim: {latent_dim}, ARI Score: {ari_score:.4f}")
        
        # Visualization
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=latent_features[:, 0], y=latent_features[:, 1], hue=true_labels, palette='viridis')
        plt.title(f"Latent Dim {latent_dim} - Clustering Visualization")
        plt.show()

if __name__ == '__main__':
    main()
