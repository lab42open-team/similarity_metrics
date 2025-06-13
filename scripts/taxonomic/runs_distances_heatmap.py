import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Function to load data and filter by biome
def load_data(input_file, biome_file):
    try:
        # Load the distance data and biome data
        df = pd.read_csv(input_file, sep="\t")
        biome_df = pd.read_csv(biome_file, sep="\t")

        # Clean the biome info column to handle cases like missing or "Unknown" values
        biome_df["biome_info"] = biome_df["biome_info"].apply(
            lambda x: ":".join(x.split(":")[:3]) if pd.notna(x) else "Unknown"
        )

        # Create a dictionary to map sample IDs to biome info
        biome_dict = dict(zip(biome_df["sample_id"], biome_df["biome_info"]))

        # Create a list of unique samples from both Run1 and Run2
        samples = pd.concat([df["Run1"], df["Run2"]]).unique()

        # Initialize a square distance matrix with `np.ones`, size based on the unique samples
        distance_matrix = pd.DataFrame(np.ones((len(samples), len(samples))), index=samples, columns=samples)

        # Ensure the 'Distance' column is numeric
        df["Distance"] = df["Distance"].astype(float)

        # Populate the distance matrix with actual distance values
        for _, row in df.iterrows():
            run1, run2, distance = row["Run1"], row["Run2"], row["Distance"]
            distance_matrix.loc[run1, run2] = distance
            distance_matrix.loc[run2, run1] = distance  # Ensure symmetry

        # Add the biome info to the distance matrix
        distance_matrix["biome_info"] = distance_matrix.index.map(biome_dict)

        # Filter out rows where the biome is "Unknown"
        distance_matrix = distance_matrix[distance_matrix["biome_info"] != "Unknown"]

        # Log the success and matrix shape
        logging.info(f"Distance matrix loaded and filtered. Shape: {distance_matrix.shape}")

        return distance_matrix

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Function to plot the heatmap
def plot_heatmap(distance_matrix, output_dir):
    try:
        # Create a pivot table for the heatmap (distance matrix already in required form)
        pivot_matrix = distance_matrix.drop(columns="biome_info").T  # Drop 'biome_info' for the heatmap
        pivot_matrix = pivot_matrix.T  # Transpose back to have the correct orientation

        # Set up the color palette and define the figure size
        plt.figure(figsize=(10, 8))

        # Create the heatmap, mapping distance values
        sns.heatmap(pivot_matrix, cmap='YlGnBu', annot=True, fmt=".2f", linewidths=0.5)

        # Add title
        plt.title('Runs Distance Heatmap')

        # Show the plot
        plt.savefig(os.path.join(output_dir, "runs_distances_heatmap.png)"))

    except Exception as e:
        logging.error(f"Error plotting the heatmap: {e}")
        raise

def main():
    input_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/filtered_cutOff/0.75_GL_0.01_LO/v4.1/tfidf_pruned/aggregated_lf_v4.1_with_tfidf_tfidf_pruned.tsv"
    biome_file = "/ccmri/similarity_metrics/data/functional/raw_data/biome_info/merged_biome_v4.1.tsv"
    # Load the data
    distance_matrix = load_data(input_file, biome_file)
    # Plot the heatmap
    plot_heatmap(distance_matrix)

if __name__ == "__main__":
    main()
