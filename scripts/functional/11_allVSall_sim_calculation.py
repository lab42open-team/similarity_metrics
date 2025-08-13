#!/usr/bin/python3.7

import os
import pandas as pd
import numpy as np
import gc  # Garbage Collector
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import time
import logging

logging.basicConfig(level=logging.INFO)

# Chunk size for batch processing
CHUNK_SIZE = 10_000

def optimize_dataframe(df):
    """Convert data types to optimize memory usage."""
    df["GO"] = df["GO"].astype("category")
    df["Run"] = df["Run"].astype("category")
    df["Relative_Abundance"] = df["Relative_Abundance"].astype("float32")  # Reduce float size
    return df

def load_data_in_chunks(input_file):
    """Read the large dataset in chunks to optimize memory usage."""
    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE):
        yield optimize_dataframe(chunk)

def compute_similarity(pair):
    # pair (tuple) contains Run names and dataframe
    # Unpack pair
    run1, run2, df = pair
    # Filter DataFrame for Run1
    df_run1 = df[df["Run"] == run1]
    # Filter DataFrame for Run2
    df_run2 = df[df["Run"] == run2]
    # Merge DataFrames on "GO" to align Relative_Abundances for both Runs
    merged_df = pd.merge(df_run1[["GO", "Relative_Abundance"]],
                         df_run2[["GO", "Relative_Abundance"]],
                         on = "GO", 
                         suffixes=("_run1", "_run2"))
     # Check if merged DataFrame is not empty
    if not merged_df.empty:
        # Extract aligned Relative_Abundances 
        run1_run_Relative_Abundances = merged_df["Relative_Abundance_run1"].values  
        run2_run_Relative_Abundances = merged_df["Relative_Abundance_run2"].values
        # Create numpy array and reshape to a 2-dimensional array
        run1_np = np.array(run1_run_Relative_Abundances).reshape(1, -1)
        run2_np = np.array(run2_run_Relative_Abundances).reshape(1, -1)
        # Calculate cosine distance and convert to similarity score
        distance = cosine_distances(run1_np, run2_np)[0][0]
        #similarity_score = 1 - distance
    # If merged DataFrame is empty, assign default similarity score as zero.
    else:
        distance = 1
    return [run1, run2, distance]

def calculate_distances(input_file, output_dir, metric):
    """Compute distances in batches and write to a file incrementally."""
    output_path = os.path.join(output_dir, f"{metric[0]}_distances_batches_{os.path.basename(input_file)}")
    
    # Write header before processing
    with open(output_path, "w") as f:
        f.write("Run1\tRun2\tDistance\n")

    for df in load_data_in_chunks(input_file):
        pairs = [(s1, s2, df) for s1 in df["Run"].unique() for s2 in df["Run"].unique() if s1 < s2]
        logging.info(f"Processing {len(pairs)} Run pairs in this batch...")

        # Compute distance for this batch
        distances = [compute_similarity(pair) for pair in pairs]

        # Convert to DataFrame
        distance_df = pd.DataFrame(distances, columns=["Run1", "Run2", "Distance"])

        # Append results to file (without loading everything into memory)
        distance_df.to_csv(output_path, sep="\t", mode="a", header=False, index=False)

        # Force memory cleanup
        del distance_df, pairs
        gc.collect()

    logging.info(f"Distances saved to: {output_path}")

def main():
    # Adjust input file
    input_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/filtered_cutOff/0.75_GL_0.01_LO/v4.1/tfidf_pruned" 
    output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances"
    start_time = time.time()
    #calculate_distances(input_file, output_dir, metric="euclidean")
    #logging.info("Euclidean similarity scores calculated successfully.")
    calculate_distances(input_file, output_dir, metric="cosine")
    logging.info("Cosine similarity scores calculated successfully.")
    end_time = time.time()
    logging.info("Total execution time = {} sec".format(end_time-start_time))

if __name__ == "__main__":
    main()
