#!/usr/bin/python3.5

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
    df["Taxa"] = df["Taxa"].astype("category")
    df["Sample"] = df["Sample"].astype("category")
    df["Count"] = df["Count"].astype("float32")  # Reduce float size
    return df

def load_data_in_chunks(input_file):
    """Read the large dataset in chunks to optimize memory usage."""
    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE):
        yield optimize_dataframe(chunk)

def compute_similarity(pair, metric):
    """Compute distance between two samples."""
    sample1, sample2, df = pair
    df_sample1 = df[df["Sample"] == sample1]
    df_sample2 = df[df["Sample"] == sample2]
    merged_df = pd.merge(df_sample1[["Taxa", "Count"]],
                         df_sample2[["Taxa", "Count"]],
                         on="Taxa", 
                         suffixes=("_sample1", "_sample2"))

    if not merged_df.empty:
        sample1_np = np.array(merged_df["Count_sample1"]).reshape(1, -1)
        sample2_np = np.array(merged_df["Count_sample2"]).reshape(1, -1)

        if metric == "euclidean":
            distance = euclidean_distances(sample1_np, sample2_np)[0][0]
            #similarity_score = 1 / (1 + distance)
        elif metric == "cosine":
            distance = cosine_distances(sample1_np, sample2_np)[0][0]
            #similarity_score = 1 - distance
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    else:
        distance = 1
        #similarity_score = 0

    return [sample1, sample2, distance]

def calculate_distances(input_file, output_dir, metric):
    """Compute distances in batches and write to a file incrementally."""
    output_path = os.path.join(output_dir, f"{metric[0]}_distances_batches_{os.path.basename(input_file)}")
    
    # Write header before processing
    with open(output_path, "w") as f:
        f.write("Sample1\tSample2\tDistance\n")

    # Counter for total sample pairs & unique samples
    total_sample_pairs = 0
    unique_samples = set()
    
    for df in load_data_in_chunks(input_file):
        samples = df["Sample"].unique()
        unique_samples.update(samples)
        pairs = [(s1, s2, df) for s1 in df["Sample"].unique() for s2 in df["Sample"].unique() if s1 < s2]
        total_sample_pairs += len(pairs)
        logging.info(f"Processing {len(pairs)} sample pairs in this batch...")

        # Compute distance for this batch
        distances = [compute_similarity(pair, metric) for pair in pairs]

        # Convert to DataFrame
        distance_df = pd.DataFrame(distances, columns=["Sample1", "Sample2", "Distance"])

        # Append results to file (without loading everything into memory)
        distance_df.to_csv(output_path, sep="\t", mode="a", header=False, index=False)

        # Force memory cleanup
        del distance_df, pairs
        gc.collect()

    logging.info(f"Unique samples: {unique_samples}")
    logging.info(f"Total sample pairs processed: {total_sample_pairs}")
    logging.info(f"Distances saved to: {output_path}")

def main():
    input_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/down_filtered_data/filtered_v4.1_SSU_ge_filtered.tsv"
    output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics"
    start_time = time.time()
    #calculate_distances(input_file, output_dir, metric="euclidean")
    #logging.info("Euclidean similarity scores calculated successfully.")
    calculate_distances(input_file, output_dir, metric="cosine")
    logging.info("Cosine similarity scores calculated successfully.")
    end_time = time.time()
    logging.info("Total execution time = {} sec".format(end_time-start_time))

if __name__ == "__main__":
    main()
