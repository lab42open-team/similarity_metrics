#!/usr/bin/python3.5

import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
#from multiprocessing import Pool, cpu_count
import logging
logging.basicConfig(level=logging.INFO)

def load_data(input_file):
    # Read DataFrame from tsv input file
    df = pd.read_csv(input_file, sep="\t")
    unique_samples = df["Sample"].unique()
    logging.info("Number of unique samples: {}".format(len(unique_samples)))
    return df

def compute_similarity(pair, metric):
    # pair (tuple) contains sample names and dataframe
    # Unpack pair
    sample1, sample2, df = pair
    # Filter DataFrame for sample1
    df_sample1 = df[df["Sample"] == sample1]
    # Filter DataFrame for sample2
    df_sample2 = df[df["Sample"] == sample2]
    # Merge DataFrames on "Taxa" to align counts for both samples
    merged_df = pd.merge(df_sample1[["Taxa", "Count"]],
                         df_sample2[["Taxa", "Count"]],
                         on = "Taxa", 
                         suffixes=("_sample1", "_sample2"))
     # Check if merged DataFrame is not empty
    if not merged_df.empty:
        # Extract aligned counts 
        sample1_sample_counts = merged_df["Count_sample1"].values  
        sample2_sample_counts = merged_df["Count_sample2"].values
        # Create numpy array and reshape to a 2-dimensional array
        sample1_np = np.array(sample1_sample_counts).reshape(1, -1)
        sample2_np = np.array(sample2_sample_counts).reshape(1, -1)
        if metric == "euclidean":
            distance = euclidean_distances(sample1_np, sample2_np)[0][0]
            similarity_score = 1 / (1 + distance)
        elif metric == "cosine":
            distance = cosine_distances(sample1_np, sample2_np)[0][0]
            similarity_score = 1 - distance
        else:
            raise ValueError("Unsupported metric: {}".format(metric))
        # Calculate cosine distance and convert to similarity score
        #cosine_dist = cosine_distances(sample1_np, sample2_np)[0][0]
        #similarity_score = 1 - cosine_dist
    # If merged DataFrame is empty, assign default similarity score as zero.
    else:
        similarity_score = 0
    return [sample1, sample2, similarity_score]

def calculate_similarity_scores(df, output_dir, file_name, metric):
    # Create pairs of samples to compare
    pairs = [(sample1, sample2, df) 
             for sample1 in df["Sample"].unique()
             for sample2 in df["Sample"].unique()
             if sample1 < sample2] # ensure no sample will be compared to itself and no duplicates will be created 
    logging.info("Number of pairs generated: {}".format(len(pairs)))
    similarity_scores = [compute_similarity(pair, metric) for pair in pairs] 
    # convert list of similarity scores to DataFrame
    similarity_df = pd.DataFrame(similarity_scores, columns=["Sample1", "Sample2", "Similarity_Score"])
    logging.info("Calculated similarity scores: \n {}".format(similarity_df.head()))
    # Define output path
    output_path = os.path.join(output_dir, "{}_distances_{}".format(metric[0], os.path.basename(file_name)))
    #output_path = os.path.join(output_dir, "c_similarity_scores_{}".format(os.path.basename(file_name)))
    # Save DataFrame to tsv file
    similarity_df.to_csv(output_path, sep="\t", index=False)
    logging.info("Output saved to: {}".format(output_path))
    return similarity_df

def main():
    input_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/down_filtered_data/filtered_v5.0_LSU_ge_filtered.tsv"
    output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/"
    df = load_data(input_file)
    calculate_similarity_scores(df, output_dir, input_file,  metric="euclidean")
    logging.info("Euclidean similarity scores calculated successfully.")
    calculate_similarity_scores(df, output_dir, input_file,  metric="cosine")
    logging.info("Cosine similarity scores calculated successfully.")
    
if __name__ == "__main__":
    main()