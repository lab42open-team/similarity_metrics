#!/usr/bin/python3.5

# script name: sim_metrics_originalVSnoisy.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate similarity scores between downsampled intial data VS noisy data. 
    # Used for initial VS noisy versions to check similarity performance
    # No input parameters provided, but input/output should be adjusted accordingly. 
        # Please check the session: ### TO BE ADJUSTED ### within the main function. 
# framework: CCMRI
# last update: 13/09/2024

import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import jensenshannon
#from scipy.spatial import distance 
#from scipy.spatial.distance import euclidean, pdist, squareform
import logging
logging.basicConfig(level=logging.INFO)

def load_data(initial_file, noisy_file):
    # Create DataFrames from input files
    df_initial = pd.read_csv(initial_file,sep="\t")
    df_noisy = pd.read_csv(noisy_file, sep="\t")
    # DEBUGGING LOGS 
    logging.info("Initial data of {} loaded correctly: {}".format(initial_file, df_initial.shape))
    logging.info("Noisy data of {} loaded correctly: {}".format(noisy_file, df_noisy.shape))
    return df_initial, df_noisy

def calculate_similarity_scores(df_initial, df_noisy, output_dir, noisy_file_name, metric):
    # Initialize dictionary to store similarity scores
    similarity_scores = []   
    # Iterate over each sample in original and noisy version
    for sample1 in df_initial["Sample"].unique():
        for sample2 in df_noisy["Sample"].unique():
            # Create DataFrames with the current sample values
            df_initial_sample = df_initial[df_initial["Sample"] == sample1]
            df_noisy_sample = df_noisy[df_noisy["Sample"] == sample2]
            # DEBUGGING LOGS
            #logging.info("Initial Sample Counts Shape: {}".format(df_initial_sample.shape))
            #logging.info("Noisy Sample Counts Shape: {}".format(df_noisy_sample.shape))
            # Merge on Taxa to align counts
            merged_df = pd.merge(df_initial_sample[["Taxa", "Count"]], df_noisy_sample[["Taxa", "Count"]], 
                                 on="Taxa", suffixes = ("_initial", "_noisy"))
            # Check if merged DataFrame is not empty
            if not merged_df.empty:
                # Extract aligned counts 
                initial_sample_counts = merged_df["Count_initial"].values  
                noisy_sample_counts = merged_df["Count_noisy"].values
                # DEBUGGING LOGS
                #logging.info("Initial Sample Counts: {}".format(initial_sample_counts.shape))
                #logging.info("Noisy Sample Counts: {}".format(noisy_sample_counts.shape))
                # Create numpy array and reshape to a 2-dimensional array
                sample1_np = np.array(initial_sample_counts).reshape(1, -1)
                sample2_np = np.array(noisy_sample_counts).reshape(1, -1)
                if metric == "euclidean":
                    euclidean_dist = euclidean_distances(sample1_np, sample2_np)[0][0]
                    similarity_score = 1 / (1 + euclidean_dist)
                elif metric == "cosine":
                    cosine_dist = cosine_distances(sample1_np, sample2_np)[0][0]
                    similarity_score = 1 - cosine_dist
                elif metric == "jsd":
                    # Add a small epsilon to avoid zeros
                    epsilon = 1e-10
                    # Normalize counts to probability distributions
                    sample1_probs = (sample1_np + epsilon) / np.sum(sample1_np + epsilon)
                    sample2_probs = (sample2_np + epsilon) / np.sum(sample2_np + epsilon)
                    # Calculate Jensen-Shannon divergence
                    distance = jensenshannon(sample1_probs[0], sample2_probs[0]) # scipy's jensenshannon function returns the square root of JSD
                    similarity_score = 1 - distance  # Convert to similarity
                else:
                    raise ValueError("Unsupported metric: {}".format(metric))
            # If merged DataFrame is empty, assign default similarity score as zero.
            else:
                similarity_score = 0
            # Store similarity scores to dictionary
            similarity_scores.append([sample1, sample2, similarity_score])
    # Convert to DataFrame
    similarity_df  = pd.DataFrame(similarity_scores, columns=["Initial_Sample", "Noisy_Sample", "Similarity_Score"])
    # DEBUGGING LOGS
    logging.info("Sample {} similarity scores:\n{}".format(metric, similarity_df.head()))
    # Group by Noisy Sample and sort according to similarity score within each group
    grouped_similarity_df = similarity_df.groupby("Noisy_Sample").apply(lambda x: x.sort_values(by="Similarity_Score", ascending=False))
    # Define euclidean output directory and path
    output_path = os.path.join(output_dir, "{}_initial_VS_{}".format(metric[0], os.path.basename(noisy_file_name)))
    # Save DataFrame
    grouped_similarity_df.to_csv(output_path, sep="\t", index=False)
    logging.info("{} output saved to: {}".format(metric.capitalize(), output_path))
    return grouped_similarity_df
    
def main():
    # Define parent directory of intial data [downsampled]
    initial_parent_directory = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/d_1000_data/normalized"   
    ### TO BE ADJUSTED ###
    # Initial files
    initial_file_name = "ra_d_v4.1_SSU_ge_filtered.tsv"
    initial_file = os.path.join(initial_parent_directory, initial_file_name)   
    # Noisy files
    downsampled_noisy_parent_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/downsampled_noisy_version/run2"
    downsampled_noise_version_directory = os.path.join(downsampled_noisy_parent_dir, "0.25_ratio")
    downsampled_file_name = "downsampled_0.25_ratio_d_v4.1_SSU_ge_filtered.tsv"  
    ### --- ###
    
    # Load noisy data 
    downsampled_noisy_file = os.path.join(downsampled_noise_version_directory, downsampled_file_name)
    downsampled_df1, downsampled_df2 = load_data(initial_file, downsampled_noisy_file)

    # Define Downsampled Output dir
    downsampled_euclidean_output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output/"
    downsampled_cosine_output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output"
    downsampled_jensen_shannon_output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSdownsampled_noisy/jensen_shannon_output"
    
    # Calculate and Save similarities 
    calculate_similarity_scores(downsampled_df1, downsampled_df2, downsampled_euclidean_output_dir, downsampled_noisy_file, metric="euclidean")
    logging.info("Euclidean similarity scores for initial VS downsampled noisy files calculated successfully.")
    calculate_similarity_scores(downsampled_df1, downsampled_df2, downsampled_cosine_output_dir, downsampled_noisy_file, metric="cosine")
    logging.info("Cosine similarity scores for initial VS downsampled noisy files calculated successfully.")
    calculate_similarity_scores(downsampled_df1, downsampled_df2, downsampled_jensen_shannon_output_dir, downsampled_noisy_file, metric="jsd")
    logging.info("Jensen-Shannon Divergence similarity scores for initial VS downsampled noisy files calculated successfully.")

if __name__ == "__main__":
    main()