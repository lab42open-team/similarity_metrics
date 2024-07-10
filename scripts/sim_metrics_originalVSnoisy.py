#!/usr/bin/python3.5

# script name: sim_metrics_originalVSnoisy.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate similarity scores between downsampled intial data VS noisy data. 
    # Used for initial VS noisy versions to check similarity performance
    # No input parameters provided, but input/output should be adjusted accordingly. 
        # Please check the session: ### TO BE ADJUSTED ### within the main function. 
# framework: CCMRI
# last update: 02/07/2024

import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
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
            logging.info("Initial Sample Counts Shape: {}".format(df_initial_sample.shape))
            logging.info("Noisy Sample Counts Shape: {}".format(df_noisy_sample.shape))
            # Merge on Taxa to align counts
            merged_df = pd.merge(df_initial_sample[["Taxa", "Count"]], df_noisy_sample[["Taxa", "Count"]], 
                                 on="Taxa", suffixes = ("_initial", "_noisy"))
            # Check if merged DataFrame is not empty
            if not merged_df.empty:
                # Extract aligned counts 
                initial_sample_counts = merged_df["Count_initial"].values  
                noisy_sample_counts = merged_df["Count_noisy"].values
                # DEBUGGING LOGS
                logging.info("Initial Sample Counts: {}".format(initial_sample_counts.shape))
                logging.info("Noisy Sample Counts: {}".format(noisy_sample_counts.shape))
                # Create numpy array and reshape to a 2-dimensional array
                sample1_np = np.array(initial_sample_counts).reshape(1, -1)
                sample2_np = np.array(noisy_sample_counts).reshape(1, -1)
                if metric == "euclidean":
                    euclidean_dist = euclidean_distances(sample1_np, sample2_np)[0][0]
                    similarity_score = 1 / (1 + euclidean_dist)
                elif metric == "cosine":
                    cosine_dist = cosine_distances(sample1_np, sample2_np)[0][0]
                    similarity_score = 1 - cosine_dist
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
    initial_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/downsampled_data/1000_samples/normalized/"
    
    ### TO BE ADJUSTED ###
    # Define parent directory of noisy data [type and level of noise] 
    # Gaussian Noise Parent dir
    gaussian_noisy_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/gaussian_noisy_data/d_1000/5_stdDev"
    # Impulse Noise Parent dir
    impulse_noisy_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/impulse_noisy_data/d_1000/0.9_noiseLevel"
    # Define file names to be compared
    initial_file_name = "ra_d_v4.1_LSU_ge_filtered.tsv"
    gaussian_noisy_file_name = "noisy_5_stdDev_d_v4.1_LSU_ge_filtered.tsv"
    impulse_noisy_file_name = "noisy_0.9_impL_d_v4.1_LSU_ge_filtered.tsv"
    initial_file = os.path.join(initial_parent_directory, initial_file_name)
    ### --- ###
    
    # Load noisy data 
    gaussian_noisy_file = os.path.join(gaussian_noisy_parent_directory, gaussian_noisy_file_name)
    gaussian_df1, gaussian_df2 = load_data(initial_file, gaussian_noisy_file)
    impulse_noisy_file = os.path.join(impulse_noisy_parent_directory, impulse_noisy_file_name)
    impulse_df1, impulse_df2 = load_data(initial_file, impulse_noisy_file)
    # Define Gaussian Output dir
    gaussian_euclidean_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output/"
    gaussian_cosine_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output"
    # Define Impulse Output dir
    impulse_euclidean_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSimpulse_noisy/euclidean_output"
    impulse_cosine_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSimpulse_noisy/cosine_output"
    # Calculate and Save similarities 
    # Gaussian noisy files 
    calculate_similarity_scores(gaussian_df1, gaussian_df2, gaussian_euclidean_output_dir, gaussian_noisy_file, metric="euclidean")
    logging.info("Euclidean similarity scores for initial VS gaussian noisy files calculated successfully.")
    calculate_similarity_scores(gaussian_df1, gaussian_df2, gaussian_cosine_output_dir, gaussian_noisy_file, metric="cosine")
    logging.info("Cosine similarity scores for initial VS gaussian noisy files calculated successfully.")
    # Impulse noisy files
    calculate_similarity_scores(impulse_df1, impulse_df2, impulse_euclidean_output_dir, impulse_noisy_file, metric="euclidean")
    logging.info("Euclidean similarity scores for initial VS impulse noisy files calculated successfully.")
    calculate_similarity_scores(impulse_df1, impulse_df2, impulse_cosine_output_dir, impulse_noisy_file, metric="cosine")
    logging.info("Cosine similarity scores for initial VS impulse noisy files calculated successfully.")

if __name__ == "__main__":
    main()