#!/usr/bin/python3.5

# script name: sim_metrics_fun_originalVSnoisy.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate similarity scores between downsampled intial data VS noisy data. 
    # Used for initial VS noisy versions to check similarity performance
    # No input parameters provided, but input/output should be adjusted accordingly. 
        # Please check the session: ### TO BE ADJUSTED ### within the main function. 
# framework: CCMRI
# last update: 24/01/2025

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
    # Iterate over each run in original and noisy version
    for run1 in df_initial["Run"].unique():
        for run2 in df_noisy["Run"].unique():
            # Create DataFrames with the current run values
            df_initial_run = df_initial[df_initial["Run"] == run1]
            df_noisy_run = df_noisy[df_noisy["Run"] == run2]
            # DEBUGGING LOGS
            #logging.info("Initial run Relative_Abundances Shape: {}".format(df_initial_run.shape))
            #logging.info("Noisy run Relative_Abundances Shape: {}".format(df_noisy_run.shape))
            # Merge on GO to align Relative_Abundances
            merged_df = pd.merge(df_initial_run[["GO", "Relative_Abundance"]], df_noisy_run[["GO", "Relative_Abundance"]], 
                                 on="GO", suffixes = ("_initial", "_noisy"))
            # Check if merged DataFrame is not empty
            if not merged_df.empty:
                # Extract aligned Relative_Abundances 
                initial_run_Relative_Abundances = merged_df["Relative_Abundance_initial"].values  
                noisy_run_Relative_Abundances = merged_df["Relative_Abundance_noisy"].values
                # DEBUGGING LOGS
                #logging.info("Initial run Relative_Abundances: {}".format(initial_run_Relative_Abundances.shape))
                #logging.info("Noisy run Relative_Abundances: {}".format(noisy_run_Relative_Abundances.shape))
                # Create numpy array and reshape to a 2-dimensional array
                run1_np = np.array(initial_run_Relative_Abundances).reshape(1, -1)
                run2_np = np.array(noisy_run_Relative_Abundances).reshape(1, -1)
                if metric == "euclidean":
                    distance = euclidean_distances(run1_np, run2_np)[0][0]
                    #similarity_score = 1 / (1 + euclidean_dist)
                elif metric == "cosine":
                    distance = cosine_distances(run1_np, run2_np)[0][0]
                    #similarity_score = 1 - cosine_dist
                elif metric == "jsd":
                    # Add a small epsilon to avoid zeros
                    epsilon = 1e-10
                    # Normalize Relative_Abundances to probability distributions
                    run1_probs = (run1_np + epsilon) / np.sum(run1_np + epsilon)
                    run2_probs = (run2_np + epsilon) / np.sum(run2_np + epsilon)
                    # Calculate Jensen-Shannon divergence
                    distance = jensenshannon(run1_probs[0], run2_probs[0]) # scipy's jensenshannon function returns the square root of JSD
                    #similarity_score = 1 - distance  # Convert to similarity
                else:
                    raise ValueError("Unsupported metric: {}".format(metric))
            # If merged DataFrame is empty, assign default similarity score as zero.
            else:
                distance = 1
                #similarity_score = 0
            # Store similarity scores to dictionary
            similarity_scores.append([run1, run2, distance])
    # Convert to DataFrame
    similarity_df  = pd.DataFrame(similarity_scores, columns=["Initial_run", "Noisy_run", "Distance"])
    # DEBUGGING LOGS
    logging.info("run {} distance scores:\n{}".format(metric, similarity_df.head()))
    # Group by Noisy run and sort according to similarity score within each group
    grouped_similarity_df = similarity_df.groupby("Noisy_run").apply(lambda x: x.sort_values(by="Distance", ascending=False))
    # Define euclidean output directory and path
    output_path = os.path.join(output_dir, "{}_initial_VS_{}".format(metric[0], os.path.basename(noisy_file_name)))
    # Save DataFrame
    grouped_similarity_df.to_csv(output_path, sep="\t", index=False)
    logging.info("{} output saved to: {}".format(metric.capitalize(), output_path))
    return grouped_similarity_df
    
def main():
    # Define parent directory of intial data [downsampled]
    initial_parent_directory = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/r_1000/normalized_data"   
    # Define noisy directory
    downsampled_noisy_parent_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/downsampling_noise"
    
    ### TO BE ADJUSTED ###
    # Initial files
    initial_file_name = "normalized_r_1000_f_aggregated_0.75_0.01_lf_v4.1_super_table.tsv"
    initial_file = os.path.join(initial_parent_directory, initial_file_name)   
    # Noisy files
    
    downsampled_noise_version_directory = os.path.join(downsampled_noisy_parent_dir, "0.1_ratio")
    downsampled_file_name = "downsampled_0.1_ratio_r_1000_f_aggregated_0.75_0.01_lf_v4.1_super_table.tsv"  
    ### --- ###
    
    # Load noisy data 
    downsampled_noisy_file = os.path.join(downsampled_noise_version_directory, downsampled_file_name)
    downsampled_df1, downsampled_df2 = load_data(initial_file, downsampled_noisy_file)

    # Define downsampled Output dir
    downsampled_euclidean_output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/euclidean_output/"
    downsampled_cosine_output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/cosine_output"
    downsampled_jensen_shannon_output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/jensen_shannon_output"
    
    # Calculate and Save similarities 
    calculate_similarity_scores(downsampled_df1, downsampled_df2, downsampled_euclidean_output_dir, downsampled_noisy_file, metric="euclidean")
    logging.info("Euclidean similarity scores for initial VS downsampled noisy files calculated successfully.")
    calculate_similarity_scores(downsampled_df1, downsampled_df2, downsampled_cosine_output_dir, downsampled_noisy_file, metric="cosine")
    logging.info("Cosine similarity scores for initial VS downsampled noisy files calculated successfully.")
    calculate_similarity_scores(downsampled_df1, downsampled_df2, downsampled_jensen_shannon_output_dir, downsampled_noisy_file, metric="jsd")
    logging.info("Jensen-Shannon Divergence similarity scores for initial VS downsampled noisy files calculated successfully.")

if __name__ == "__main__":
    main()