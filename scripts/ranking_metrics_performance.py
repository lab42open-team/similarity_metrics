#!/usr/bin/python3.5

# script name: ranking_metrics_performance.py
# developed by: Nefeli Venetsianou
# description: 
    # Rank performance of similarity metrics applied to initial versus noisy data. 
# framework: CCMRI
# last update: 26/06/2024

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

### DEFINE GLOBAL VARIABLES ###
# Define parent directory of data [metrics]
#euclidean_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output"
euclidean_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSimpulse_noisy/euclidean_output"
#cosine_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output"
cosine_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSimpulse_noisy/cosine_output"
# Define file names to be compared
#euclidean_file_name = "e_initial_VS_noisy_5_stdDev_v4.1_LSU_ge_filtered.tsv"
euclidean_file_name = "e_initial_VS_noisy_impulse_v4.1_LSU_ge_filtered.tsv"
#cosine_file_name = "c_initial_VS_noisy_5_stdDev_v4.1_LSU_ge_filtered.tsv"
cosine_file_name = "c_initial_VS_noisy_impulse_v4.1_LSU_ge_filtered.tsv"
# Define output directories
#euclidean_ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output/ranking_output"
euclidean_ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSimpulse_noisy/euclidean_output/ranking_output"
#cosine_ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output/ranking_output"
cosine_ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSimpulse_noisy/cosine_output/ranking_output"
# Read data into DataFrames 
euclidean_df = pd.read_csv(os.path.join(euclidean_parent_directory, euclidean_file_name), sep="\t")
cosine_df = pd.read_csv(os.path.join(cosine_parent_directory,cosine_file_name), sep="\t")

# Function to rank DataFrame and extract position
def rank_df(df):
    # Group by Noisy Sample and rank within each group
    df["Rank"] = df.groupby("Noisy_Sample")["Similarity_Score"].rank("dense", ascending=False)
    # Extract ranking position where "Initial_Sample" matches "Noisy_Sample"
    match_ranks = df[df["Initial_Sample"] == df["Noisy_Sample"]][["Initial_Sample", "Noisy_Sample", "Rank"]]
    return match_ranks
    
def main(): 
    # Apply ranking to both DataFrames
    euclidean_match_ranks = rank_df(euclidean_df)
    cosine_match_ranks = rank_df(cosine_df)
    # Save results to new .tsv file
    euclidean_ranked_file_path = os.path.join(euclidean_ranking_output_dir, "ranking_" + euclidean_file_name)
    cosine_ranked_file_path = os.path.join(cosine_ranking_output_dir, "ranking_" + cosine_file_name)
    euclidean_match_ranks.to_csv(euclidean_ranked_file_path, sep="\t", index=False)
    logging.info("Euclidean ranking output saved to: {}".format(euclidean_ranked_file_path))
    cosine_match_ranks.to_csv(cosine_ranked_file_path, sep="\t", index=False)
    logging.info("Cosine ranking output saved to: {}".format(cosine_ranked_file_path))
    
if __name__ == "__main__":
    main()
    




