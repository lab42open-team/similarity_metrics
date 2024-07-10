#!/usr/bin/python3.5

# script name: ranking_metrics_performance.py
# developed by: Nefeli Venetsianou
# description: 
    # Rank performance of similarity metrics applied to initial versus noisy data. 
    # Please check "### TO BE ADJUSTED ###" part for input file name changes. 
# framework: CCMRI
# last update: 10/07/2024

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def load_data(input_file):
    df = pd.read_csv(input_file, sep="\t")
    logging.info("Data from {} loaded correctly.".format(os.path.basename(input_file)))
    return df

# Function to rank DataFrame and extract position
def rank_df(df):
    # Group by Noisy Sample and rank within each group
    df["Rank"] = df.groupby("Noisy_Sample")["Similarity_Score"].rank("dense", ascending=False)
    # Extract ranking position where "Initial_Sample" matches "Noisy_Sample"
    match_ranks = df[df["Initial_Sample"] == df["Noisy_Sample"]][["Initial_Sample", "Noisy_Sample", "Rank"]]
    return match_ranks
    
def main(): 
    # Define parent directory of data [metrics]
    euclidean_gaussian_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output/"
    cosine_gaussian_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output/"    
    euclidean_impulse_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSimpulse_noisy/euclidean_output/"
    cosine_impulse_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSimpulse_noisy/cosine_output/"

    ### TO BE ADJUSTED ###
    # Define file names to be compared
    euclidean_gaussian_file_name = "e_initial_VS_noisy_5_stdDev_d_v4.1_SSU_ge_filtered.tsv"
    cosine_gaussian_file_name = "c_initial_VS_noisy_5_stdDev_d_v4.1_SSU_ge_filtered.tsv"
    euclidean_impulse_file_name = "e_initial_VS_noisy_0.9_impL_d_v5.0_SSU_ge_filtered.tsv"
    cosine_impulse_file_name = "c_initial_VS_noisy_0.9_impL_d_v4.1_SSU_ge_filtered.tsv"
    
    # Define output directories
    euclidean_gaussian_ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output/ranking_output"
    cosine_gaussian_ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output/ranking_output"   
    euclidean_impulse_ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSimpulse_noisy/euclidean_output/ranking_output"
    cosine_impulse_ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSimpulse_noisy/cosine_output/ranking_output"
    
    # Load data - create DataFrames
    euclidean_gaussian_df = load_data(os.path.join(euclidean_gaussian_parent_directory, euclidean_gaussian_file_name))
    cosine_gaussian_df = load_data(os.path.join(cosine_gaussian_parent_directory, cosine_gaussian_file_name))
    euclidean_impulse_df = load_data(os.path.join(euclidean_impulse_parent_directory, euclidean_impulse_file_name))
    cosine_impulse_df = load_data(os.path.join(cosine_impulse_parent_directory, cosine_impulse_file_name))
    
    # Apply ranking 
    euclidean_gaussian_match_ranks = rank_df(euclidean_gaussian_df)
    cosine_gaussian_match_ranks = rank_df(cosine_gaussian_df)
    euclidean_impulse_match_ranks = rank_df(euclidean_impulse_df)
    cosine_impulse_match_ranks = rank_df(cosine_impulse_df)
    
    # Construct output path and filename
    euclidean_gaussian_ranked_file_path = os.path.join(euclidean_gaussian_ranking_output_dir, "ranking_" + euclidean_gaussian_file_name)
    cosine_gaussian_ranked_file_path = os.path.join(cosine_gaussian_ranking_output_dir, "ranking_" + cosine_gaussian_file_name)
    euclidean_impulse_ranked_file_path = os.path.join(euclidean_impulse_ranking_output_dir, "ranking_" + euclidean_impulse_file_name)
    cosine_impulse_ranked_file_path = os.path.join(cosine_impulse_ranking_output_dir, "ranking_" + cosine_impulse_file_name)
    
    # Save output to new .tsv file
    euclidean_gaussian_match_ranks.to_csv(euclidean_gaussian_ranked_file_path, sep="\t", index=False)
    logging.info("Euclidean Gaussian file ranking output saved to: {}".format(euclidean_gaussian_ranked_file_path))
    cosine_gaussian_match_ranks.to_csv(cosine_gaussian_ranked_file_path, sep="\t", index=False)
    logging.info("Cosine Gaussian file ranking output saved to: {}".format(cosine_gaussian_ranked_file_path))
    euclidean_impulse_match_ranks.to_csv(euclidean_impulse_ranked_file_path, sep="\t", index=False)
    logging.info("Euclidean Impulse file ranking output saved to: {}".format(euclidean_impulse_ranked_file_path))
    cosine_impulse_match_ranks.to_csv(cosine_impulse_ranked_file_path, sep="\t", index=False)
    logging.info("Cosine Impulse file ranking output saved to: {}".format(cosine_impulse_ranked_file_path))
    
if __name__ == "__main__":
    main()
    




