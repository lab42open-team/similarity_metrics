

#!/usr/bin/python3.5

# script name: ranking_metrics_performance.py
# developed by: Nefeli Venetsianou
# description: 
    # Rank performance of similarity metrics applied to initial versus noisy data. 
    # Please check "### TO BE ADJUSTED ###" part for input file name changes. 
# framework: CCMRI
# last update: 25/07/2024

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
    euclidean_downsampled_directory = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/euclidean_output/"
    cosine_downsampled_directory = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/cosine_output"
    jensen_shannon_downsampled_directory = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/jensen_shannon_output"

    ### TO BE ADJUSTED ###
    # Define file names to be compared
    euclidean_downsampled_file_name = "euclidean_initial_VS_downsampled_0.9_ratio_d_v5.0_SSU_ge_filtered.tsv"
    cosine_downsampled_file_name = "cosine_initial_VS_downsampled_0.9_ratio_d_v5.0_SSU_ge_filtered.tsv"
    jensen_shannon_downsampled_file_name = "jsd_initial_VS_downsampled_0.25_ratio_d_v4.1_LSU_ge_filtered.tsv"
    ### --- ###
    
    # Define output directories    
    downsampled_ranking_output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/ranking_output/v4.1_LSU"
    
    # Load data - create DataFrames   
    euclidean_downsampled_df = load_data(os.path.join(euclidean_downsampled_directory, euclidean_downsampled_file_name))
    cosine_downsampled_df = load_data(os.path.join(cosine_downsampled_directory, cosine_downsampled_file_name))
    jensen_shannon_downsampled_df = load_data(os.path.join(jensen_shannon_downsampled_directory, jensen_shannon_downsampled_file_name))

    # Apply ranking     
    euclidean_downsampled_match_ranks = rank_df(euclidean_downsampled_df)
    cosine_downsampled_match_ranks = rank_df(cosine_downsampled_df)
    jensen_shannon_downsampled_match_ranks = rank_df(jensen_shannon_downsampled_df)

    # Construct output path and filename 
    euclidean_downsampled_ranked_file_path = os.path.join(downsampled_ranking_output_dir, "ranking_" + euclidean_downsampled_file_name)
    cosine_downsampled_ranked_file_path = os.path.join(downsampled_ranking_output_dir, "ranking_" + cosine_downsampled_file_name)
    jensen_shannon_downsampled_ranked_file_path = os.path.join(downsampled_ranking_output_dir, "ranking_" + jensen_shannon_downsampled_file_name)
    
    # Save output to new .tsv file    
    euclidean_downsampled_match_ranks.to_csv(euclidean_downsampled_ranked_file_path, sep="\t", index=False)
    logging.info("Euclidean Downsampled file ranking output saved to: {}".format(euclidean_downsampled_ranked_file_path))
    cosine_downsampled_match_ranks.to_csv(cosine_downsampled_ranked_file_path, sep="\t", index=False)
    logging.info("Cosine Downsampled file ranking output saved to: {}".format(cosine_downsampled_ranked_file_path))
    jensen_shannon_downsampled_match_ranks.to_csv(jensen_shannon_downsampled_ranked_file_path, sep="\t", index=False)
    logging.info("JSD Downsampled file ranking output saved to: {}".format(jensen_shannon_downsampled_ranked_file_path))
    
if __name__ == "__main__":
    main()