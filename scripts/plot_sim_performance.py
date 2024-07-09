#!/usr/bin/python3.5

# script name: plot_sim_performance.py
# developed by: Nefeli Venetsianou
# description: 
    # Plot similarity metrics performance.
    # Please check "### TO BE ADJUSTED ###" part for input file name changes. 
# framework: CCMRI
# last update: 09/07/2024

import os, re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def load_and_combine_data(euclidean_input_file, cosine_input_file):
    df_euclidean = pd.read_csv(euclidean_input_file, sep="\t")
    df_euclidean["Metric"] = "Euclidean"
    df_cosine = pd.read_csv(cosine_input_file, sep="\t")
    df_cosine["Metric"] = "Cosine"
    logging.info("Data from e: {} and c: {} loaded correctly.".format(os.path.basename(euclidean_input_file), os.path.basename(cosine_input_file)))
    # Combine data into one DataFrame
    combined_df = pd.concat([df_euclidean, df_cosine])
    return combined_df

def plot_sim_performance(combined_df, output_file, noise_level):
    plt.figure(figsize=(12,6))
    sns.boxplot(x="Metric", y="Rank", data=combined_df)
    plt.title("Similarity Metrics Performance (Noise Level: {})".format(noise_level))
    plt.xlabel("Metric")
    plt.ylabel("Rank")
    plt.savefig(output_file)
    logging.info("Plot saved successfully to: {}".format(output_file))

def extract_pattern(filename):
    pattern_match = re.search(r"_noisy_(.*)\.tsv", filename)
    noise_level_match = re.search(r"_noisy_(.*?)_d", filename)
    pattern = pattern_match.group(1) if pattern_match else ""
    noise_level = noise_level_match.group(1) if noise_level_match else ""
    return pattern, noise_level

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Created directory: {}".format(directory))
    else:
        logging.info("Directory already exists.")

def main():
    ### TO BE ADJUSTED ###
    # Define output directories
    gaussian_ranking_parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/ranking_output/v5.0_LSU"
    impulse_ranking_parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSimpulse_noisy/ranking_output/v5.0_LSU"
        
    # Define file names to be compared
    euclidean_gaussian_file = os.path.join(gaussian_ranking_parent_dir, "ranking_e_initial_VS_noisy_5_stdDev_d_v5.0_LSU_ge_filtered.tsv")
    cosine_gaussian_file = os.path.join(gaussian_ranking_parent_dir, "ranking_c_initial_VS_noisy_5_stdDev_d_v5.0_LSU_ge_filtered.tsv")
    euclidean_impulse_file = os.path.join(impulse_ranking_parent_dir, "ranking_e_initial_VS_noisy_0.9_impL_d_v5.0_LSU_ge_filtered.tsv")
    cosine_impulse_file = os.path.join(impulse_ranking_parent_dir, "ranking_c_initial_VS_noisy_0.9_impL_d_v5.0_LSU_ge_filtered.tsv")
    ### --- ###
    
    # Extract pattern of interest and noise level from filename
    gaussian_pattern, gaussian_noise_level = extract_pattern(euclidean_gaussian_file)
    impulse_pattern, impulse_noise_level = extract_pattern(euclidean_impulse_file)
    
    # Create subdirectories
    gaussian_plots_dir = os.path.join(gaussian_ranking_parent_dir, "plots")
    impulse_plots_dir = os.path.join(impulse_ranking_parent_dir, "plots")
    # Check if directory exists - if not, create dir 
    ensure_dir(gaussian_plots_dir)
    ensure_dir(impulse_plots_dir)
    # Construct file name     
    gaussian_output_file = os.path.join(gaussian_plots_dir, gaussian_pattern) + ".png"
    impulse_output_file = os.path.join(impulse_plots_dir, impulse_pattern) + ".png"
    
    # Load data
    ranking_gaussian_noise = load_and_combine_data(euclidean_gaussian_file, cosine_gaussian_file)
    ranking_impulse_noise = load_and_combine_data(euclidean_impulse_file, cosine_impulse_file)
    
    # Plot
    plot_sim_performance(ranking_gaussian_noise, gaussian_output_file, gaussian_noise_level)
    plot_sim_performance(ranking_impulse_noise, impulse_output_file, impulse_noise_level)
    
if __name__ ==  "__main__":
    main()