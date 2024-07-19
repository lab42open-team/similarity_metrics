#!/usr/bin/python3.5

# script name: avg_ranking_sim_performance.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate average ranking per noise level & type in already created ranking files (produced by: ranking_metrics_performance.py)
# framework: CCMRI
# last update: 16/07/2024

import logging.config
import os, re
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def find_files(dir, noise_level):
    files = []
    logging.debug("searching in directory {}".format(dir))
    for root, dirs, filenames in os.walk(dir):
        euclidean_files = [os.path.join(root, f) for f in filenames if noise_level in f and "e_" in f]
        cosine_files = [os.path.join(root, f) for f in filenames if noise_level in f and "c_" in f]
        if euclidean_files and cosine_files:
            files.append((euclidean_files[0], cosine_files[0]))
    return files

def load_ranking_data(file):
    df = pd.read_csv(file, sep="\t")
    logging.info("Data from {} loaded correctly.".format(format(os.path.basename(file))))
    return df["Rank"]

def calculate_average_ranking(files):
    euclidean_ranks = []
    cosine_ranks = []
    for e_file, c_file in files:
        euclidean_ranks.extend(load_ranking_data(e_file))
        cosine_ranks.extend(load_ranking_data(c_file))
    avg_euclidean_rank = sum(euclidean_ranks) / len(euclidean_ranks) if euclidean_ranks else float("nan")
    avg_cosine_rank = sum(cosine_ranks) / len(cosine_ranks) if cosine_ranks else float("nan")
    return avg_euclidean_rank, avg_cosine_rank

def save_results(output_file, noise_level, avg_euclidean_rank, avg_cosine_rank):
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("Noise\tEuclidean Rank\tCosine Rank\n")
    with open(output_file, "a") as f:
        f.write("{}\t{}\t{}\n".format(noise_level, avg_euclidean_rank, avg_cosine_rank))

def main():
    # Define output directories
    gaussian_parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/ranking_output"
    impulse_parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSimpulse_noisy/ranking_output"
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics"
    output_file_name = "average_ranking.tsv"
    output_file = os.path.join(output_dir, output_file_name)
    # Define noise level
    noise_levels = ["0.2_stdDev", "0.5_stdDev", "1_stdDev", "5_stdDev", "0.01_impL", "0.03_impL" ,"0.05_impL", "0.1_impL"]
    # Iterate over each noise level
    for noise_level in noise_levels:
        if "stdDev" in noise_level:
            parent_dir = gaussian_parent_dir
        else:
            parent_dir = impulse_parent_dir
        # Find all relevant file
        all_files = find_files(parent_dir, noise_level)
        # Calculate average ranking 
        avg_euclidean_rank, avg_cosine_rank = calculate_average_ranking(all_files)
        # Save results 
        save_results(output_file, noise_level, avg_euclidean_rank, avg_cosine_rank)
        # Log results 
        logging.info("Average euclidean rank for noise level {} : {}".format(noise_level, avg_euclidean_rank))
        logging.info("Average cosine rank for noise level {} : {}".format(noise_level, avg_cosine_rank))
        
if __name__ == "__main__":
    main()
            
    
