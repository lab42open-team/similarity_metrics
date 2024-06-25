#!/usr/bin/python3.5

# script name: similarity_metrics.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate similarity scores between downsampled intial data VS noisy data. 
    # Used for initial VS noisy versions to check similarity performance
# framework: CCMRI
# last update: 25/06/2024

import os
import pandas as pd
import numpy as np
from scipy.spatial import distance 
from scipy.spatial.distance import euclidean, pdist, squareform
import time
import logging
logging.basicConfig(level=logging.INFO)

# Define parent directory of intial data [downsampled]
initial_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/downsampled_data/1000_samples/normalized/"
# Define parent directory of noisy data [level of noise]
noisy_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/gaussian_noisy_data/d_1000/std_noise_level_0.1"
# Define file names to be compared
intial_file_name = "ra_v5.0_LSU_ge_filtered.tsv"
noisy_file_name = "noisy_0.1_stdDev_v5.0_LSU_ge_filtered.tsv"

def load_data(initial_file, noisy_file):
    # Create DataFrames from input files
    df1 = pd.read_csv(initial_file,sep="\t")
    df2 = pd.read_csv(noisy_file, sep="\t")
    # Combine the data into a single DataFrame
    combined_df = pd.concat([df1, df2])
    # Pivot table to have taxa as rows and samples as columns 
    pivot_df = combined_df.pivot_table(index="Taxa", columns="Sample", values="Count", fill_value=0)
    return df1, df2, pivot_df

def calculate_euclidean_scores(df1, df2, pivot_df):
    euclidean_similarity_pairs = []
    for sample1 in df1["Sample"].unique():
        for sample2 in df2["Sample"].unique():
            # Compute the Euclidean distance matrix between Sample1 and Sample2
            euclidean_dist = euclidean(pivot_df[sample1], pivot_df[sample2])
            # Calculate similarity score
            euclidean_similarity_score = 1 / (1 + euclidean_dist)
            # Append to euclidean similarity pairs list 
            euclidean_similarity_pairs.append([sample1, sample2, euclidean_similarity_score])
    # Convert to DataFrame
    euclidean_similarity_df = pd.DataFrame(euclidean_similarity_pairs, columns=["Initial_Sample", "Noisy_Sample", "Similarity_Score"])
    # Group by Noisy Sample and sort according to similarity score within each group
    grouped_euclidean_similarity_df = euclidean_similarity_df.groupby("Noisy_Sample").apply(lambda x: x.sort_values(by="Similarity_Score", ascending=False))
    # Define euclidean output directory and path
    euclidean_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output/"
    euclidean_output_path = os.path.join(euclidean_output_dir, "e_initial_VS_{}".format(os.path.basename(noisy_file_name)))
    # Save DataFrame
    grouped_euclidean_similarity_df.to_csv(euclidean_output_path, sep="\t", index=False)
    logging.info("Euclidean output saved to: {}".format(euclidean_output_path))
    return grouped_euclidean_similarity_df
    
def calculate_cosine_scores(df1, df2, pivot_df):
    cosine_similarity_pairs = []
    for sample1 in df1["Sample"].unique():
        for sample2 in df2["Sample"].unique():
            # Compute the Cosine distance between Sample1 and Sample2
            cosine_dist = distance.cosine(pivot_df[sample1], pivot_df[sample2])
            # Calculate cosine score
            cosine_similarity_score = 1 - cosine_dist
            # Append to cosine similarity pairs list
            cosine_similarity_pairs.append([sample1, sample2, cosine_similarity_score])
    # Convert to DataFrame
    cosine_similarity_df = pd.DataFrame(cosine_similarity_pairs, columns=["Initial_Sample", "Noisy_Sample", "Similarity_Score"])
    # Group by Noisy Sample and sort according to similarity score within each group
    grouped_cosine_similarity_df = cosine_similarity_df.groupby("Noisy_Sample").apply(lambda x: x.sort_values(by="Similarity_Score", ascending=False))
    # Define cosine output directory and path 
    cosine_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output/"
    cosine_output_path = os.path.join(cosine_output_dir, "c_initial_VS_{}".format(os.path.basename(noisy_file_name)))
    # Save DataFrame 
    grouped_cosine_similarity_df.to_csv(cosine_output_path, sep="\t", index=False)
    logging.info("Cosine output saved to: {}".format(cosine_output_path))
    return grouped_cosine_similarity_df
"""
def create_euclidean_ranking_table(euclidean_similarity_df): #TODO not working properly
    ranking_pairs = []
    for noisy_sample in euclidean_similarity_df["Noisy_Sample"].unique():
        noisy_sample_df = euclidean_similarity_df[euclidean_similarity_df["Noisy_Sample"] == noisy_sample]
        rank = noisy_sample_df[noisy_sample_df["Initial_Sample"]].index[0] + 1
        ranking_pairs.append([noisy_sample, rank])
    ranking_df = pd.DataFrame(ranking_pairs, columns=["Noisy_Sample", "Rank"])
    # Define ranking output directory and path
    ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output/ranking_output"
    ranking_output_path = os.path.join(ranking_output_dir, "ranking_output_{}".format(os.path.basename(noisy_file_name)))
    # Save DataFrame
    ranking_df.to_csv(ranking_output_path, sep="\t", index=False)
    logging.info("Ranking euclidean output saved to: {}".format(ranking_output_path))
    
def create_cosine_ranking_table(cosine_similarity_df): #TODO not working properly
    ranking_pairs = []
    for noisy_sample in cosine_similarity_df["Noisy_Sample"].unique():
        noisy_sample_df = cosine_similarity_df[cosine_similarity_df["Noisy_Sample"] == noisy_sample]
        rank = noisy_sample_df[noisy_sample_df["Initial_Sample"]].index[0] + 1
        ranking_pairs.append([noisy_sample, rank])
    ranking_df = pd.DataFrame(ranking_pairs, columns=["Noisy_Sample", "Rank"])
    # Define ranking output directory and path
    ranking_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output/ranking_output"
    ranking_output_path = os.path.join(ranking_output_dir, "ranking_output_{}".format(os.path.basename(noisy_file_name)))
    # Save DataFrame
    ranking_df.to_csv(ranking_output_path, sep="\t", index=False)
    logging.info("Ranking cosine output saved to: {}".format(ranking_output_path))
"""
def main():
    # Load data
    initial_file = os.path.join(initial_parent_directory, intial_file_name)
    noisy_file = os.path.join(noisy_parent_directory, noisy_file_name)
    df1, df2, pivot_df = load_data(initial_file, noisy_file)
    logging.info("Data loaded successfully.")
    # Calculate similarity metrics
    euclidean_similarity_matrix = calculate_euclidean_scores(df1, df2, pivot_df)
    logging.info("Euclidean similarity scores calculated successfully.")
    cosine_similarity_matrix = calculate_cosine_scores(df1, df2, pivot_df)
    logging.info("Cosine similarity scores calculated successfully.")
    """ #TODO not working properly
    # Create ranking tables
    create_euclidean_ranking_table(grouped_euclidean_similarity_df)
    create_cosine_ranking_table(grouped_cosine_similarity_df)
    """
if __name__ == "__main__":
    main()