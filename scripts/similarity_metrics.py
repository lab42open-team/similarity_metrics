#!/usr/bin/python3.5

# script name: similarity_metrics.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate similarity scores between downsampled intial data VS noisy data. 
    # Used for initial VS noisy versions to check similarity performance
    # No input parameters provided, but input/output should be adjusted accordingly. 
        # Please check the session: ### TO BE ADJUSTED ### within the main function. 
# framework: CCMRI
# last update: 27/06/2024

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
    df1 = pd.read_csv(initial_file,sep="\t")
    df2 = pd.read_csv(noisy_file, sep="\t")
    # DEBUGGING LOGS 
    logging.info("Initial data loaded correctly: {}".format(df1.shape))
    logging.info("Noisy data loaded correctly: {}".format(df2.shape))
    # Combine the data into a single DataFrame
    combined_df = pd.concat([df1, df2])
    # Pivot table to have taxa as rows and samples as columns 
    pivot_df = combined_df.pivot_table(index="Taxa", columns="Sample", values="Count", fill_value=0)
    # DEBUGGING LOGS
    logging.info("Pivot table shape: {}".format(pivot_df.shape))
    return df1, df2, pivot_df

def calculate_similarity_scores(df1, df2, pivot_df, output_dir, noisy_file_name, metric):
    similarity_pairs = []
    for sample1 in df1["Sample"].unique():
        for sample2 in df2["Sample"].unique():
            # Extract vectors for the samples
            vector1 = pivot_df[sample1].values.reshape(1, -1)
            vector2 = pivot_df[sample2].values.reshape(1, -1)
            if metric == "euclidean":
                # Calculate Euclidean distance and similarity score
                euclidean_dist = euclidean_distances(vector1, vector2)[0][0]
                similarity_score = 1 / (1 + euclidean_dist)
            elif metric == "cosine":
                # Compute Cosine distance and similarity score
                cosine_distance = cosine_distances(vector1, vector2)[0][0]
                similarity_score = 1 - cosine_distance
            else:
                raise ValueError("Unsupported metric: {}".format(metric))
            
            # Append to similarity pairs list 
            similarity_pairs.append([sample1, sample2, similarity_score])
    # Convert to DataFrame
    similarity_df  = pd.DataFrame(similarity_pairs, columns=["Initial_Sample", "Noisy_Sample", "Similarity_Score"])
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
    gaussian_noisy_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/gaussian_noisy_data/d_1000/5_stdDev"
    # Impulse Noise Parent dir
    impulse_noisy_parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/impulse_noisy_data/d_1000/0.9_noiseLevel"
    # Define file names to be compared
    intial_file_name = "ra_d_v4.1_LSU_ge_filtered.tsv"
    gaussian_noisy_file_name = "noisy_5_stdDev_d_v4.1_LSU_ge_filtered.tsv"
    impulse_noisy_file_name = "noisy_0.9_impL_d_v4.1_LSU_ge_filtered.tsv"
    initial_file = os.path.join(initial_parent_directory, intial_file_name)
    ### --- ###
    
    # Load noisy data 
    gaussian_noisy_file = os.path.join(gaussian_noisy_parent_directory, gaussian_noisy_file_name)
    gaussian_df1, gaussian_df2, gaussian_pivot_df = load_data(initial_file, gaussian_noisy_file)
    impulse_noisy_file = os.path.join(impulse_noisy_parent_directory, impulse_noisy_file_name)
    impulse_df1, impulse_df2, impulse_pivot_df = load_data(initial_file, impulse_noisy_file)
    # Define Gaussian Output dir
    gaussian_euclidean_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/euclidean_output/"
    gaussian_cosine_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSgaussian_noisy/cosine_output"
    # Define Impulse Output dir
    impulse_euclidean_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSimpulse_noisy/euclidean_output"
    impulse_cosine_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/similarity_metrics/sim_initialVSimpulse_noisy/cosine_output"
    # Calculate and Save similarities 
    # Gaussian noisy files 
    calculate_similarity_scores(gaussian_df1, gaussian_df2, gaussian_pivot_df, gaussian_euclidean_output_dir, gaussian_noisy_file, metric="euclidean")
    logging.info("Euclidean similarity scores for initial VS gaussian noisy files calculated successfully.")
    calculate_similarity_scores(gaussian_df1, gaussian_df2, gaussian_pivot_df, gaussian_cosine_output_dir, gaussian_noisy_file, metric="cosine")
    logging.info("Cosine similarity scores for initial VS gaussian noisy files calculated successfully.")
    # Impulse noisy files
    calculate_similarity_scores(impulse_df1, impulse_df2, impulse_pivot_df, impulse_euclidean_output_dir, impulse_noisy_file, metric="euclidean")
    logging.info("Euclidean similarity scores for initial VS impulse noisy files calculated successfully.")
    calculate_similarity_scores(impulse_df1, impulse_df2, impulse_pivot_df, impulse_cosine_output_dir, impulse_noisy_file, metric="cosine")
    logging.info("Cosine similarity scores for initial VS impulse noisy files calculated successfully.")

if __name__ == "__main__":
    main()
    
    
    """
    ### FIRST WAY | TRIAL ###
    def calculate_euclidean_scores(df1, df2, pivot_df, euclidean_output_dir, noisy_file_name):
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
    # DEBUGGING LOGS
    logging.info("Sample Euclidean similarity scores:\n{}".format(euclidean_similarity_df.head()))
    # Group by Noisy Sample and sort according to similarity score within each group
    grouped_euclidean_similarity_df = euclidean_similarity_df.groupby("Noisy_Sample").apply(lambda x: x.sort_values(by="Similarity_Score", ascending=False))
    # Define euclidean output directory and path
    euclidean_output_path = os.path.join(euclidean_output_dir, "e_initial_VS_{}".format(os.path.basename(noisy_file_name)))
    # Save DataFrame
    grouped_euclidean_similarity_df.to_csv(euclidean_output_path, sep="\t", index=False)
    logging.info("Euclidean output saved to: {}".format(euclidean_output_path))
    return grouped_euclidean_similarity_df
    
def calculate_cosine_scores(df1, df2, pivot_df, cosine_output_dir, noisy_file_name):
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
    # DEBUGGING LOGS
    logging.info("Sample Cosine similarity scores:\n{}".format(cosine_similarity_df.head()))
    # Group by Noisy Sample and sort according to similarity score within each group
    grouped_cosine_similarity_df = cosine_similarity_df.groupby("Noisy_Sample").apply(lambda x: x.sort_values(by="Similarity_Score", ascending=False))
    # Define cosine output directory and path 

    cosine_output_path = os.path.join(cosine_output_dir, "c_initial_VS_{}".format(os.path.basename(noisy_file_name)))
    # Save DataFrame 
    grouped_cosine_similarity_df.to_csv(cosine_output_path, sep="\t", index=False)
    logging.info("Cosine output saved to: {}".format(cosine_output_path))
    return grouped_cosine_similarity_df

    """