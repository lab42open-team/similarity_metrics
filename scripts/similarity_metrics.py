#!/usr/bin/python3.5

# script name: similarity_metrics.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate similarity scores between samples.
    # Used for initial VS noisy versions to check similarity performance
# framework: CCMRI
# last update: 19/06/2024

import os
import pandas as pd
import numpy as np
from scipy.spatial import distance 
from scipy.spatial.distance import euclidean, pdist, squareform
import time
import logging
logging.basicConfig(level=logging.INFO)

# Load data from two different .tsv files to be compared
file1 = "/ccmri/similarity_metrics/data/test_dataset/metrics_dataset/ra_200_v4.1_LSU_ge_filtered.tsv"
file2 = "/ccmri/similarity_metrics/data/test_dataset/metrics_dataset/ra_0.5_200_v4.1_LSU_ge_filtered.tsv"
# Create DataFrames from input files
df1 = pd.read_csv(file1,sep="\t")
df2 = pd.read_csv(file2, sep="\t")
# Combine the data into a single DataFrame
combined_df = pd.concat([df1, df2])
# Pivot table to have taxa as rows and samples as columns 
pivot_df = combined_df.pivot_table(index="Taxa", columns="Sample", values="Count", fill_value=0)

### EUCLIDEAN ###
# Compute the Euclidean distance matrix 
euclidean_dist_matrix = squareform(pdist(pivot_df.T, metric="euclidean"))
# Convert distance to similarity scores (= 1 / (1+ euclidean_dist_matrix))
euclidean_similarity_matrix = 1 / (1 + euclidean_dist_matrix)
# Create DataFrame 
euclidean_similarity_df = pd.DataFrame(euclidean_similarity_matrix, index=pivot_df.columns, columns=pivot_df.columns)
# Add column to DataFrame for the sample being compared (set indexing to sample)
euclidean_similarity_df.insert(0, "Sample", euclidean_similarity_df.index)
# Save euclidean_similarity_df
euclidean_output_file = "/ccmri/similarity_metrics/data/test_dataset/metrics_dataset/test_euclidean_output_200VS0.5_metrics.tsv"
euclidean_similarity_df.to_csv(euclidean_output_file, sep="\t", index=False)
logging.info("Euclidean output saved to: {}".format(euclidean_output_file))
#print(euclidean_similarity_df)

### COSINE ###
# Compute the Cosine distance matrix 
cosine_dist_matrix = squareform(pdist(pivot_df.T, metric="cosine"))
# Convert distance matrix to similarity scores (= 1 - cosine_dist_matrix)
cosine_similarity_matrix = 1 - cosine_dist_matrix
# Create DataFrame 
cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=pivot_df.columns, columns=pivot_df.columns)
# Add columns to DataFrame for the sample being compared (set indexing to sample)
cosine_similarity_df.insert(0, "Sample", cosine_similarity_df.index)
# Save cosine_similarity_df
cosine_output_file = "/ccmri/similarity_metrics/data/test_dataset/metrics_dataset/test_cosine_output_200VS0.5_metrics.tsv"
cosine_similarity_df.to_csv(cosine_output_file, sep="\t", index=False)
logging.info("Cosine output saved to: {}".format(cosine_output_file))
#print(cosine_similarity_df)