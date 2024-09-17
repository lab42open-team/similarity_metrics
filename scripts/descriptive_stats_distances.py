#!/usr/bin/python3.5

# script name: descriptive_stats_distances.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve the descriptive statistics for the calculated distances per file (version pipeline of MGnify)
# To Be Adjusted (TBA) parameters:
    # input_file: define the file with calculated distances of which you want to get the results.
# last update: 17/09/2024

import os 
import pandas as pd

# Define files
parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics"
input_file = "c_distances_filtered_v4.1_LSU_ge_filtered.tsv" # TBA
file_path = os.path.join(parent_directory, input_file)

# Load data
df = pd.read_csv(file_path, sep = "\t")
print(df.describe())
