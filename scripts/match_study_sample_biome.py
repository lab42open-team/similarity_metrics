#!/usr/bin/python3.5

# script name: biome_retrieval_mined_info.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve biome from mined info txt files.
# framework: CCMRI
# last update: 17/10/2024

study_sample_file = "/ccmri/similarity_metrics/data/raw_data/studies_samples/studies_samples_v5.0_SSU.tsv"
study_biome_file = "/ccmri/similarity_metrics/data/raw_data/studies_samples/study_biome_info.tsv"

import pandas as pd
import os

# Load the TSV files into dataframes
study_sample_df = pd.read_csv(study_sample_file, sep="\t")
study_biome_df = pd.read_csv(study_biome_file, sep="\t")

# Merge the two dataframes on the "MGYS_ID" from study_sample_df and "study_id" from study_biome_df
merged_df = pd.merge(study_sample_df, study_biome_df, left_on="MGYS_ID", right_on="study_id", how="left")

# Fill NaN biome values with "unknown"
merged_df["biome_info"].fillna("unknown", inplace=True)

# Rename columns
merged_df.rename(columns={"MGYS_ID": "study_id", "Sample_Name": "sample_id"}, inplace=True)

# Drop the duplicate "study_id" column from the merge (keeping the renamed one)
merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

# Remove duplicates
merged_df.drop_duplicates(subset=["study_id", "sample_id", "biome_info"], inplace=True)

# Set up the output directory and filename
output_dir = "/ccmri/similarity_metrics/data/raw_data/studies_samples/biome_info"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Dynamically create output name based on input filename
output_name = os.path.basename(study_sample_file)[16:]  # Adjust as necessary
output_path = os.path.join(output_dir, "study-sample-biome_{}".format(output_name))

# Save the resulting dataframe as a new TSV file in the output directory
merged_df[["study_id", "sample_id", "biome_info"]].to_csv(output_path, sep="\t", index=False)

print("File saved to {}".format(output_path))
