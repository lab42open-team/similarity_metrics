#!/usr/bin/python3.5

# script name: runs_separation.py
# developed by: Nefeli Venetsianou
# description: 
    # Separate runs from assemblies.
# framework: CCMRI
# last update: 25/11/2024

import os 
import pandas as pd

parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/normalized_counts"
runs_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/normalized_counts/runs_files"

"""
### PART 1: Separate runs from assemblies
# Process each .tsv file in the parent directory
for filename in os.listdir(parent_dir):
    if filename.endswith(".tsv"):
        file_path = os.path.join(parent_dir, filename)
        # Read tsv file into a DataFrame
        df = pd.read_csv(file_path, sep="\t")
        # Filter rows where the run/assebly name of second columns ends in R (meaning it's a run code)
        runs_df = df[df["Sample"].str[:3].str.endswith("R")]
        # Save filtered DataFrame to a new tsv file 
        output_file_path = os.path.join(runs_output_dir, "runs_{}".format(filename))
        runs_df.to_csv(output_file_path, sep="\t", index=False)
        print("Runs file saved to: {}".format(output_file_path))
"""
### PART 2: Retrieve only metagenomic experiment type
# Define files
extracted_runs_file = "/ccmri/similarity_metrics/data/experiment_type/extracted_runs_data.tsv"
metagenomics_runs_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/normalized_counts/runs_files/metagenomics"

# Load metagenomics runs ids
metagenomic_ids = pd.read_csv(extracted_runs_file, sep="\t", usecols=["run_id"])
metagenomic_set = set(metagenomic_ids["run_id"])
# Process each runs tsv file
for filename in os.listdir(runs_output_dir):
    if filename.endswith(".tsv"):
        file_path = os.path.join(runs_output_dir, filename)
        # Read runs filtered tsv file
        runs_df = pd.read_csv(file_path, sep="\t")
        # Keep only rows where Sample IDs exists in metagenomic experiment IDs
        comparison_df = runs_df[runs_df["Sample"].isin(metagenomic_set)]
        # Save the compared data to a new file
        output_file_path = os.path.join(metagenomics_runs_output_dir, "mtgmc_{}".format(filename))
        comparison_df.to_csv(output_file_path, sep="\t", index=False)
        print("Comparison file saved to: {}".format(output_file_path))


