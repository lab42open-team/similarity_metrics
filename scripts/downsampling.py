#!/usr/bin/python3.5

# script name: downsampling.py
# developed by: Nefeli Venetsianou
# description: 
    # Create downsampled versions for noise injection step. 
    # Sample method returns a random sample of items from an axis of object and this object of same type as the caller.
# framework: CCMRI
# last update: 26/06/2024

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

input_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/v4.1_LSU_ge_filtered.tsv"
output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/downsampled_data/1000_samples"

def main():
    # Read file as pandas DataFrame 
    df = pd.read_csv(input_file, delimiter = "\t")
    # Extract genus information
    df["Genus"] = df["Taxa"].str.split(";").str[-1].str.replace("g__", "")
    # Filter taxa with more than 30 reads (count)
    df = df[df["Count"] > 30] 
    # Find unique samples that have more than 10 genera
    sample_genera_count = df.groupby("Sample")["Genus"].nunique()
    valid_samples = sample_genera_count[sample_genera_count > 10].index
    # Filter DataFrame based on valid samples
    df = df[df["Sample"].isin(valid_samples)]
    # Find unique samples 
    unique_samples = df["Sample"].unique()
    # Downsample at 10% of total samples
    #frac = 0.1
    # Downsample at 1000 unique samples
    num_samples = min(1000, len(unique_samples))
    downsampled_samples = pd.Series(unique_samples).sample(n=num_samples, random_state=1)
    # Ensure retrieval of 1000 Samples
    if len(downsampled_samples) < 1000:
        logging.info("Expected 1000 Samples, but retrieved {}".format(len(downsampled_samples)))
    # Filter DataFrame based on the downsampled unique samples
    downsampled_data = df[df["Sample"].isin(downsampled_samples)]
    # Drop genus column 
    downsampled_data = downsampled_data.drop(columns=["Genus"])
    # Sort DataFrame by Sample 
    sorted_downsampled_df = downsampled_data.sort_values(by="Sample")
    # Save downsampled version 
    # # Extract file name from input file 
    file_name = os.path.basename(input_file)
    # Construct output file path with frac + file_name 
    output_file = os.path.join(output_dir, "d_" + file_name)
    # Save downsampled version to tsv file 
    sorted_downsampled_df.to_csv(output_file, sep="\t", index=False)
    logging.info("Output saved: {}".format(output_file))
    
if __name__ == "__main__":
    main()