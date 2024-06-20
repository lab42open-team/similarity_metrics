#!/usr/bin/python3.5

# script name: downsampling.py
# developed by: Nefeli Venetsianou
# description: 
    # Create downsampled versions for noise injection step. 
    # Sample method returns a random sample of items from an axis of object and this object of same type as the caller.
# framework: CCMRI
# last update: 20/06/2024

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

input_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/v5.0_LSU_ge_filtered.tsv"
output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions_data/downsampled_data"

def main():
    # Read file as pandas DataFrame 
    df = pd.read_csv(input_file, delimiter = "\t") 
    # Find unique samples
    unique_samples = df["Sample"].unique()
    # Downsample at 10% of total samples
    frac = 0.1
    num_samples = max(1, int(len(unique_samples) * frac))
    downsampled_samples = pd.Series(unique_samples).sample(n=num_samples, random_state=1)
    # Filter DataFrame based on the downsampled unique samples
    downsampled_data = df[df["Sample"].isin(downsampled_samples)]
    # Sort DataFrame by Sample 
    sorted_downsampled_df = downsampled_data.sort_values(by="Sample")
    # Save downsampled version 
    # # Extract file name from input file 
    file_name = os.path.basename(input_file)
    # Construct output file path with frac + file_name 
    output_file = os.path.join(output_dir, "{}_{}".format(frac, file_name))
    # Save downsampled version to tsv file 
    sorted_downsampled_df.to_csv(output_file, sep="\t", index=False)
    logging.info("Output saved: {}".format(output_file))
    
if __name__ == "__main__":
    main()