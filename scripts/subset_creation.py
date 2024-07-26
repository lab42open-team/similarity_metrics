#!/usr/bin/python3.5

# script name: subset.py
# developed by: Nefeli Venetsianou
# description: 
    # Create downsampled versions.
    # Filter by number of reads per taxon and number of taxa per sample. 
    # If number of samples for the output is defined, then purpose is for noise injection step. 
    # Sample method returns a random sample of items from an axis of object and this object of same type as the caller.
# framework: CCMRI
# last update: 19/07/2024

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

input_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/v5.0_SSU_ge_filtered.tsv"
output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/downsampled_data"
# output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/downsampled_data/1000_samples" # For noise injection

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
    # Drop genus column 
    df = df.drop(columns=["Genus"])
    # Normalize counts
    sample_total = df.groupby("Sample")["Count"].sum().to_dict()
    df["Count"] = df.apply(lambda row: row["Count"] / sample_total[row["Sample"]] if sample_total[row["Sample"]] != 0 else 0, axis=1)
    # Sort DataFrame by Sample 
    sorted_df = df.sort_values(by="Sample")
    # Save downsampled version 
    # # Extract file name from input file 
    file_name = os.path.basename(input_file)
    # Construct output file path with frac + file_name 
    output_file = os.path.join(output_dir, "filtered_" + file_name)
    # Save downsampled version to tsv file 
    sorted_df.to_csv(output_file, sep="\t", index=False)
    logging.info("Output saved: {}".format(output_file))
    
if __name__ == "__main__":
    main()