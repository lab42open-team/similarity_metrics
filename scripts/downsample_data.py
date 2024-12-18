#!/usr/bin/python3.5

# script name: downsample_data.py
# developed by: Nefeli Venetsianou
# description: 
    # Downsample Data: 
        # Pick (if needed) 1000 Unique Samples 
        # Taxa with more than 30 reads (counts)
        # Samples with more than 10 genera
    # Then, Renormalize Data 
# framework: CCMRI
# last update: 18/12/2024

import pandas as pd

# Load the dataset
file_path = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/cc_related/cc_ge_filtered_data.tsv"  
data = pd.read_csv(file_path, sep="\t")

# Step 1: Filter taxa with more than 30 reads
# Convert "Count" column to numeric if it"s not already
data["Count"] = pd.to_numeric(data["Count"], errors="coerce")
data = data[data["Count"] > 30]

# Step 2: Filter samples with more than 10 genera
# Extract genus level from the "Taxa" column (assuming genus is the last part of the taxonomy split by `;`)
data["Genus"] = data["Taxa"].str.split(";").str[-1]

# Count unique genera per sample
genera_per_sample = data.groupby("Sample")["Genus"].nunique()
samples_to_keep = genera_per_sample[genera_per_sample > 10].index

# Keep only rows from samples with more than 10 genera
data = data[data["Sample"].isin(samples_to_keep)]

# Step 3: Re-normalize counts within each sample
# Normalize counts so that they sum to 1 within each sample
data["Count"] = data.groupby("Sample")["Count"].transform(lambda x: x / x.sum())

# Drop the intermediate "Genus" column if not needed
data = data.drop(columns=["Genus"])

# Save the processed data to a new file
output_path = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/cc_related/downsampled_filtered_data.tsv"
data.to_csv(output_path, sep="\t", index=False)

print(f"Processed data saved to {output_path}")
