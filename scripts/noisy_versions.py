#!/usr/bin/python3.5

# script name: noisy_versions.py
# developed by: Nefeli Venetsianou
# description: 
    # Create noisy versions for noise injection step. 
    # Sample method returns a random sample of items from an axis of object and this object of same type as the caller.
    # Define frac to retrieve the number of noise you wish (eg. 0.2 = 20% of total rows)
# framework: CCMRI
# last update: 18/06/2024

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

input_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/v4.1_SSU_ge_filtered.tsv"
output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/v4.1_SSU/"

# Read file as pandas DataFrame 
df = pd.read_csv(input_file, delimiter = "\t")
# Noisy Version creation by randomly choosing x% of rows - please define percentage in the frac variable
# For now, frac = 0.2 || 0.3 || 0.5 || 0.7
frac = 0.7
noisy_version_df = df.sample(frac=frac)
# Save noisy version 
# Extract file name from input file 
file_name = os.path.basename(input_file)
# Construct output file path with frac + file_name 
output_file = os.path.join(output_dir, "{}_{}".format(frac, file_name))
# Save noisy version to tsv file 
noisy_version_df.to_csv(output_file, sep="\t", index=False)
logging.info("Output saved: {}".format(output_file))