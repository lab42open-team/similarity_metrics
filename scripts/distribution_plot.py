#!/usr/bin/python3.5

# script name: distribution_plot.py
# developed by: Nefeli Venetsianou
# description: 
    # Plot distribution in taxonomic depth accross all samples per version.
    # Applied in normalized & aggregated data in Super Table (produced by: normalization_counts_ra.py & ra_aggregation.py)
# input parameters from cmd:
    # --input_dir="your_input_directory"
    # --output_dir="your_output_directory"
# framework: CCMRI
# last update: 13/05/2024

import pandas as pd
import matplotlib 
# use non-interactive backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Parse data into a DataFrame 
df = pd.read_csv("/ccmri/similarity_metrics/data/super_table/long_format/lf_v1.0_super_table.tsv", sep="\t")
print(df)
# Extract taxa column
#taxa = df["Taxa"]
# Calculate frequency (number of occurrences) per unique taxon 
# value_counts() function in pandas returns a Series containing counts of unique values in a DF column. 
taxa_frequency = df["Taxa"].value_counts()

# Plot histogram 
plt.figure(figsize=(12,8))
taxa_frequency.plot(kind="bar")
plt.xlabel("Taxa")
plt.ylabel("Total Count") # TODO fix y axis to see more details on frequency numbers
plt.title("Taxa Distribution Plot")
plt.xticks(rotation=90)
plt.yticks(range(0, max(taxa_frequency)+1, 200))
plt.tight_layout()
plt.savefig("/ccmri/similarity_metrics/data/super_table/long_format/distribution_test.png")
