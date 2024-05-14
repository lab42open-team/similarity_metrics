#!/usr/bin/python3.5

# script name: taxonomy_counts.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate counts per taxonomy (k > p > c > o > f > g > s) in each version Super Table. 
# no input parameters are allowed 
# framework: CCMRI
# last update: 14/05/2024

import pandas as pd 
import re
from matplotlib import pyplot as plt

# Read file 
df = pd.read_csv("/ccmri/similarity_metrics/data/super_table/long_format/lf_v5.0_LSU_super_table.tsv", sep = "\t")
# Extract taxa column
taxa = df["Taxa"]
#print(taxa)
# Initialize dictionary for each taxonomic level 
counts = {"k__" : set(), "p__" : set(), "c__" : set(), 
          "o__" : set(), "f__": set(), "g__" : set(), "s__" : set()}
# Iterate over each taxon 
for taxon in taxa: 
    # Split taxon by semicolon if necessary 
    sub_taxa = taxon.split(";")
    # Ensure each subtaxon is counted only once - remove duplicates
    unique_sub_taxa = set(sub_taxa)
    for sub in unique_sub_taxa:
        # Extract taxonomic level
        matched_taxa = re.match(r"([kpcofgs])__", sub) # regex used to match any taxonomic level of interest followed by underscores
        if matched_taxa:
            # Define level by grouping according to the single character matched within the regex
            level = matched_taxa.group(1)
            # Extract key to access counts dictionary and add subtaxon to the list
            counts[level + "__"].add(sub) 
# Count unique taxa for each taxonomic level 
unique_counts = {level:len(taxa) for level, taxa in counts.items()}

# Create pie plot
fig = plt.figure(figsize=(10,7))
# filter out items with zero counts
non_zero_counts = {level : count for level, count in unique_counts.items() if count != 0}
# Combine taxonomic labels and counts for display
labels_with_counts = ["{}:{}".format(level, count) for level, count in non_zero_counts.items()]
plt.pie(non_zero_counts.values(), labels = labels_with_counts)
plt.title("Unique Taxa Counts v5.0_LSU")
plt.savefig("/ccmri/similarity_metrics/data/super_table/long_format/plots/v5.0_LSU_taxonomy.png")

# Print counts of unique taxa found
for level, count in unique_counts.items():
    print("{}:{}".format(level, count))
        
