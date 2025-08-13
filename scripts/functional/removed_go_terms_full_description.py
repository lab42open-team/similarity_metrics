#!/usr/bin/python3.5

# script name: removed_remained_go_terms_full_description.py
# developed by: Nefeli Venetsianou
# description: Match GO terms with their corresponding information from full entities and full names TSV files, either for removed or for remained GO terms. 
# framework: CCMRI
# last update: 24/02/2025

import pandas as pd
import os

# Define file paths
input_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/filtered_cutOff/test_set_logs/"
LOG_FILE = os.path.join(input_dir, "ERR9456921_filtered_0.50_0.01_log.tsv")
ENTITIES_FILE = "/ccmri/similarity_metrics/data/functional/dictionary/full_entities.tsv"
NAMES_FILE = "/ccmri/similarity_metrics/data/functional/dictionary/full_names.tsv"
input_file_name = os.path.basename(LOG_FILE)
OUTPUT_FILE = os.path.join(input_dir, "full_description_{}".format(input_file_name))

# Load the data (Include Count column)
go_terms = pd.read_csv(LOG_FILE, sep="\t", usecols=[0, 2], names=["GO", "Count"]).drop_duplicates()

# Load entities and names data
entities = pd.read_csv(ENTITIES_FILE, sep="\t", names=["ID", "Entity_Type", "GO"], dtype={"GO": str})
names = pd.read_csv(NAMES_FILE, sep="\t", names=["ID", "Full_Name"])

# Filter entities to keep only relevant GO terms
filtered_entities = entities.merge(go_terms, on="GO", how="inner")  # Keep Count column

# Merge with names to get full names
merged_df = filtered_entities.merge(names, on="ID", how="left")

# Group by ID and Entity_Type to combine multiple names into a single comma-separated string
merged_df = merged_df.groupby(["ID", "Entity_Type", "GO", "Count"])["Full_Name"].apply(lambda x: ",".join(x.dropna())).reset_index()

# Sort by Entity Type before saving
merged_df = merged_df.sort_values(by="Entity_Type")

# Save the final output
merged_df.to_csv(OUTPUT_FILE, sep="\t", index=False)

print(f"Processing completed. Output saved to {OUTPUT_FILE}")
