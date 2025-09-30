#!/usr/bin/python3.5

# script name: filter_taxonomy_depth.py
# developed by: Nefeli Venetsianou
# description: 
#   Filter data to keep only the taxonomic level of choice.
#   Removes suffix after the taxonomic level of choice, but keeps prefix to know the hierarchy. 
#   Should be applied after super table creation and before normalization. 
#   Output saved automatically and user can choose (optionally) to save unique filtered found in a separate .tsv file.
#   Now includes:
#     Step 1: Filter taxa with more than 30 reads
#     Step 2: Filter samples with more than 10 genera
#
# Input parameters: filename of super table to be analyzed (eg. lf_v1.0_super_table.tsv)
# framework: CCMRI
# last update: 13/08/2025

import sys, os, re
import pandas as pd

parent_directory = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/no_filtered"
output_directory = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/"

def read_file(filename):
    file_path = os.path.join(parent_directory, filename)
    if not os.path.exists(file_path) or not file_path.endswith(".tsv"):
        print("File not found or not a .tsv file. Please provide a valid file. If you wish, check parent directory '{}' \
              for available input files.".format(parent_directory))
        sys.exit(1)
    return pd.read_csv(file_path, sep="\t")

def filter_taxa_depth(df):
    filtered_df = df[df["Taxa"].str.contains("g__")].copy()
    filtered_df.loc[:, "Taxa"] = filtered_df["Taxa"].str.replace(r"(g__[^;]*);.*", r"\1", regex=True)
    return filtered_df

def filter_by_reads(df, min_reads=30):
    df["Count"] = pd.to_numeric(df["Count"], errors="coerce")
    return df[df["Count"] > min_reads]

def filter_by_genera_per_sample(df, min_genera=10):
    df["Depth"] = df["Taxa"].str.split(";").str[-1]
    genera_per_sample = df.groupby("Sample")["Depth"].nunique()
    samples_to_keep = genera_per_sample[genera_per_sample > min_genera].index
    return df[df["Sample"].isin(samples_to_keep)].drop(columns=["Depth"])

def sum_abundance(filtered_df):
    summed_df = filtered_df.groupby(["Taxa", "Sample"]).agg({"Count": "sum"}).reset_index()
    return summed_df
    
def unique_filtered_df(df):
    unique_filtered_df = df.drop_duplicates(subset=["Taxa"])
    unique_filtered_count = unique_filtered_df.shape[0]
    print("Number of unique filtered level: {}".format(unique_filtered_count))

def save_filtered_data(df, output_filename):
    output_file_path = os.path.join(output_directory, output_filename)
    df.to_csv(output_file_path, sep="\t", index=False)
    print("Filtered data saved to {}".format(output_file_path))

def main():
    if len(sys.argv) < 2:
        print("Please provide the filename (eg. lf_v4.1_LSU_super_table.tsv)")
        sys.exit(1)
    
    filename = sys.argv[1]
    df = read_file(filename)

    # Step A: Keep only genus-level taxa
    filtered_data = filter_taxa_depth(df)

    # Step 1: Filter taxa with more than 30 reads
    filtered_data = filter_by_reads(filtered_data, min_reads=30)

    # Step 2: Filter samples with more than 10 genera
    filtered_data = filter_by_genera_per_sample(filtered_data, min_genera=10)

    # Step B: Sum abundances per sample-taxon
    filtered_summed_data = sum_abundance(filtered_data)

    # Step C: Print unique taxa info
    unique_filtered_df(filtered_summed_data)

    # Step D: Save results
    filtered_summed_data_filename = re.sub(r"^lf_", "", filename)[:-16] + "_ge_filtered.tsv"
    save_filtered_data(filtered_summed_data, filtered_summed_data_filename)

if __name__ == "__main__":
    main()
