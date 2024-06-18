#!/usr/bin/python3.5

# script name: filter_taxonomy_depth.py
# developed by: Nefeli Venetsianou
# description: 
    # Filter data to keep only the taxonomic level of choice.
    # Removes suffix after the taxonomic level of choice, but keeps prefix to know the hierarchy. 
    # Shoyld be applied after super table creation and before normalization. 
    # Output saved automatically and user can choose (optionally) to save unique filtered found is a separate .tsv file.
# Input parameters: filename of super table to be analyzed (eg. lf_v1.0_super_table.tsv)
# framework: CCMRI
# last update: 17/06/2024

import sys, os, re
import pandas as pd

parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/no_filtered"
output_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/"

def read_file(filename):
    file_path = os.path.join(parent_directory, filename)
    # check is file exists and is a tsv file
    if not os.path.exists(file_path) or not file_path.endswith(".tsv"):
        print("File not found or not a .tsv file. Please provide a valid file. If you wish, check parent directory '{}' \
              for available input files.".format(parent_directory))
        sys.exit(1)
    return pd.read_csv(file_path, sep="\t")

def filter_taxa_depth(df):
    # Filter rows where Taxa contains "g__"
    filtered_df = df[df["Taxa"].str.contains("g__")].copy() 
    # Remove every taxon after genus, using semicolon separator
    filtered_df.loc[:, "Taxa"] = filtered_df["Taxa"].str.replace(r"(g__[^;]*);.*", r"\1",regex = True)
    return filtered_df

def sum_abundance(filtered_df):
    # Sum abundance (count) for the same sample - taxa group
    sumed_df = filtered_df.groupby(["Taxa", "Sample"]).agg({"Count":"sum"}).reset_index()
    return sumed_df
    
def unique_filtered_df(df):
    unique_filtered_df = df.drop_duplicates(subset=["Taxa"])
    unique_filtered_count = unique_filtered_df.shape[0]
    print("Number of unique filtered level: {}".format(unique_filtered_count))
    #for taxon in unique_filtered_df["Taxa"]:
        #print(taxon)
    #return unique_filtered_df # uncomment if you want to save output     

def save_filtered_data(df, output_filename):
    output_file_path = os.path.join(output_directory, output_filename)
    df.to_csv(output_file_path, sep = "\t", index = False)
    print("Filtered data saved to {}".format(output_file_path))

def main():
    # Add filename from command-line arguments
    if len(sys.argv) < 2:
        print("Please provide the filename (eg. lf_v4.1_LSU_super_table.tsv)")
        sys.exit(1)   
    filename = sys.argv[1]
    df = read_file(filename)
    filtered_data = filter_taxa_depth(df)
    filtered_sumed_data = sum_abundance(filtered_data)
    unique_filtered_df(filtered_sumed_data)
    # Uncomment below if you want to save in tsv file the unique taxa found
    #unique_filtered_data = unique_filtered_df(filtered_data)
    #print(unique_filtered_data)
    #unique_filtered_filename = re.sub(r"^lf_", "", filename)[:-16] + "_unique_filtered.tsv"
    #save_filtered_data(unique_phlya_data, unique_filtered_filename)
    filtered_sumed_data_filename = re.sub(r"^lf_", "", filename)[:-16] + "_ge_filtered.tsv"
    save_filtered_data(filtered_sumed_data, filtered_sumed_data_filename)
    
    
if __name__ == "__main__":
    main()   