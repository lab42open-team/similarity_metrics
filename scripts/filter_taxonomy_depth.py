#!/usr/bin/python3.5

# script name: filter_taxonomy_depth.py
# developed by: Nefeli Venetsianou
# description: 
    # Filter data to keep only the taxonomic level of choice.
    # Removes suffix after the taxonomic level of choice, but keeps prefix to know the hierarchy. 
    # Output saved automatically and user can choose (optionally) to save unique phyla found is a separate .tsv file.
# Input parameters: filename of super table to be analyzed (eg. lf_v1.0_super_table.tsv)
# framework: CCMRI
# last update: 24/05/2024

import sys, os, re
import pandas as pd

parent_directory = "/ccmri/similarity_metrics/data/lf_super_table/no_filtered/"
output_directory = "/ccmri/similarity_metrics/data/lf_super_table/phylum_filtered/"

def read_file(filename):
    file_path = os.path.join(parent_directory, filename)
    # check is file exists and is a tsv file
    if not os.path.exists(file_path) or not file_path.endswith(".tsv"):
        print("File not found or not a .tsv file. Please provide a valid file. If you wish, check parent directory '{}' for available input files.".format(parent_directory))
        sys.exit(1)
    return pd.read_csv(file_path, sep="\t")

def filter_taxa_depth(df):
    # Filter rows where Taxa contains "p__"
    phylum_df = df[df["Taxa"].str.contains("p__")].copy() 
    # Remove every taxon after phylum, using semicolon separator
    phylum_df.loc[:, "Taxa"] = phylum_df["Taxa"].str.replace(r"(p__[^;]*);.*", r"\1",regex = True)
    return phylum_df

def unique_phyla_df(df):
    unique_phyla_df = df.drop_duplicates(subset=["Taxa"])
    unique_phyla_count = unique_phyla_df.shape[0]
    print("Number of unique phyla: {}".format(unique_phyla_count))
    #return unique_phyla_df # uncomment if you want to save output 

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
    unique_phyla_df(filtered_data)
    # Uncomment below if you want to save in tsv file the unique taxa found
    #unique_phlya_data = unique_phyla_df(filtered_data)
    #unique_phyla_filename = re.sub(r"^lf_", "", filename)[:-16] + "_unique_phyla.tsv"
    #save_filtered_data(unique_phlya_data, unique_phyla_filename)
    filtered_data_filename = re.sub(r"^lf_", "", filename)[:-16] + "_outfiltered.tsv"
    save_filtered_data(filtered_data, filtered_data_filename)
    
    
if __name__ == "__main__":
    main()   