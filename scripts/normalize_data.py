#!/usr/bin/python3.5

# script name: normalize_data.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate relative abundances in raw data with process_raw_data function.
    # Calculate relative abundances in filtered data with process_filtered_data function.
    # Please make sure you use the correct input-output directories according to the data folder you wish to proceed. 
# framework: CCMRI
# last update: 17/06/2024

import os
import pandas as pd
import re
import time
import logging
logging.basicConfig(level=logging.INFO)

def process_raw_data(input_file):
   # Read tsv file into a DataFrame
    df = pd.read_csv(input_file, delimiter="\t", index_col=0, engine="python")
    # Replace null values with 0 - if any
    df.fillna(0, inplace=True)    
    # Initialize DataFrame to store relative aundances
    relative_abundance_df = pd.DataFrame(index=df.index)
    # Iterate over columns (samples) in dataframe 
    for sample in df.columns:
        # Calculate total counts per sample
        total_counts = df[sample].sum()
        # Calculate relative abundance for each taxa in the sample
        relative_abundance = df[sample] / total_counts
        # Store relative abundance for each taxa in sample counts dictionary
        relative_abundance_df[sample] = relative_abundance 
    return relative_abundance_df

def process_filtered_data(input_file):
    # Read tsv file into a DataFrame
    df = pd.read_csv(input_file, delimiter="\t", engine="python")
    # Initialize dictionary to store total count per sample
    total_count = df.groupby("Sample")["Count"].sum().to_dict()
    # Calculate relative abundance 
    df["Count"] = df.apply(lambda row: row["Count"] / total_count[row["Sample"]], axis=1)
    return df

def write_output(output_dir, study_name, input_file, data_frame):
    # Extract file name from input file 
    file_name = os.path.basename(input_file)
    # IMPORTANT # For process_raw_data:
    # Construct output file path with study name + file name - uncomment below
    #output_file = os.path.join(output_dir, study_name + "_" + file_name)
    # Save DataFrame with relative abundances to a new tsv file
    # IMPORTANT # For process_filtered_data:
    # Construct output file path with ra + file_name - uncomment below accordingly
    output_file = os.path.join(output_dir, "ra_" + file_name)
    
    data_frame.to_csv(output_file, sep="\t")
    logging.info("Output saved to: {}".format(output_file))

def normalization(input_file, output_dir):
    # Process data 
    # Please select here if you want to proceed with raw data or filtered data
    relative_abundance_df = process_filtered_data(input_file)
    # Extract the study name from the parent directory of the input file 
    study_name = os.path.basename(os.path.dirname(input_file))
    # Write output
    write_output(output_dir, study_name, input_file, relative_abundance_df)
    
# IMPORTANT # If normalization in raw data is selected (process_raw_data), please uncomment below   
"""
def process_raw_folders(parent_dir, output_dir):
    # List directories in the parent folder 
    all_folders = [folder for folder in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, folder))]
    # Filter folders containing "MGYS" in their names
    target_folders = [folder for folder in all_folders if "MGYS" in folder]
    # Counter of matching files
    num_matching_files = 0
    # Process each target folder 
    for folder_name in target_folders:
        # Construct the full path to the current folder
        folder_path = os.path.join(parent_dir, folder_name)
		# Process in read file, only if completed file exists 
        input_file_path = os.path.join(folder_path, "COMPLETED")
        # Check if the file specified exists
        if os.path.exists(input_file_path):
            # Open file
            with open(input_file_path, "r") as input_file:     
                # Find all files in the directory that match the pattern 'secondaryAccession_taxonomy_abundances(?:_SSU|LSU)?.v.[d].[d].tsv'  
                matching_files = [file_name for file_name in os.listdir(folder_path) if re.match(r'\w+_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)
                                  and not re.match(r'(.+)_phylum_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)]
                # Count number of total matching files
                num_matching_files += len(matching_files)           
                # Check if there are matching files
                if matching_files:
                    # Process each matching file
                    for file_name in matching_files:
                        input_file = os.path.join(folder_path, file_name)
                        logging.info("Processing file: {}".format(input_file))
                        normalization(input_file, output_dir)           
                else:
                    logging.warning("No matching files found in {}.".format(folder_name))
    logging.info("Number of matching files: {}".format(num_matching_files))             
"""

def main():
    # IMPORTANT # If process_raw_data, then uncomment below:
    """
    input_dir = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"
    output_dir = "/ccmri/similarity_metrics/data/normalized_raw_data" 
    # Process normalization for directory 
    process_folders(input_dir, output_dir)
    """
    # IMPORTANT # If process_filtered_data, then:
    input_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/v4.1_LSU_ge_filtered.tsv" # add super table of choice
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/normalized_counts/"
    
    # Record start time
    start_time = time.time()
    # Process data to normalization 
    normalization(input_file, output_dir)
    # Record end time
    end_time = time.time()
    # Calculate execution time 
    execution_time = end_time - start_time
    logging.info("Execution time is {} seconds.".format(execution_time))
    

if __name__ == "__main__":
    main() 