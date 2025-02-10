#!/usr/bin/python3.5

# script name: super_table_creation.py
# developed by: Nefeli Venetsianou
# description: create super table, containing all counts per GO term and run, for each version.
    # In case count for a GO term in a run equals 0, it is ignored (not written in the final output file). Normally, this is not going to happen, since MGnify's GO-slim files are used, so counts are already supposed to be aggregated.
    # User can choose to save output either in .tsv format (write_output function) or in .parquet format (write_parquet_output function). Zip file format is also provided.
    # Final format is long (3 columns): GO Term | Run |Count
# input parameters: 
    # --input_folder="input_folder_name" (please pick one of the input folders prefered, 
    # eg. "v1.0", "v2.0", "v3.0", "v4.0", "v4.1", "v5.0")
    # For now these lines are disabled (commented). Please uncomment them if you wish to use parameters via cmd. 
# framework: CCMRI
# last update: 22/01/2025

import os, sys
import time
import logging
logging.basicConfig(level=logging.INFO)
#import psutil # Uncomment this line if you want to retrieve total CPU usage 
#import pyarrow as pa
#import pyarrow.parquet as pq
import pandas as pd
#import numpy as np
#import zipfile # Uncomment this line if you want to create .zip file 

def merge_functional_data(input_folder, skipped_files_output):
    """Merge functional data from all TSV files in the input folder"""
    all_go_terms = set()
    f_counts = {}
    total_files = 0 
    skipped_files = 0
    
    start_time = time.time() # record start execution time
    
    with open(skipped_files_output, "a") as skipped_files_log:
        skipped_files_log.write("Skipped files due to unexpected format:\n")
        for file in os.listdir(input_folder):
            if file.endswith(".tsv"):
                total_files += 1 # Increment total files counter
                # Logging debug
                logging.debug(".tsv files not found")
                target_file = os.path.join(input_folder, file)
                logging.debug("Processing file: {}".format(target_file))
                # Read file into Pandas DataFrame
                df = pd.read_csv(target_file, sep="\t")
                # Validate column structure
                if len(df.columns) < 4:
                    logging.warning("Unexpected format in file: {}. Skipping file.".format(file))
                    skipped_files += 1 # Increment skipped files counter 
                    skipped_files_log.write("{}\n".format(file))  # Log skipped file name
                    continue
                # Extract first column (GO term)
                go_col = df.columns[0]
                # Extract run columns - from the 4th col and onward
                runs_col = df.columns[3:]
                for _, row in df.iterrows():
                    go = row[go_col]
                    all_go_terms.add(go)
                    # Populate counts for each run
                    for run in runs_col:
                        count = row[run]
                        if count > 0:
                            f_counts[(go, run)] = count
    end_time = time.time() # record end execution time 
    logging.info("Execution time for mergine data: {} seconds.".format(end_time-start_time))
    logging.info("Total unique GO terms: {}".format(len(all_go_terms)))
    logging.info("Total files in input folder: {} \n Skipped files: {}".format(total_files, skipped_files))
    return all_go_terms, f_counts

def write_output(f_counts, input_folder, output_dir):
    # Retrieve input folder name 
    input_folder_name = os.path.basename(input_folder)
    output_file_name = "lf_{}_super_table.tsv".format(input_folder_name)
    output_file = os.path.join(output_dir, output_file_name)
    with open(output_file, "w") as file:
        file.write("GO\tRun\tCount\n")
        # Iterate through all counts and write non-zero values
        for (go, run), count in sorted(f_counts.items()):
            file.write("{}\t{}\t{}\n".format(go, run, count))

def main():
    parent_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/raw_data"
    input_folder_name = "v4.0" # To be adjusted  
    input_folder = os.path.join(parent_dir, input_folder_name)
    output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/super_table"
    skipped_input_files = os.path.join(output_dir, "skipped_files_log.txt")
    # Merge data 
    merge_data, f_counts = merge_functional_data(input_folder, skipped_input_files)
    # Write output 
    write_output(f_counts, input_folder, output_dir)

if __name__ == "__main__":
    main()
            