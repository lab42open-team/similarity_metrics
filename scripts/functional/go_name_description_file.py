#!/usr/bin/python3.5

# script name: go_name_description_file.py
# developed by: Nefeli Venetsianou
# description: create super table containing unique GO terms with their description and name

import os, sys
import time
import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd

def merge_functional_data(input_folder):
    """Merge functional data from all TSV files in the input folder keeping only the unique GO terms with their description and name"""
    unique_go_terms = {}
    total_files = 0 
    skipped_files = 0
    
    start_time = time.time() # record start execution time
    
    for file in os.listdir(input_folder):
        if file.endswith(".tsv"):
            total_files += 1 # Increment total files counter
            logging.debug("Processing file: {}".format(file))
            target_file = os.path.join(input_folder, file)
            # Read file into Pandas DataFrame
            df = pd.read_csv(target_file, sep="\t")
            # Validate column structure
            if len(df.columns) < 3:
                logging.warning("Unexpected format in file: {}. Skipping file.".format(file))
                skipped_files += 1 # Increment skipped files counter 
                continue
            # Extract the first three columns (GO term, description, name)
            for _, row in df.iterrows():
                go = row[0]   # GO term
                description = row[1]  # description
                category = row[2]  # category
                if go not in unique_go_terms:
                    unique_go_terms[go] = (description, category)  # Store description and name for each unique GO term

    end_time = time.time() # record end execution time 
    logging.info("Execution time for merging data: {} seconds.".format(end_time-start_time))
    logging.info("Total unique GO terms: {}".format(len(unique_go_terms)))
    logging.info("Total files in input folder: {} \n Skipped files: {}".format(total_files, skipped_files))
    return unique_go_terms

def write_output(unique_go_terms, input_folder, output_dir):
    # Retrieve input folder name 
    input_folder_name = os.path.basename(input_folder)
    output_file_name = "unique_go_terms_{}.tsv".format(input_folder_name)
    output_file = os.path.join(output_dir, output_file_name)
    
    with open(output_file, "w") as file:
        file.write("GO\tDescription\tCategory\n")
        # Iterate through unique GO terms and write them to output file
        for go, (description, name) in sorted(unique_go_terms.items()):
            file.write("{}\t{}\t{}\n".format(go, description, name))

def main():
    parent_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/raw_data"
    input_folder_name = "v5.0"  # To be adjusted  
    input_folder = os.path.join(parent_dir, input_folder_name)
    output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances"
    
    # Merge data to get unique GO terms with description and name
    unique_go_terms = merge_functional_data(input_folder)
    
    # Write output
    write_output(unique_go_terms, input_folder, output_dir)

if __name__ == "__main__":
    main()