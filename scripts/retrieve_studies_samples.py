#!/usr/bin/python3.5

# script name: retrieve_studies_samples.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve Study name and samples from taxonomy_abundances tsv file.
# framework: CCMRI
# last update: 11/07/2024

import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def process_file(file, folder_path):
    # Iterate over files with taxonomy abundances in them
    if "taxonomy_abundances" in file:
        # Extract desired pattern and define file path 
        pattern = file.split("_")[0]
        file_path = os.path.join(folder_path, file)
        logging.debug("Processing file: {}".format(file_path))
        try:
            # Read tsv file content
            df = pd.read_csv(file_path, sep="\t")
            # Extract samples from header - exclude column 1 where #SampleID is written
            sample_names = df.columns.to_list()[1:]
            # Prepare data to return
            data = {"MGYS_ID": [pattern] * len(sample_names), "Sample_Name": sample_names}
            return pd.DataFrame(data)
        except Exception as e:
            logging.error("Error processing file {}: {}".format(file, e))
            return pd.DataFrame()  # Return an empty DataFrame in case of error

def main():
    parent_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/taxonomy_abundances"
    output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples"
    folder_name = "v5.0_LSU"

    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Define folder path
    folder_path = os.path.join(parent_dir, folder_name)
    if os.path.isdir(folder_path):
        logging.info("Processing folder: {}".format(folder_name))
        # List of files contained in folder
        files = os.listdir(folder_path)
        # Initialize an empty DataFrame to accumulate results
        combined_df = pd.DataFrame()  # Initialize an empty DataFrame to accumulate results
        # Iterate over each file
        for file in files:
            df = process_file(file, folder_path)
            # Accumulate results
            combined_df = pd.concat([combined_df, df], ignore_index=True)  
        # Save the combined DataFrame to a single TSV file
        output_filename = "studies_samples_{}.tsv".format(folder_name)
        combined_df.to_csv(os.path.join(output_dir, output_filename), sep="\t", index=False)
        logging.info("Output saved to {}".format(output_filename))
    else:
        logging.error("Folder {} does not exist".format(folder_name))

if __name__ == "__main__":
    main()
