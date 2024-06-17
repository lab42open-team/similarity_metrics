#!/usr/bin/python3.5

# script name: isolate_abundances_file.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve only taxonomy_abundances_files and save in different folder, with MGYS name. 
# input parameters from cmd:
    # --input_dir="your_input_directory"
    # --output_dir="your_output_directory"
# framework: CCMRI

import os, re, shutil
import logging
logging.basicConfig(level=logging.INFO)


# Set global variables to control script execution upon development (initialize parameters to default)
input_directory = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"
#input_directory = "/ccmri/similarity_metrics/data/test_dataset/raw_data/MGYS" # test 
#output_directory = "/ccmri/similarity_metrics/data/test_dataset/raw_data" # test
output_directory = "/ccmri/similarity_metrics/data/raw_data"

# Define target folders
target_folders = [folder for folder in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, folder))]

# Initialize count of matching files
num_matching_files = 0

# Process each target folder 
for folder_name in target_folders:
    # Construct the full path to the current folder
    folder_path = os.path.join(input_directory, folder_name)
    # Process in read file, only if completed file exists 
    input_file_path = os.path.join(folder_path, "COMPLETED")
    # Check if completed file exiests
    if os.path.exists(input_file_path):
        # Find all files in the directory that match the pattern 'secondaryAccession_taxonomy_abundances(?:_SSU|LSU)?.v.[d].[d].tsv'  
        matching_files = [file_name for file_name in os.listdir(folder_path) if re.match(r'\w+_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name) 
                          and not re.match(r'(.+)_phylum_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)]
        # Increase count of matching files
        num_matching_files += len(matching_files)
        # Check if there are matching files
        if matching_files:
            # Process each matching file 
            for file_name in matching_files:
                file_path = os.path.join(folder_path, file_name)
                
                # Check output directory
                if output_directory: 
                    # Construct output file name with MGYS study name as prefix
                    output_file_name = folder_name + "_" + file_name
                    # construct output file path 
                    output_file_path = os.path.join(output_directory, output_file_name)
                    # check if output already exists
                    if os.path.exists(output_file_path):
                        logging.warning("Output file already exists for: {}".format(file_name))
                        continue
                    # Copy the file to the output directory with the new name 
                    shutil.copy(file_path, output_file_path)
                    logging.info("Copied: {} to {}".format(file_name, output_file_name))
# Log total number of matching files (file processed)
logging.info("Total number of matching files: {}".format(num_matching_files))