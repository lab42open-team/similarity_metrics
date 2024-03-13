#!/usr/bin/python3.5

# script name: version_pipeline.py
# developed by: Nefeli Venetsianou
# description: retrieve MGYS and corresponding version pipeline
# input parameters from cmd:
    # --input_dir="your_input_directory"
    # --output_dir="your_output_directory"
# framework: CCMRI

import os, re
import sys

# Set global variables to control script execution upon development (initialize parameters to default)
input_directory = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"
output_directory = "/ccmri/similarity_metrics/data"

# Extract argument values imported from sys.argv
arguments_dict = {}
for arg in sys.argv[1:]:
    if '=' in arg:
        sep = arg.find('=')
        key, value = arg[:sep], arg[sep + 1:]
        arguments_dict[key] = value
        
# Update parameters based on the values passed by the command line (if any)
if "--input_dir" in arguments_dict:
    input_directory = arguments_dict["--input_dir"]
if "--output_dir" in arguments_dict:
    output_directory = arguments_dict["--output_dir"]

# Set input working directory
input_wd = input_directory 

# Control existence of input working directory
if input_wd: 
    # List directories in the parent folder 
    all_folders = [folder for folder in os.listdir(input_wd) if os.path.isdir(os.path.join(input_wd, folder))]

    # Filter folders containing "MGYS" in their names
    target_folders = [folder for folder in all_folders if "MGYS" in folder]

    # Initialize an empty string to accumulate processed data
    all_processed_data = ""

    # Process each target folder 
    for folder_name in target_folders:
        # Construct the full path to the current folder
        folder_path = os.path.join(input_wd, folder_name)
		# Process in read file, only if complete file exists 
        input_file_path = os.path.join(folder_path, "COMPLETED")
        if os.path.exists(input_file_path):
            with open(input_file_path, "r") as input_file:
                input_data = input_file.read()
     
                # Find all files in the directory that match the pattern 'secondaryAccession_taxonomy_abundances.v.[d].[d].tsv'  
                matching_files = [file_name for file_name in os.listdir(folder_path) if re.match(r'\w+_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)
                                  and not re.match(r'(.+)_phylum_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)]            
                # Check if there are matching files
                if matching_files:  
                    # Extract version pipeline for each study
                    version_pipeline = [re.search(r'((?:_SSU|_LSU)?_v\d+\.\d+)', file_name).group(0) for file_name in matching_files]
                    processed_data = folder_name + "\t" + "\t".join(version_pipeline)
                    
                    # Append the processed data to the accumulator string
                    all_processed_data += processed_data + "\n"
                    
                else:
                    print("No matching files found in", folder_path)
        else:
            print("Input file not found in", folder_path)
else:
    print("No input directory selected.")
    
# Set output working directory 
output_wd = output_directory
# Write all accumulated processed data into the output file in the selected directory
if output_wd:
    output_file_path = os.path.join(output_wd, "study_version.tsv")
    with open(output_file_path, "w") as output_file:
        output_file.write(all_processed_data)
    print("Processed data written to", output_file_path)
else:
    print("No output directory selected.")