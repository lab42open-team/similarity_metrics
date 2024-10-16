#!/usr/bin/python3.5

# script name: biome_retrieval_mined_info.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve biome from mined info txt files.
# framework: CCMRI
# last update: 08/10/2024

import os
import re 
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

def parse_study_data(file_path):
    # Initialize variables and list
    study_id = None
    biome_info = None
    samples = []
    # Read file
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            # Remove extra spaces or newlines
            line = line.strip()
            # Extract study_id, sample_id, biome_info
            if line.startswith("study_id"):
                study_id = line.split("\t")[1]
            if line.startswith("biome_info"):
                biome_info = line.split("\t")[1]
            if "sample_id" in line:
                sample_id = line.split("\t")[1]
                samples.append(sample_id)
    return study_id, samples, biome_info

def process_multiple_files(input_folder, output_file):
    # Open output file and write header
    with open(output_file, "w") as file: 
        file.write("study_id\tsample_id\tbiome_info\n")
        # Loop through all files in input folder
        for root, dirs, files in os.walk(input_folder):
            # Check if "COMPLETED" file exists and iterate over each directory where it exists
            if "COMPLETED" in files:
                for filename in files:
                    # Process only files named "mined_info.txt"
                    if filename.endswith(".txt"):
                    #if "mined_info" in filename:
                        file_path = os.path.join(root, filename)
                        # Parse each "mined_info.txt" file
                        study_id, samples, biome_info = parse_study_data(file_path)
                        # Write each sample along with the study_id and viome_info
                        for sample_id in samples:
                            file.write("{}\t{}\t{}\n".format(study_id, sample_id, biome_info))
    logging.info("Output saved successfully to: {}".format(output_file))

input_directory = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"
output_directory = "/ccmri/similarity_metrics/data/raw_data"
output_filename = "study_samples_biome_info.tsv"
output_file = os.path.join(output_directory, output_filename)
process_multiple_files(input_directory, output_file)        
