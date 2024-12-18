#!/usr/bin/python3.5

# script name: cc_studies_info_retrieval.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve taxonomic data for CC-related studies only. 
# framework: CCMRI
# last update: 13/12/2024

import os
import re 
import pandas as pd

def retrieve_cc_strudies_info(parent_dir, cc_studies_file, output_file_info):
    # Prepare a list to store data
    output_data = []
    # Load list with CC-related studies
    with open(cc_studies_file, "r") as f:
        cc_studies = set(line.strip() for line in f)
    # Walk through each folder in parent directory 
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        # Check if folder name is in cc studies file 
        if folder_name in cc_studies and os.path.isdir(folder_path):
            # Find all files in the directory that match the pattern 'secondaryAccession_taxonomy_abundances(?:_SSU|LSU)?.v.[d].[d].tsv'  
            matching_files = [file_name for file_name in os.listdir(folder_path) if re.match(r'\w+_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name) 
                              and not re.match(r'(.+)_phylum_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)]
            # Process each matching file 
            for file_name in matching_files:
                # Extract secondary accession code
                parts = file_name.split("_")
                if len(parts) > 4:  # Ensure filename has the expected structure
                    sec_accession_code = parts[0]
                    # Extract LSU/SSU and version
                    lsu_ssu_match = re.search(r'(LSU|SSU)_v\d+\.\d+', file_name)
                    if lsu_ssu_match:
                        version = lsu_ssu_match.group()  # This will extract LSU_vX.X or SSU_vX.X
                    else:
                        version = parts[-1].replace(".tsv", "")
                    # Read the file to extract sample IDs from the header row 
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as f:
                        # Read first line as header
                        header = f.readline().strip()
                        # Retrieve sample IDs 
                        sample_ids = header.split("\t")[1:]
                    # Combine folder name and secondary accession code for the first column 
                    combined_name = f"{folder_name}_{sec_accession_code}"
                    # Append data to the output list 
                    output_data.append([combined_name, version, ",".join(sample_ids)])
    # Convert the output data to a Data Frame 
    output_df = pd.DataFrame(output_data, columns=["Study_Second_Acc", "Version", "Samples_IDs"])
    # Save DataFrame to tsv file 
    output_df.to_csv(output_file_info, sep="\t", index=False)
    print(f"Output saved successfully to {output_file_info}")

def extract_cc_taxonomy_sample_counts(parent_dir, cc_studies_file, output_taxonomy_file):
    # Prepare a list to store data
    taxonomy_data = []
    # Load cc-related studies from the file 
    with open(cc_studies_file, "r") as f:
        cc_studies = set(line.strip() for line in f)
    # Loop through each folder in parent dir
    for folder_name in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder_name)
        # Process only if the folder matches a study name from cc_studies
        if folder_name in cc_studies and os.path.isdir(folder_path):
            # Find matching abundance files in the folder
            matching_files = [file_name for file_name in os.listdir(folder_path) if re.match(r'\w+_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name) 
                              and not re.match(r'(.+)_phylum_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)]
            # Process each matching file 
            for file_name in matching_files:
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, "r") as f:
                    # Read header to get sample IDs
                    header = f.readline().strip()
                    samples_ids = header.split("\t")[1:]
                    # REad the rest of the file for taxonomy and counts 
                    for line in f:
                        columns = line.strip().split("\t")
                        taxonomy = columns[0]
                        counts = columns[1:]
                        # Map taxonomy to each samples ID and count
                        for sample_id, count in zip(samples_ids, counts):
                            taxonomy_data.append([taxonomy, sample_id, count])
    # Save output to DataFrame 
    output_taxonomy_df = pd.DataFrame(taxonomy_data, columns=["Taxa", "Sample", "Count"])
    output_taxonomy_df.to_csv(output_taxonomy_file, sep="\t", index=False)
    print(f"Taxonomy Sample Count for CC-related studies saved to {output_taxonomy_file}")

def main():
    # Define parent dir 
    parent_dir = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"
    # Define CC_related studies file 
    cc_studies_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/cc_related/yes_studies_Dec_version.tsv"
    # Define output for study-version-sampleID info
    output_file_info = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/cc_related/studies_second_acc_samples.tsv"
    # Define output for extracted Taxonomy-SampleID-count info
    output_taxonomy_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/cc_related/cc_taxonomy_abundances.tsv"
    # Retrieve_cc_strudies_info(parent_dir, cc_studies_file, output_file_info)
    extract_cc_taxonomy_sample_counts(parent_dir, cc_studies_file, output_taxonomy_file)

if __name__ == "__main__":
    main()
