#!/usr/bin/python3.5

# Script name: cc_taxonomy_abundances_retrieval.py
# Developed by: Venetsianou Nefeli
# Description:
#   Retrieve taxonomy abundances data for CC-related studies only (version v4.1 & v5.0).
# Framework: CCMRI
# Last update: 01/04/2025

import os
import re
import shutil

def retrieve_cc_go_abundance(parent_dir, cc_studies_file, output_dir):
    """Copy taxonomy abundances files (v4.1, v5.0) for CC-related studies."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load list of CC-related studies
    with open(cc_studies_file, "r") as f:
        cc_studies = set(line.strip() for line in f)

    # Regex pattern to extract secondary accession, version, and LSU/SSU
    file_pattern = re.compile(r'(?P<sec_accession>.+)_taxonomy_abundances(?:_(?P<rrna>SSU|LSU))?_v(?P<version>4\.1|5\.0)\.tsv$')    
    # Regex pattern to exclude "phylum_taxonomy_abundances" files
    exclude_pattern = re.compile(r'.*_phylum_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv$')

    # Walk through each study folder
    for study_name in os.listdir(parent_dir):
        study_path = os.path.join(parent_dir, study_name)

        # Check if the folder matches a CC-related study
        if study_name in cc_studies and os.path.isdir(study_path):
            # Search for matching files in the study folder
            for file_name in os.listdir(study_path):
                if exclude_pattern.match(file_name):
                    continue  # Skip files with "phylum_taxonomy_abundances"
                
                match = file_pattern.match(file_name)
                if match:
                    src_path = os.path.join(study_path, file_name)
                    dest_path = os.path.join(output_dir, file_name)
                    shutil.copy(src_path, dest_path)
                    print(f"Copied: {file_name} from {study_name}")

    print(f"All matching files (v4.1 & v5.0) have been copied to {output_dir}")

def main():
    # Define parent directory containing all studies
    parent_dir = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"

    # Define file with CC-related study names
    cc_studies_file = "/ccmri/similarity_metrics/data/cc_related_only/mgys_cc.tsv"

    # Define output directory for copied files
    output_dir = "/ccmri/similarity_metrics/data/cc_related_only/taxonomic/raw_data"

    # Run function to retrieve matching files
    retrieve_cc_go_abundance(parent_dir, cc_studies_file, output_dir)

if __name__ == "__main__":
    main()