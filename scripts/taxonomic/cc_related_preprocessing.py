#!/usr/bin/python3.5

# Script name: cc_taxonomy_abundance_extract_info.py
# Developed by: Venetsianou Nefeli
# Description:
#   Extract secondary accession and version for taxonomy abundance data (v4.1 & v5.0) for CC-related studies.
# Framework: CCMRI
# Last update: 01/04/2025

import os
import re
import pandas as pd

def extract_cc_go_abundance_info(parent_dir, cc_studies_file, output_tsv):
    """Extract secondary accession, version, and LSU/SSU for *_taxonomy_abundances_*.tsv (v4.1, v5.0)."""
    
    # Prepare list to store extracted data
    extracted_data = []

    # Load list of CC-related studies
    with open(cc_studies_file, "r") as f:
        cc_studies = set(line.strip() for line in f)

    # Regex pattern to extract secondary accession, version, and LSU/SSU
    file_pattern = re.compile(r'(?P<sec_accession>.+)_taxonomy_abundances(?:_(?P<rrna>SSU|LSU))?_v(?P<version>\d+\.\d+)\.tsv$')
    
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
                    sec_accession = match.group("sec_accession")  # Everything before _taxonomy_abundances_
                    version = match.group("version") 
                    rrna = match.group("rrna") if match.group("rrna") else "N/A"  # Store LSU/SSU or "N/A"

                    extracted_data.append([study_name, sec_accession, version, rrna])

    # Convert extracted data to DataFrame and save to TSV
    df = pd.DataFrame(extracted_data, columns=["Study", "Secondary_Accession", "Version", "LSU_SSU"])
    df.to_csv(output_tsv, sep="\t", index=False)
    print(f"Extracted data saved successfully to {output_tsv}")

def main():
    # Define parent directory
    parent_dir = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"

    # Define file with CC-related study names
    cc_studies_file = "/ccmri/similarity_metrics/data/cc_related_only/mgys_cc.tsv"

    # Define output TSV file
    output_tsv = "/ccmri/similarity_metrics/data/cc_related_only/taxonomic/info_all_cc_taxonomy_files.tsv"

    # Run function to extract information
    extract_cc_go_abundance_info(parent_dir, cc_studies_file, output_tsv)

if __name__ == "__main__":
    main()
