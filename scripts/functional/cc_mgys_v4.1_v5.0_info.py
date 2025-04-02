#!/usr/bin/python3.5

# Script name: cc_go_abundance_retrieval.py
# Developed by: Venetsianou Nefeli
# Description:
#   Retrieve GO abundance data for CC-related studies only (version v4.1 & v5.0).
# Framework: CCMRI
# Last update: 21/03/2025

import os
import re
import shutil

def retrieve_cc_go_abundance(parent_dir, cc_studies_file, output_dir):
    """Copy *_GO_abundances_*.tsv & *_GO-slim_abundances_*.tsv (v4.1, v5.0) for CC-related studies."""

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load list of CC-related studies
    with open(cc_studies_file, "r") as f:
        cc_studies = set(line.strip() for line in f)

    # Define regex pattern to match valid filenames with version v4.1 or v5.0
    file_pattern = re.compile(r'.*_(GO|GO-slim)_abundances_v(4\.1|5\.0)\.tsv$')

    # Walk through each study folder
    for study_name in os.listdir(parent_dir):
        study_path = os.path.join(parent_dir, study_name)

        # Check if the folder matches a CC-related study
        if study_name in cc_studies and os.path.isdir(study_path):
            # Search for matching files in the study folder
            for file_name in os.listdir(study_path):
                if file_pattern.match(file_name):
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
    output_dir = "/ccmri/similarity_metrics/data/cc_related_only/filtered_go_files"

    # Run function to retrieve matching files
    retrieve_cc_go_abundance(parent_dir, cc_studies_file, output_dir)

if __name__ == "__main__":
    main()