#!/usr/bin/python3.5

# script name: per_taxon_calculation.py
# developed by: Nefeli Venetsianou
# description: retrieve counts of specific taxon of interest per sample from MGYS 
# input parameters: 
    # input_dir="your_input_directory"
    # study_name="MGYS000xxxxxx"
    # taxon_of_interest="taxon"
# framework: CCMRI

import os, sys, time
import logging
logging.basicConfig(level=logging.INFO)

def search_taxon(input_dir, study_name, taxon_of_interest):
    # Search for files in parent directory containing the study name provided in the file name
    target_file = [file for file in os.listdir(input_dir) if study_name in file]
    if not target_file:
        logging.error("No files found for the study name provided.")
    else:
        # Iterate for file containing study name
        with open(os.path.join(input_dir, target_file[0]), "r") as file:
            logging.info("Counts of {} in {} :".format(taxon_of_interest, target_file[0]))
            header_printed = False
            # Keep track of unique taxa identified - initialize set
            unique_taxa = set()
            for line in file:
                columns = line.strip().split("\t")
                if not header_printed:
                    logging.info("\t".join(columns))
                    header_printed = True
                if taxon_of_interest in columns[0]:
                    taxon_counts = "\t".join(columns[1:])
                    logging.info("{}\t{}".format(columns[0], taxon_counts))
                    """ # Uncomment below if you want to print the unique taxa found in total in the MGYS provided. 
                    # Extract taxon from the first column
                    taxa = columns[0].split(";")[0]
                    unique_taxa.add(taxa)
            # Print unique taxa found
            logging.info("Unique taxa fount: {}".format(",".join(unique_taxa)))  
            """
if __name__ == "__main__":
    # Set default parameters
    # input_dir = "/ccmri/profile_matching/test_dataset/MGYS_taxa_counts_output" - test 
    input_dir = "/ccmri/data_similarity_metrics/taxa_counts_output"
    study_name = ""
    taxon_of_interest = ""
    
    # Extract arguments imported from sy.argv via command line
    arguments_dict = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            sep = arg.find('=')
            key, value = arg[:sep], arg[sep + 1:]
            arguments_dict[key] = value
        
    # Update parameters based on the values passed by the command line (if any)
    if "--input_dir" in arguments_dict:
        input_dir = arguments_dict["--input_dir"]
    if "--study_name" in arguments_dict:
        study_name = arguments_dict["--study_name"]
    if "--taxon_of_interest" in arguments_dict:
        taxon_of_interest = arguments_dict["--taxon_of_interest"]    
    
    # Call function
    if study_name and taxon_of_interest:
        search_taxon(input_dir, study_name, taxon_of_interest)
    else:
        logging.warning("Please provide both study name and taxon of interest")