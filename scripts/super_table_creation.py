#!/usr/bin/python3.5

# script name: super_table_creation.py
# developed by: Nefeli Venetsianou
# description: create super table, containing all counts per taxon and sample, for each version and/or LSU/SSU type 
    # Data from all related tsv files (created with: taxa_counts.py script) are merged into one single tsv file (super table). 
    # In case a taxon doesn't exist in a sample, count is considered zero (0).
# input parameters: For now, no input parameters are allowed. 
# framework: CCMRI

import os, sys 
import time
import logging
logging.basicConfig(level=logging.INFO)

def merge_data(input_dir):
    all_taxa = set()
    all_sample_names =set()
    taxa_counts = {}
    
    # Record start time
    start_time = time.time()
    
    # Iterate through all files in parent directory
    for file in os.listdir(input_dir):
        # Logging debug
        logging.debug("Input directory not defined")
        if file.endswith(".tsv"):
            # Logging debug
            logging.debug(".tsv files not found")
            target_file = os.path.join(input_dir, file)
            logging.debug("Processing file: {}".format(target_file))
            with open(target_file, "r") as file:
                # Skip header
                header = next(file).strip()
                # Extract sample names from header
                sample_names = header.split("\t")[1:]
                # Keep unique sample names 
                all_sample_names.update(sample_names)
                                    
                # Process each line in the file 
                for line in file:
                    columns = line.strip().split("\t")
                    # Define taxa
                    taxa = columns[0]
                    # Define counts
                    counts = columns[1:]
                    # Update set of all taxa
                    all_taxa.add(taxa)
                    
                    # Update taxa counts dictionary - zip function to pair the elements of the two iterables to produce tuple 
                    for sample, count in zip(sample_names, counts):
                        # Get counts per taxon in dictionary with tuple key - if pair exists > return count, else return 0 
                        taxa_counts[(taxa, sample)] = taxa_counts.get((taxa, sample), 0) + float(count)
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info("Execution time: {} seconds".format(execution_time))
                                            
    return all_sample_names, all_taxa, taxa_counts


def write_output(all_sample_names, all_taxa, taxa_counts, output_dir):
    # Construct output file name
    #output_file_name = os.path.join(output_dir, "output.tsv") 
    with open(output_dir, "w") as file:
        file.write("\t" + "\t".join(sorted(all_sample_names)) + "\n")
        for taxa in sorted(all_taxa):
            file.write(taxa)
            for sample in sorted(all_sample_names):
                file.write("\t")
                # Write counts per taxon and sample, complete with 0 if taxon not exist
                file.write(str(taxa_counts.get((taxa, sample), 0)))
            file.write("\n")        
            
def main():    
    #input_dir = "/ccmri/similarity_metrics/data/version_5.0/v5_LSU" # input - LSU version 5.0 
    #output_dir = "/ccmri/similarity_metrics/data/version_5.0/v5.0_LSU_super_table.tsv" # output - LSU version 5.0
    input_dir = "/ccmri/similarity_metrics/data/version_5.0/v5_SSU" # input - SSU version 5.0
    output_dir = "/ccmri/similarity_metrics/data/version_5.0/v5.0_SSU_super_table.tsv" # output - SSU version 5.0
    #test # input_dir = "/ccmri/similarity_metrics/data/test_dataset"
    #test # output_dir = "/ccmri/similarity_metrics/data/test_dataset"
    start_time = time.time()
    version_sample_names, version_taxa, version_taxa_counts = merge_data(input_dir)
    write_output(version_sample_names, version_taxa, version_taxa_counts, output_dir)
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info("Total execution time: {} seconds".format(execution_time))

if __name__ == "__main__":
    main()         