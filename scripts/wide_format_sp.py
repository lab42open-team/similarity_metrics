#!/usr/bin/python3.5

# script name: super_table_creation.py
# developed by: Nefeli Venetsianou
# description: create super table, containing all counts per taxon and sample, for each version and/or LSU/SSU type 
    # Data from all related tsv files (created with: taxa_counts.py script) are merged into one single tsv file (super table). 
    # In case a taxon doesn't exist in a sample, count is considered zero (0).
# input parameters: 
    # --input_folder="input_folder_name" (please pick one of the input folders prefered, 
    # eg. "v1.0", "v2.0", "v3.0", "v4.0_LSU", "v4.0_SSU", "v4.1_SSU", "v4.1_LSU", "v5.0_LSU", "v5.0_SSU" )
# framework: CCMRI

import os, sys
import time
import logging
logging.basicConfig(level=logging.INFO)
#import psutil # Uncomment this line if you want to retrieve total CPU usage 

def merge_data(input_folder):
    all_taxa = set()
    all_sample_names =set()
    taxa_counts = {}
    
    # Record start time
    start_time = time.time()
    
    # Iterate through all files in parent directory
    for file in os.listdir(input_folder):
        # Logging debug
        logging.debug("Input directory not defined")
        if file.endswith(".tsv"):
            # Logging debug
            logging.debug(".tsv files not found")
            target_file = os.path.join(input_folder, file)
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
        #print("Size of dictionary until {} is {}".format(file, sys.getsizeof(taxa_counts)))    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Number of unique taxa: {}".format(len(all_taxa)))
    logging.info("Execution time for merging data: {} seconds".format(execution_time))
                                            
    return all_sample_names, all_taxa, taxa_counts


def write_output(all_sample_names, all_taxa, taxa_counts, input_folder, output_dir):
    # Retrieve input folder name 
    input_folder_name = os.path.basename(input_folder) 
    # Construct output file name and path
    output_file_name = "wf_" + input_folder_name + "_super_table.tsv"
    output_file = os.path.join(output_dir,output_file_name) 
    with open(output_file, "w") as file:
        file.write("\t" + "\t".join(sorted(all_sample_names)) + "\n")
        for taxa in sorted(all_taxa):
            file.write(taxa)
            for sample in sorted(all_sample_names):
                file.write("\t")
                # Write counts per taxon and sample, complete with 0 if taxon not exist
                file.write(str(taxa_counts.get((taxa, sample), 0)))
            file.write("\n")        
            
def main():    
    # Extract argument values imported from sys.argv
    arguments_dict = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            sep = arg.find('=')
            key, value = arg[:sep], arg[sep + 1:]
            arguments_dict[key] = value
    # Set default input - output directories
    input_folder = "/ccmri/similarity_metrics/data/test_dataset/test_folder"
    #input_dir = "/ccmri/similarity_metrics/data/taxa_counts_output/"
    output_dir = "/ccmri/similarity_metrics/data/SuperTable/wide_format"
           
    """
    # Update parameters based on the values passed by the command line 
    input_folder = arguments_dict.get("--input_folder")
    if not input_folder:
        input_folder = input("Please provide input folder name, eg. v5.0_LSU: ")
    if input_folder:
        input_folder = os.path.join(input_dir, input_folder)
    else:
        logging.warning("No input file provided.")
        sys.exit(1)            
    """
    # Record total start time 
    tot_start_time = time.time()    
    
    # Perform merge data
    version_sample_names, version_taxa, version_taxa_counts = merge_data(input_folder)
        
    # Write output file
    write_output(version_sample_names, version_taxa, version_taxa_counts, input_folder, output_dir)    
    logging.info("Output written successfully to: {}".format(output_dir))
   
    # Record total end time  
    tot_end_time = time.time()
    
    # Calculate total execution time
    tot_execution_time = tot_end_time - tot_start_time
    logging.info("Total execution time is {} seconds".format(tot_execution_time))
    
    # Get current CPU usage for 5 seconds - interval time can be adjusted 
    #logging.info("CPU usage = ", psutil.cpu_percent(interval = 5)) # Uncomment this line if you want to retrieve total CPU usage

if __name__ == "__main__":
    main()         