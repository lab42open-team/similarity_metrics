#!/usr/bin/python3.5

# script name: super_table_creation.py
# developed by: Nefeli Venetsianou
# description: create super table, containing all counts per taxon and sample, for each version and/or LSU/SSU type 
    # Data from all related tsv files (created with: taxa_counts.py script) are merged into one single tsv file (super table). 
    # In case count for a taxon in a sample equals 0, it is ignored (not written in the final output file).
    # User can choose to save output either in .tsv format (write_output function) or in .parquet format (write_parquet_output function).
    # Final format is long (3 columns): Taxa | Sample |Count
# input parameters: 
    # --input_folder="input_folder_name" (please pick one of the input folders prefered, 
    # eg. "v1.0", "v2.0", "v3.0", "v4.0_LSU", "v4.0_SSU", "v4.1_SSU", "v4.1_LSU", "v5.0_LSU", "v5.0_SSU" )
    # For now these lines are disabled (commented). Please uncomment them if you wish to use parameters via cmd. 
# framework: CCMRI

import os, sys
import time
import logging
logging.basicConfig(level=logging.INFO)
#import psutil # Uncomment this line if you want to retrieve total CPU usage 
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
#import numpy as np
#import zipfile # Uncomment this line if you want to create .zip file 

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
                        if (taxa, sample) not in taxa_counts:
                            # Initialize count
                            taxa_counts[(taxa, sample)] = 0
                        # Get counts per taxon in dictionary with tuple key  
                        taxa_counts[(taxa, sample)] += float(count)
        #print("Size of dictionary until {} is {}".format(file, sys.getsizeof(taxa_counts)))    
    end_time = time.time()
    execution_time = end_time - start_time
    print("Total number of unique taxa: {}".format(len(all_taxa)))
    logging.info("Execution time for merging data: {} seconds".format(execution_time))
                                            
    return all_sample_names, all_taxa, taxa_counts


def write_output(all_sample_names, all_taxa, taxa_counts, input_folder, output_dir):
    # Retrieve input folder name 
    input_folder_name = os.path.basename(input_folder) 
    # Construct output file name and path
    output_file_name = "lf_" + input_folder_name + "_super_table.tsv"
    output_file = os.path.join(output_dir, output_file_name) 
    sorted_sample_names = sorted(all_sample_names)
    sorted_taxa = sorted(all_taxa)
    with open(output_file, "w") as file:
        file.write("Taxa\tSample\tCount\n")
        for sample in sorted_sample_names:
            for taxa in sorted_taxa:
                count = taxa_counts.get((taxa, sample), 0)
                # Only write non zero counts 
                if count != 0:
                    file.write("{}\t{}\t{}\n".format(taxa, sample, count))
    ### Please uncomment below if you want to create .zip file
    # Create zip file containing tsv 
    #with zipfile.ZipFile(os.path.join(output_dir, output_file_name + ".zip"), "w") as zipf:
        #zipf.write(output_file, arcname=output_file_name)
    # Remove initial tsv file
    #os.remove(output_file)
              

def write_parquet_output(all_sample_names, all_taxa, taxa_counts, input_folder, output_dir):
    # Retrieve input folder name 
    input_folder_name = os.path.basename(input_folder) 
    # Construct output file name and path
    output_file_name = "lf_" + input_folder_name + "_super_table.parquet"
    output_file = os.path.join(output_dir,output_file_name) 
    # Create dictionary to save data
    data = {"Taxa": [], "Sample": [], "Count": []}            
    for taxa in sorted(all_taxa):
        for sample in sorted(all_sample_names):
            count = taxa_counts.get((taxa, sample), 0)
            # Only append non-zero counts 
            if count != 0:
                data["Taxa"].append(taxa)
                data["Sample"].append(sample)
                data["Count"].append(count)            
    # Convert dictionary to pandas dataframe
    df = pd.DataFrame(data)
    # Convert Dataframe to PyArrow Table
    table = pa.Table.from_pandas(df)
    # Write PyArrow Table to Parquet file
    pq.write_table(table, output_file)
    logging.info("Parquet output written seccessfully to {}.".format(output_file))    
            
def main():    
    # Extract argument values imported from sys.argv
    arguments_dict = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            sep = arg.find('=')
            key, value = arg[:sep], arg[sep + 1:]
            arguments_dict[key] = value
    # Set default input - output directories
    input_folder = "/ccmri/similarity_metrics/data/taxonomic/raw_data/v4.1_LSU"
    #input_dir = "/ccmri/similarity_metrics/data/taxonomic/taxa_counts_output"
    output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table"
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