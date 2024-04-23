#!/usr/bin/python3.5

# script name: ra_aggregation
# developed by: Nefeli Venetsianou
# description: 
    # Get taxa counts per sample in study (Taxonomic hierarchy: k > p > c > o > f  > g > s). 
    # Works for normalized data (relative abundances calculated via normalization_counts.py).
# input parameters from cmd:
    # --input_dir="your_input_directory"
    # --output_dir="your_output_directory"
    # --ignore_prefix="true" or "yes" or "1" - set this true if you want to ignore not fully specified taxa  
# framework: CCMRI
# last update: 22/04/2024

import os, sys, time
#import psutil # Uncomment this line if you want to retrieve total CPU usage 
import logging
logging.basicConfig(level=logging.INFO)

# Set global variables to control script execution upon development (initialize parameters to default)
input_directory = "/ccmri/similarity_metrics/data/normalized_raw_data" 
#input_directory = "/ccmri/similarity_metrics/data/test_dataset/raw_data/test_ra" # only for testing
output_directory = "/ccmri/similarity_metrics/data/aggregated_data/ra_taxa_counts_output"
#output_directory = "/ccmri/similarity_metrics/data/test_dataset/raw_data/aggregated_ra" # only for testing
ignore_prefix = False 

def count_taxa_occurrences(file_path, ignore_prefix):
    # Logging debug
    logging.debug("Parsing file: {}".format(file_path))
    # Store counts for each taxon into a dictionary - initialize dic
    taxon_counts = {}
    header = None
    sample_names = []
    # Open file
    with open(file_path, "r") as file:
        # Retrieve first line as header - remove whitespaces - split columns using tab delimiter  
        header = next(file).strip()
        sample_names = header.split("\t")[1:]
        # Set line number 
        line_number = 2
        # Initialize previous prefix and previous taxon variables
        prev_prefix = None
        prev_taxon = None
        # Iterate for each line in file
        for line in file:
            # Separate columns by tab 
            columns = line.strip().split("\t")
            # Check existence of adequate coulumns -containing both taxa info and sample counts
            if len(columns) >= 2:
                # Iterate for each column  
                for col_index, column in enumerate(columns[1:]):
                    # Retrieve sample names corresponding to current column index
                    sample_name = sample_names[col_index]
                    # Extraxt taxa from first column, separated by ;
                    taxa = columns[0].split(";")
                    # Iterate for each taxon in line
                    for taxon in taxa:
                        if ignore_prefix and taxon in ("k__", "p__" , "c__", "o__", "f__", "g__", "s__"):
                            continue
                        # Isolate prefix
                        parts = taxon.split("__")
                        prefix = parts[0] if parts[0] else prev_prefix
                        # Update previous prefix if not null
                        if prefix:
                            prev_prefix = prefix
                        # Check if taxon is null after prefix
                        if len(parts) == 1 or (len(parts) > 1 and not parts[-1]):
                            if taxon in ("k__", "p__" , "c__", "o__", "f__", "g__", "s__"):
                                # Append previous taxon + prefix + "__unknown"
                                if prev_taxon:
                                    taxon = prev_taxon + ";" +  prefix  + "__unknown"  
                        # Update previous taxon with the current taxon for the next iteration
                        prev_taxon = taxon  
                        # Increment the count for the taxon and sample name in the dictionary
                        taxon_counts[taxon] = taxon_counts.get(taxon, {})
                        if column.strip() == "":
                            column = "0.0"
                        try:
                            taxon_counts[taxon][sample_name] = taxon_counts[taxon].get(sample_name, 0.0) + float(column.strip())
                        except ValueError as e:
                            print("Error converting to float:", e)
                            print("String causing the error:", column.strip())
            else:
                # Raise warning if not enough columns in line & ignore line
                logging.warning("Ignoring line: {} in {}. Not enough columns.".format(line_number, file_path))
            # Increase line number 
            line_number += 1
    # Return dictionary with taxon counts, header and sample names 
    return taxon_counts, header, sample_names

def write_output(output_file_path, header, taxon_counts, sample_names):
    with open(output_file_path, "w") as output_file:
        # Write header
        output_file.write(header + "\n")
        # Write each taxon and its counts per sample to the output (per row)
        for taxon, count_dict in taxon_counts.items():
            output_file.write(taxon)
            # Iterate over each sample
            for sample_name in sample_names:
                count = count_dict.get(sample_name, 0)
                output_file.write("\t{}".format(count))
            # Add new line =- end of current row
            output_file.write("\n")

def main(input_directory, output_directory, ignore_prefix):
    # Record start time
    start_time = time.time()
    # Check input directory existence
    if not os.path.isdir(input_directory):
        logging.error("Input directory not found {}.".format(input_directory))
        return
    # Iterate over each file of input directory 
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".tsv"):
            file_path = os.path.join(input_directory, file_name)
            logging.info("Processing file: {}".format(file_name))
            output_file_name = file_name[:-4] + "_output.tsv"
            output_file_path = os.path.join(output_directory, output_file_name)
            # Check if output file already exits
            if os.path.exists(output_file_path):
                logging.warning("Output file already exists for {}. - Skip processing.".format(file_name))
                continue
            taxon_counts, header, sample_names = count_taxa_occurrences(file_path, ignore_prefix)
            write_output(output_file_path, header, taxon_counts, sample_names)
            logging.info("Processed data written to: {}".format(output_file_path))
                
    # Record end time
    end_time =time.time()
    # Calculate total execution time
    execution_time = end_time - start_time
    logging.info("Total execution time is {} seconds.".format(execution_time))
    
if __name__ == "__main__":
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
    if "--ignore_prefix" in arguments_dict:
        ignore_prefix = arguments_dict.get("--ignore_prefix").lower() in ("true", "yes", "1")
    
    main(input_directory, output_directory, ignore_prefix)