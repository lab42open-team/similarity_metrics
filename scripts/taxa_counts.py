#!/usr/bin/python3.5

# script name: taxa_counts.py
# developed by: Nefeli Venetsianou
# description: get taxa counts per sample in study (Taxonomic hierarchy: k > p > c > o > f  > g > s)
# input parameters from cmd:
    # --input_dir="your_input_directory"
    # --output_dir="your_output_directory"
    # --ignore_prefix="true" or "yes" or "1" - set this true if you want to ignore not fully specified taxa  
# framework: CCMRI

import os, re, sys, time
#import psutil # Uncomment this line if you want to retrieve total CPU usage 
import logging
logging.basicConfig(level=logging.INFO)


# Set global variables to control script execution upon development (initialize parameters to default)
input_directory = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"
#input_directory = "/ccmri/similarity_metrics/data/test_dataset/test_MGYS" #- test 
#output_directory = "/ccmri/similarity_metrics/data/test_dataset/MGYS_taxa_counts_output" #- test
output_directory = "/ccmri/similarity_metrics/data/taxa_counts_output"
ignore_prefix = False 

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

# Define function with single argument file_path
def count_taxa_occurrences(file_path):
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
                        taxon_counts[taxon][sample_name] = taxon_counts[taxon].get(sample_name, 0.0) + float(column)
            else:
                # Raise warning if not enough columns in line & ignore line
                logging.warning("Ignoring line: {} in {}. Not enough columns.".format(line_number, file_path))
            # Increase line number 
            line_number += 1
    # Return dictionary with taxon counts, header and sample names 
    return taxon_counts, header, sample_names

# Set input working directory
input_wd = input_directory 

# Control existence of input working directory
if input_wd: 
    # List directories in the parent folder 
    all_folders = [folder for folder in os.listdir(input_wd) if os.path.isdir(os.path.join(input_wd, folder))]

    # Filter folders containing "MGYS" in their names
    target_folders = [folder for folder in all_folders if "MGYS" in folder]
    
    # Counter of matching files
    num_matching_files = 0
    
    # Start time of execution
    start_time = time.time()

    # Process each target folder 
    for folder_name in target_folders:
        # Construct the full path to the current folder
        folder_path = os.path.join(input_wd, folder_name)
		# Process in read file, only if completed file exists 
        input_file_path = os.path.join(folder_path, "COMPLETED")
        # Check if the file specified exists
        if os.path.exists(input_file_path):
            # Open file
            with open(input_file_path, "r") as input_file:
                input_data = input_file.read()     
                # Find all files in the directory that match the pattern 'secondaryAccession_taxonomy_abundances(?:_SSU|LSU)?.v.[d].[d].tsv'  
                matching_files = [file_name for file_name in os.listdir(folder_path) if re.match(r'\w+_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)
                                  and not re.match(r'(.+)_phylum_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name)]
                # Count number of total matching files
                num_matching_files += len(matching_files)           
                # Check if there are matching files
                if matching_files:
                    # Process each matching file
                    for file_name in matching_files:
                        file_path = os.path.join(folder_path, file_name)
                        pattern_name = re.search(r'(.+)_taxonomy_abundances(?:_SSU|_LSU)?_v\d+\.\d+\.tsv', file_name).group()
                        
                        # Set output working dirextory
                        output_wd = output_directory
                        # Check output_wd is not empty
                        if output_wd:
                            # Construct output file name 
                            output_file_name = folder_name + "_" + file_name.split(pattern_name)[0] + pattern_name + "_output.tsv"
                            # Construct the output file dynamically based on the input file
                            output_file_path = os.path.join(output_wd, output_file_name)
                            # Check if output already exists
                            if os.path.exists(output_file_path):
                                logging.warning("Output file already exists for: {} - Skip processing.".format(file_name))
                                continue
                            # Process file since output not exist
                            taxon_counts, header, sample_names = count_taxa_occurrences(file_path)
                            # Open output file - write mode
                            with open(output_file_path, "w") as output_file:
                                # Write header
                                output_file.write(header + "\n")
                                # Write each taxon and its counts per sample to the output file (per row)
                                for taxon, count_dict in taxon_counts.items():
                                    output_file.write(taxon)
                                    # Iterate over each sample
                                    for sample_name in sample_names:
                                        count = count_dict.get(sample_name, 0)
                                        output_file.write("\t{}".format(count))
                                    # Add newline - end of current row 
                                    output_file.write("\n")
                            logging.info("Processed data written to {}".format(output_file_path))
                        else:
                            logging.warning("No output directory selected.")            
                else:
                    logging.warning("No matching files found in input directory: {}".format(input_file_path))
        else:
            logging.warning("Input file not found in: ".format(folder_path))
                                           
# End time of execution
end_time =time.time()
# Calculate exact execution time
execution_time = end_time - start_time
logging.info("Total matching files parsed: {}".format(num_matching_files))
logging.info("Execution time: {} seconds.".format(execution_time)) # or divide by 3600 to get time in hour (if needed)
# Get current CPU usage for 5 seconds - interval time can be adjusted 
#print("CPU usage = ", psutil.cpu_percent(interval = 5)) # Uncomment this line if you want to retrieve total CPU usage