#!/usr/bin/python3.5

import os 
import time
import logging
logging.basicConfig(level=logging.INFO)

# TODO not working properly yet - needs correction + Normalization will have to be implemented in the "raw" data and then implement aggregation. 

def normalize_counts(input_folder, output_dir):
    start_time = time.time()
    # List files in provided folder
    for filename in os.listdir(input_folder):
        # Iterate over those file with the corresponding file format
        if filename.endswith(".tsv"):
            target_file = os.path.join(input_folder, filename)
            logging.info("Processing file: {}".format(target_file))
            # Initialize dictionary to store counts per sample / taxon 
            sample_counts = {}
            with open(target_file, "r") as file:
                # Skip header   
                header = next(file).strip()
                # Extract sample names from header
                sample_names = header.split("\t")[1:]
                # Initialize sample counts for each sample 
                total_counts_per_sample = {sample: 0 for sample in sample_names}
                
                # Process each line in the file 
                for line in file:
                    columns = line.strip().split("\t")
                    # Define taxa
                    taxa = columns[0]
                    # Define counts 
                    counts = list(map(float, columns[1:]))
                    for i, sample in enumerate(sample_names):
                        total_counts_per_sample[sample] += counts[i]
                        # Initialize sample_counts for taxon if not exists
                        if taxa not in sample_counts:
                            sample_counts[taxa] = {}
                        """try:
                            relative_abundance = counts[i] / total_counts_per_sample[sample]
                        except ZeroDivisionError:
                            logging.error("Zero total count for sample {} : {}".format(sample, total_counts_per_sample[sample]))
                            raise
                        sample_counts[taxa][sample] = relative_abundance"""
                            
                        
                # Reset file pointer
                file.seek(0)
                next(file)
                for line in file: 
                    columns = line.strip().split("\t")
                    taxa = columns[0]
                    #  Iterate over each sample to calculate relative abundance
                    for i, sample in enumerate(sample_names):
                        # Initialize sample counts for taxon if not exists
                        if taxa not in sample_counts:
                            sample_counts[taxa] = {}
                        # Initialize list for the current sample if not exists
                        if sample not in sample_counts[taxa]:
                            sample_counts[taxa][sample] = []                            
                        # Calculate relative abundance per taxon & sample
                        relative_abundances = counts[i] / total_counts_per_sample[sample]
                        # Store relative abundance for the specific taxon
                        sample_counts[taxa][sample] = relative_abundances     
                        
            write_output(sample_counts, header, filename, output_dir)  
            logging.info("Output written successfully to: {}".format(output_dir))            
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info("Def execution time = {} seconds.".format(execution_time))
    return sample_counts , header                     

def write_output(sample_counts, header, filename, output_dir):
    # Construct output file name 
    output_filename = "ra" + filename
    output_file_path = os.path.join(output_dir, output_filename)
    with open(output_file_path, "w") as output_file:
        # Write header 
        output_file.write(header + "\n")
        # Write normalized counts for each taxon
        for taxa, counts in sample_counts.items():
            output_file.write("{}\t{}\n".format(taxa, "\t".join(map(str, counts.values()))))
            
def main():
    input_folder = "/ccmri/similarity_metrics/data/test_dataset/test_folder/input"
    output_dir = "/ccmri/similarity_metrics/data/test_dataset/test_folder/output/ra_test"
    # Perform normalize data
    normalize_counts(input_folder, output_dir)  
    
if __name__ == "__main__":
    main()        
