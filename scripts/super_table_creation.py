import os, sys, re
import numpy as np
from scipy.spatial import distance

def merged_data(directory):
    merged_data = {}
    sample_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                lines = file.readlines()
                header = lines[0].strip().split("\t")[1:]
                sample_names.extend(header)
                # Iterate for each line in file
                for line in lines[1:]:
                    # Separate columns by tab 
                    columns = line.strip().split("\t")
                    # Retrieve taxa from first column
                    taxa = columns[0]
                    # Retrieve values from the rest columns (except 1st)
                    values = list(columns[1:]) ### TODO check test_similarity_metrics.py for dict
                    if len(values) < len(sample_names):
                        values.extend([0] * (len(sample_names) - len(values))) ### TODO correction needed
                    merged_data.setdefault(taxa, []).extend(values)
    return sample_names, merged_data

def write_output(sample_names, merged_data, output_file):
    with open(output_file, "w") as file:
        file.write("Taxa\t" + "\t".join(sample_names) + "\n")
        for taxa, values in merged_data.items():
            file.write(taxa +"\t" + "\t".join((map(str, values))) + "\n")
               
def main():
    # Set input directory
    #input_ssu_directory = "/ccmri/similarity_metrics/data/taxa_counts_output/v5_SSU"
    #input_lsu_directory = "/ccmri/similarity_metrics/data/taxa_counts_output/v5_LSU"
    input_test_directory = "/ccmri/similarity_metrics/data/test_dataset"
    # Set output directory
    #output_ssu_file_path = "/ccmri/similarity_metrics/data/version_5.0/SSU/SSU_v5_super_table.tsv"
    #output_lsu_file_path = "/ccmri/similarity_metrics/data/version_5.0/LSU/LSU_v5_super_table.tsv"                    
    output_test_directory = "/ccmri/similarity_metrics/data/test_dataset/output.tsv"
    
    """sample_names_ssu, merged_data_ssu = merged_data(input_ssu_directory)
    write_output(sample_names_ssu, merged_data_ssu)
    
    sample_names_lsu, merged_data_lsu = merged_data(input_lsu_directory)
    write_output(sample_names_lsu, merged_data_lsu)      """
    # TODO : correct script, there are some wrong values + gaps 
    sample_names_test, merged_data_test = merged_data(input_test_directory)
    write_output(sample_names_test, merged_data_test, output_test_directory) 

if __name__ == "__main__":
    main()
    
