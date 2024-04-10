import os 
import pandas as pd
"""
def normalize_counts(input_folder):
    # Initialize dictionary to store counts per sample / taxon 
    sample_counts = {}
    # List files in provided folder
    for file in os.listdir(input_folder):
        # Iterate over those file with the corresponding file format
        if file.endswith(".tsv"):
            target_file = os.path.join(input_folder, file)
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
                    # Iterate over each sample to calculate total sample counts 
                    for i, sample in enumerate(sample_names):
                        total_counts_per_sample[sample] += counts[i]
                # Iterate over each sample to calculate relative abundances per sample
                for sample in sample_names:
                    sample_counts.setdefault(sample, [])
                    for i, _ in enumerate(sample_names):
                        sample_counts[sample].append(counts[i] / total_counts_per_sample[sample])
                        #print(sample_counts)
    return sample_counts

def write_output(sample_counts, target_file, output_dir):
    # Retrieve input target file name
    output_filename = os.path.splitext(os.path.basename(target_file))[0] + "relative_abundances.tsv" 
    # Construct output file path
    output_file_path = os.path.join(output_dir, output_filename) 
    with open(output_file_path, "w") as output_file:
        output_file.write("taxa\t" + "\t".join(sample_counts.keys()) + "\n")
        for taxa, counts in sample_counts.items():
            # write counts for each taxa
            output_file.write("{}\t" + "\t".join(map(str,counts)) + "\n")
          

input_folder = "/ccmri/similarity_metrics/data/taxa_counts_output/v1.0"
output_dir = "/ccmri/similarity_metrics/data/taxa_counts_output/normalized_data/ra_v1.0"
for filename in os.listdir(input_folder):
    target_file = os.path.join(input_folder, filename)
    test = normalize_counts(input_folder)
    write_output(test, target_file, output_dir) 
    
    
    # Reset file pointer to the beginning    
                file.seek(0)
                # Skip header
                next(file) 
                # Process each line in file 
                for line in file:
                    columns = line.strip().split("\t")
                    # Define taxa
                    taxa = columns[0]
                    # Define counts
                    counts = list(map(float, columns[1:]))       
"""
input_file = "/ccmri/similarity_metrics/data/test_dataset/MGYS00000462_ERP009703_taxonomy_abundances_LSU_v5.0.tsv_output.tsv"

### TODO needs correction
def normalize_counts(input_folder):
    # Initialize dictionary to store counts per sample / taxon 
    sample_counts = {}
    # List files in provided folder
    for file in os.listdir(input_folder):
        # Iterate over those file with the corresponding file format
        if file.endswith(".tsv"):
            target_file = os.path.join(input_folder, file)
            with open(input_file, "r") as file:
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
                    
                    # Calculate relative abundance per taxon & sample
                    relative_abundances = [ count / total_counts_per_sample[sample] for count, sample in zip(counts, sample_names)]     
                    # Store relative abundance for the specific taxon
                    sample_counts[taxa] = dict(zip(sample_names, relative_abundances))                 
    return sample_counts , header                     

def write_output(sample_counts, header, input_file, output_dir):
    # Extract filename from input file path 
    filename = os.path.basename(input_file)
    # Construct output file name 
    output_file_path = os.path.join(output_dir, filename)
    with open(output_file_path, "w") as output_file:
        # Write header 
        output_file.write(header + "\n")
        # Write normalized counts for each taxon
        for taxa, counts in sample_counts.items():
            output_file.write("{}\t{}\n".format(taxa, "\t".join(map(str, counts.values()))))
            
def main():
    input_folder = "/ccmri/similarity_metrics/data/test_dataset/test_folder"
    output_dir = "/ccmri/similarity_metrics/data/taxa_counts_output/normalized_data/ra_test"
    # Perform merge data
    normalize_sample_counts, header_samples = normalize_counts(input_folder)
    write_output(normalize_sample_counts, header_samples, input_folder, output_dir)
    
    print("Output written successfully to {}.".format(output_dir))   