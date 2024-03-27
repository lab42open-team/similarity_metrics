#!/usr/bin/python3.5
import os, sys 

def merge_data(input_dir):
    all_taxa = set()
    all_sample_names =set()
    taxa_counts = {}
    
    # Iterate through all files in parent directory
    for file in os.listdir(input_dir):
        if file.endswith(".tsv"):
            target_file = os.path.join(input_dir, file)
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
                                            
    return all_sample_names, all_taxa, taxa_counts


def write_output(all_sample_names, all_taxa, taxa_counts, output_file):
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
    input_dir = "/ccmri/similarity_metrics/data/version_5.0/v5_SSU"
    output_file = "/ccmri/similarity_metrics/data/version_5.0/v5.0_SSU_super_table.tsv"
    version_sample_names, version_taxa, version_taxa_counts = merge_data(input_dir)
    write_output(version_sample_names, version_taxa, version_taxa_counts, output_file)

if __name__ == "__main__":
    main()       
    
    
### TODO 
# 1. Add input parameters 
# 2. Add script description 
# 3. Add execution time 
# 4. Add debug parameters     