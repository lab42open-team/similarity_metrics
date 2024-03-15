import os, sys 

def merge_data(input_dir):
    sample_names = []
    taxa_counts = {}
    all_taxa = set()
    
    # Iterate through all files in parent directory
    for file in os.listdir(input_dir):
        if file.endswith(".tsv"):
            target_file = os.path.join(input_dir, file)
            with open(target_file, "r") as file:
                print("processing file: ", target_file)
                # Extract sample named from header
                header = next(file).strip().split("\t")[1:]
                #print("Header", header)
                sample_names.extend(header)
                #print("Sample names", sample_names)
                
                # Initialize taxa counts dictionary 
                for sample_name in sample_names:
                    taxa_counts[sample_name] = {}
                    
                # Process each line in the file 
                for line in file:
                    columns = line.strip().split("\t")
                    #print(columns)
                    taxa = columns[0]
                    #print(taxa)
                    counts = columns[1:]
                    # Update set of all taxa
                    all_taxa.add(taxa)
                    
                    # Update taxa counts dictionary 
                    for i, sample_name in enumerate(sample_names):
                        count = counts[i] if i < len(counts) else "0"
                        try:
                            count = float(count)
                        except ValueError:
                            count = 0
                        taxa_counts[sample_name][taxa] = count
                print(sample_name, taxa, columns)        
    # Fill missing taxa counts with zeros
    for sample_name in sample_names: 
        for taxa in all_taxa:
            if taxa not in taxa_counts[sample_name]:
                taxa_counts[sample_name][taxa] = 0
                    
    return sample_names, taxa_counts

# TODO: print is correct, but output not. Needs correction 
def write_output(sample_names, taxa_counts, output_file):
    sample_names, taxa_counts = merge_data
    with open(output_file, "w") as file:
        # Write header
        file.write("Taxa\t" + "\t".join(sample_names) + "\n")
        # Write taxa counts for each sample
        for taxa, count_dict in taxa_counts.items():
            file.write(taxa + "\t")
            for sample_name in sample_names: 
                count = count_dict.get(sample_name, 0)
                file.write("\t{}".format(count))
            file.write("\n")
        
            
def main():
    input_dir = "/ccmri/similarity_metrics/data/test_dataset"
    output_file = "/ccmri/similarity_metrics/data/test_dataset/output.tsv"
    sample_names_test, taxa_counts_test = merge_data(input_dir)
    write_output(sample_names_test, taxa_counts_test, output_file)

if __name__ == "__main__":
    main()       