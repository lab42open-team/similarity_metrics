### In this script, a user can set as input parameters, the working directory of choice, the MGYS and the taxon of interest 
### the number of counts of the specific taxon per sample will be returned 
# input_dir="your_input__direcroty"
# study_name="MGYS00000xxxx"
# taxon_of_interest="taxon"

import os, sys, time

def search_taxon(input_dir, study_name, taxon_of_interest):
    # Search for files in parent directory containing the study name provided in the file name
    target_file = [file for file in os.listdir(input_dir) if study_name in file]
    if not target_file:
        print("No files found for the study name provided.")
    else:
        # Iterate for file containing study name
        with open(os.path.join(input_dir, target_file[0]), "r") as file:
            print("Counts of {}".format(taxon_of_interest), "in {}".format(target_file[0]), ":")
            header_printed = False
            for line in file:
                columns = line.strip().split("\t")
                if not header_printed:
                    print("\t".join(columns))
                    header_printed = True
                if taxon_of_interest in columns[0]:
                    print("\t".join(columns[1:]))  

if __name__ == "__main__":
    # Set default parameters
    # input_dir = "/ccmri/profile_matching/test_dataset/MGYS_taxa_counts_output" - test 
    input_dir = "/ccmri/profile_matching/taxa_counts_output"
    study_name = ""
    taxon_of_interest = ""
    
    # Extract arguments imported from sy.argv via command line
    arguments_dict = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            sep = arg.find('=')
            key, value = arg[:sep], arg[sep + 1:]
            arguments_dict[key] = value
        
    # Update parameters based on the values passed by the command line (if any)
    if "--input_dir" in arguments_dict:
        input_dir = arguments_dict["--input_dir"]
    if "--study_name" in arguments_dict:
        study_name = arguments_dict["--study_name"]
    if "--taxon_of_interest" in arguments_dict:
        taxon_of_interest = arguments_dict["--taxon_of_interest"]    
    
    # Call function
    if study_name and taxon_of_interest:
        search_taxon(input_dir, study_name, taxon_of_interest)
    else:
        print("Please provide both study name and taxon of interest")