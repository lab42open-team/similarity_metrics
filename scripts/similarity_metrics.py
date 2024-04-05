#!/usr/bin/python3.5

### This is a script to calculate similarity metrics between studies and samples in a study. 
# All vs All:
# 1. Sample vs Sample within a study 
# 2. Samples vs Samples in different studies of the same version (eg. v1.0, v2.0, etc)
# 3. For v4.0 and v5.0: Samples vs Samples in different studies of the same LSU or SSU type
# 4. Studies vs Studies independently from version or type 

# script name: similarity_metrics.py
# developed by: Nefeli Venetsianou
# description: calculate similarity metrics between studies and samples in a study
# input parameters from cmd:
    # --input_file="input_file_name"
# framework: CCMRI

import os, sys
import numpy as np
from scipy.spatial import distance
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import time
import logging
logging.basicConfig(level=logging.INFO)
#import psutil # Uncomment this line if you want to retrieve total CPU usage 

def jaccard_score(input_file):
    # Dictionary to store counts for each sample
    sample_counts = {}
    # Record start time 
    start_time = time.time()
    
    with open(input_file, "r") as file:
        # Read header & sample IDs
        header = file.readline().strip().split("\t")
        sample_ids = header[1:]
        # Initialize sample counts dictionary
        for sample_id in sample_ids:
            sample_counts[sample_id] = {}
        # Read counts for each taxon 
        for line in file:
            columns = line.strip().split("\t")
            taxon = columns[0]
            counts = list(map(float, columns[1:]))
            for i, sample_id in enumerate(sample_ids):
                sample_counts[sample_id][taxon] = counts[i]
            #print(header, taxon, counts)
        # Initialize jaccard_score 
        jaccard_scores = []    
    # Iterate over indices of Sample IDs
    for i in range(len(sample_ids)):
    # Iterate over indices of sample IDs starting from i+1 to avoid comparing a sample to itself (duplicates)
        for j in range(i + 1, len(sample_ids)):
            # Define sets of ith and jth sample
            set1 = np.array(list(sample_counts[sample_ids[i]].values()))
            set2 = np.array(list(sample_counts[sample_ids[j]].values()))
            jaccard_distance = distance.jaccard(set1, set2) 
            jaccard_scores.append((sample_ids[i], sample_ids[j], jaccard_distance))
            #print("Jaccard Similarity Score between {}, {} = {}.".format(sample_ids[i], sample_ids[j], jaccard_distance))
    # Record end time 
    end_time = time.time()
    # Calculate execution time 
    execution_def_time = end_time - start_time
    logging.info("Execution def time = {} seconds".format(execution_def_time))
    return jaccard_scores
                
def write_output(jaccard_scores, input_file, output_dir):
    # Extract version pipeline from input file name 
    input_file_name = os.path.basename(input_file)
    version = input_file_name.split("_super_table.tsv")[0]
    # Construct output file name and path
    output_file_name = version + "_jaccard_output.tsv"
    output_file = os.path.join(output_dir, output_file_name)        
    with open(output_file, "w") as file:
        file.write("Sample i\tSample j\tJaccard Score\n")
        for item in jaccard_scores:
            file.write("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
            
def main():   
    # Extract argument values imported from sys.argv
    arguments_dict = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            sep = arg.find('=')
            key, value = arg[:sep], arg[sep + 1:]
            arguments_dict[key] = value
    # Set default input - output directories
    input_file = "/ccmri/similarity_metrics/data/SuperTable/wide_format/wf_test_folder_super_table.tsv" 
    #input_dir = "/ccmri/similarity_metrics/data/SuperTable/"
    output_dir = "/ccmri/similarity_metrics/data/Metrics/"
    """       
    # Update parameters based on the values passed by the command line 
    input_file = arguments_dict.get("--input_file")
    if not input_file:
        input_file = input("Please provide input file name, eg. v5.0_LSU_super_table.tsv: ")
    if input_file:
        input_file = os.path.join(input_dir, input_file)
    else:
        logging.warning("No input file provided.")
        sys.exit(1)
    """    
    # Record total start time 
    tot_start_time = time.time()
    
    # Perform jaccard similarity calculation
    jaccard_smlrt = jaccard_score(input_file)
    
    # Write output file
    write_output(jaccard_smlrt, input_file, output_dir)
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
   
        
        
        
"""
        # Try different similarity metrics
        bray_curtis_dissimilarity = distance.braycurtis(set1, set2)
        #jaccard_distance = distance.jaccard(set1, set2) # no sense to calculate this one, doesn't take into account the frequence [counts]
        cosine_similarity = distance.cosine(set1, set2)
        #canbera_distance = distance.canberra(set1, set2) # no sense to calculate this one, doesn't take into account the frequence [counts]
        jehsen_shannon_divergence = distance.jensenshannon(set1, set2)        
        dice_similarity = distance.dice(set1, set2)
        tb_coefficient, p_value_tb = kendalltau(set1, set2)
        pearson_coefficient, p_value_pearson = pearsonr(set1,set2)
        rho_coefficient, p_value_spearman = spearmanr(set1, set2)
"""

"""
        # Print statements
        print("Bray-Curtis Dissimilarity between {}, {} = {}".format(sample_ids[i], sample_ids[j], bray_curtis_dissimilarity))
        #print("Jaccard Similarity Score between {}, {} = {}.".format(sample_ids[i], sample_ids[j], jaccard_distance))
        print("Cosine Similarity Score between {}, {} = {}".format(sample_ids[i], sample_ids[j], cosine_similarity))
        #print("Canberra Distance between {}, {} = {}".format(sample_ids[i], sample_ids[j], canbera_distance))
        print("Jensen-Shannon Distance between {}, {} = {}".format(sample_ids[i], sample_ids[j], jehsen_shannon_divergence))
        print("Dice Similarity between {}, {} = {}".format(sample_ids[i], sample_ids[j], dice_similarity))
        print("Kendall's tau-b correlation coefficient between {}, {} = {} and p-value = {}".format(sample_ids[i], sample_ids[j], tb_coefficient, p_value_tb))
        print("Pearson correlation coefficient between {}, {} = {} and p-value = {}".format(sample_ids[i], sample_ids[j], pearson_coefficient, p_value_pearson))
        print("Spearman correlation coefficient between {}, {} = {} and p-value = {}".format(sample_ids[i], sample_ids[j], rho_coefficient, p_value_spearman))
"""