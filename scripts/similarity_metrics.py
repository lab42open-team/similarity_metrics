#!/usr/bin/python3.5

# script name: lf_jaccard.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate similarity metrcis by using long format super table (lf - sp) (lf -sp produced by long_format_sp.py)
    # Used in noisy injection to get similarity metrics performance curve.
    # For now: euklidean and cosine are calculated. 
# framework: CCMRI
# last update: 18/06/2024

import os
import pandas as pd
import numpy as np
#from sklearn.metrics import jaccard_score 
from scipy.spatial import distance
import time
import logging
logging.basicConfig(level=logging.INFO)

def preprocess_data(input_file):
    # Read input file 
    #df = pd.read_parquet(input_file) # please uncomment if you want to import parquet
    df = pd.read_csv(input_file, delimiter="\t")
    # Pivot table to get samples as columns and taxa as rows
    df_pivot = df.pivot(index="Taxa", columns="Sample", values="Count")
    # Fill NaN calues with 0
    #df_pivot = df_pivot.fillna(0)
    # Convert count to binary (0 - do not exist or 1 - exist)
    df_pivot = (df_pivot > 0).astype(int)
    return(df_pivot)


def euklidean_distance():
    #TODO
             
def write_euklidean_output():
    #TODO
    
def cosine_similarity():
    #TODO

def write_cosine_output():
    #TODO

def main():
    # Set input file (either .tsv or .parquet)
    input_file = "" 
    # Set output directory 
    output_dir = ""
    # Record total start time
    tot_start_time = time.time()
    # Perform jaccard dissimilarity calculation
    jaccard_dissimality = jaccard_score_long(input_file)
    write_output(jaccard_dissimality, input_file, output_dir)
    # Record total end time 
    tot_end_time = time.time()
    # Calculate total execution time 
    tot_execution_time = tot_end_time- tot_start_time
    logging.info("Total execution time = {} seconds.".format(tot_execution_time))

if __name__ == "__main__":
    main()

"""
# JACCARD 
def jaccard_score(input_file):
    # Record start time
    start_time = time.time()
    # Preprocess data
    df_pivot = preprocess_data(input_file)
    # Extract sample IDs from columns
    sample_ids = df_pivot.columns.tolist()
    # Initialize jaccard scores list to store results
    jaccard_scores = []
    for i in range(len(sample_ids)):
        for j in range(i+1, len(sample_ids)):
            # Define sets and get counts
            set1 = df_pivot[sample_ids[i]].values
            set2 = df_pivot[sample_ids[j]].values
            jaccard_distance = distance.jaccard(set1,set2)
            # Append results to list
            jaccard_scores.append((sample_ids[i], sample_ids[j], jaccard_distance))
    # Record end time
    end_time = time.time()
    # Calculate execution time 
    execution_def_time = end_time - start_time
    logging.info("Execution def time = {} seconds.".format(execution_def_time))        
    return jaccard_scores
    
def write_jaccard_output(jaccard_scores, input_file, output_dir):
     # Extract version pipeline from input file name 
    input_file_name = os.path.basename(input_file)
    version = input_file_name.split("_super_table.tsv")[0]
    # Construct output file name and path
    output_file_name = "jaccard_" + version + "_output.tsv"
    output_file = os.path.join(output_dir, output_file_name)        
    with open(output_file, "w") as file:
        file.write("Sample i\tSample j\tJaccard Dissimilarity Score\n")
        for item in jaccard_scores:
            file.write("{}\t{}\t{}\n".format(item[0], item[1], item[2]))

print("Jaccard Dissimilarity (Long Format): ")
for score in jaccard_scores:
    print(score)
"""