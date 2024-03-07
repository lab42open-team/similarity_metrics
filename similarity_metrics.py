### This is a script to calculate similarity metrics between studies and samples in a study. 
# All vs All:
# 1. Sample vs Sample within a study 
# 2. Samples vs Samples in different studies of the same version (eg. v1.0, v2.0, etc)
# 3. For v4.0 and v5.0: Samples vs Samples in different studies of the same LSU or SSU type
# 4. Studies vs Studies independently from version or type 

import os, sys
import numpy as np
# from sklearn.metrics import jaccard_score - works in python > 3


def jaccard_similarity(set1, set2):
    # Define intersection of two sets
    intersection = len(set1.intersection(set2))
    # Define union of two sets
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection/union


# Set input directory 
#input_dir = "/ccmri/profile_matching/test_dataset/MGYS_taxa_counts_output" # test
# Read target file 
input_file = "/ccmri/profile_matching/test_dataset/MGYS_taxa_counts_output/MGYS00000308_ERP001739_taxonomy_abundances_v1.0.tsv_output.tsv" #[file for file in os.listdir(input_dir)]
# Dictionary to store counts for each sample
sample_counts = {}
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
            
# Iterate over indices of Sample IDs
for i in range(len(sample_ids)):
    # Iterate over indices of sample IDs starting from i+1 to avoid comparing a sample to itself (duplicates)
    for j in range(i + 1, len(sample_ids)):
        # Define sets of ith and jth sample
        set1 = set(sample_counts[sample_ids[i]].keys())
        set2 = set(sample_counts[sample_ids[j]].keys())
        # Calculate jaccard similarity 
        similarity = jaccard_similarity(set1, set2)
        print("Jaccard Similarity for {} between samples {}, {} is {}".format(taxon, sample_ids[i], sample_ids[j], similarity))
            