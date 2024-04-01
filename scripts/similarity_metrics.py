### This is a script to calculate similarity metrics between studies and samples in a study. 
# All vs All:
# 1. Sample vs Sample within a study 
# 2. Samples vs Samples in different studies of the same version (eg. v1.0, v2.0, etc)
# 3. For v4.0 and v5.0: Samples vs Samples in different studies of the same LSU or SSU type
# 4. Studies vs Studies independently from version or type 

import os, sys
import numpy as np
# from sklearn.metrics import jaccard_score - works in python > 3
from scipy.spatial import distance
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# Set input directory 
#input_dir = "/ccmri/profile_matching/test_dataset/MGYS_taxa_counts_output" # test
# Read target file 
#input_file = "/ccmri/data_similarity_metrics/test_dataset/MGYS_taxa_counts_output/MGYS00000305_ERP001568_taxonomy_abundances_v1.0.tsv_output.tsv" #[file for file in os.listdir(input_dir)]
#input_file = "/ccmri/similarity_metrics/data/test_dataset/output.tsv" #[file for file in os.listdir(input_dir)]

def jaccard_score(input_file):
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

    return jaccard_scores
                
#jaccard = jaccard_score(input_file)                

def write_output(jaccard_scores, output_file):        
    with open(output_file, "w") as file:
        file.write("Sample i\tSample j\tJaccard Score\n")
        for item in jaccard_scores:
            file.write("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
            
def main():
    input_file = "/ccmri/similarity_metrics/data/SuperTable/v5.0_LSU_super_table.tsv"
    output_file = "/ccmri/similarity_metrics/data/Metrics/v5.0_LSU_jaccard_output.tsv"
    test_jaccard_smlrt = jaccard_score(input_file)
    write_output(test_jaccard_smlrt, output_file)
    
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