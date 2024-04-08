import numpy as np
import pandas as pd
#from sklearn.metrics import jaccard_score 
from scipy.spatial import distance

def jaccard_score_long(input_file):
    # Define long format table and convert to DataFrame
    df = pd.read_csv(input_file, delimiter="\t")
    # Pivot DF to make Sample Ids as columns and Taxa as rows
    df_pivot = df.pivot(index="Taxa", columns="Sample", values="Count")
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
            
    return jaccard_scores

input_file = "/ccmri/similarity_metrics/data/SuperTable/long_format/lf_test_folder_super_table.tsv"
jaccard_scores = jaccard_score_long(input_file)
print("Jaccard Dissimilarity (Long Format): ")
for score in jaccard_scores:
    print(score)
    


    


print(distance.jaccard([1, 0, 0], [1, 1, 1]))