#!/usr/bin/python3.5

# script name: study_distances.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve distances biome VS all for samples. 
    # Study level: for each sampleX in study1, retrieve all distances for sampleY of study2 and vice versa. Keep an average or the best similarity score.  
# framework: CCMRI
# last update: 10/07/2025

"""Step 1: Adjust similarity files to append the MGYS ID. 
import pandas as pd
import os 
# Load the distance data 
#distance_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics"
distance_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances"
#distance_file = "c_distances_batches_filtered_v5.0_SSU_ge_filtered.tsv"
distance_file = "c_distances_batches_normalized_aggregated_lf_v4.1_with_tfidf_tfidf_pruned.tsv"
distance_df = pd.read_csv(os.path.join(distance_dir, distance_file), sep="\t")
# Load the sample-to-study mapping
#studies_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info"
studies_dir = "/ccmri/similarity_metrics/data/functional/raw_data/biome_info"
#studies_file = "merged.tsv"
studies_file = "merged_biome_output.tsv"
mapping_df = pd.read_csv(os.path.join(studies_dir, studies_file), sep="\t")
# Create a dictionary for quick lookups
sample_to_study = dict(zip(mapping_df['run_id'], mapping_df['study_id']))
# Map study1 and study2
distance_df['study1'] = distance_df['Run1'].map(sample_to_study) # note that in taxanomic it's "Sample1/2"
distance_df['study2'] = distance_df['Run2'].map(sample_to_study)
# Save the result
#output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/similarity_with_studies"
output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances/similarity_with_studies"
output_file_name = f"distances_with_studies_{os.path.basename(distance_file)}"
output_file = os.path.join(output_dir, output_file_name)
distance_df.to_csv(output_file, sep="\t", index=False)
print(f"Merged file saved to {output_file}")
"""

"""Step 2A: Retrieve distances based on average of all similarities between samples per study. 
import pandas as pd
import os

# Load the distances with study info
distance_study_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances/similarity_with_studies"
distance_study_file = "distances_with_studies_c_distances_batches_normalized_aggregated_lf_v5.0_with_tfidf_tfidf_pruned.tsv"
df = pd.read_csv(os.path.join(distance_study_dir, distance_study_file), sep="\t")
# Remove comparisons where both samples come from the same study
df = df[df['study1'] != df['study2']].copy()
# Drop rows with missing study IDs
df = df.dropna(subset=['study1', 'study2'])
# Normalize study pair order so we don't duplicate comparisons
df['study_pair'] = df.apply(lambda row: tuple(sorted([row['study1'], row['study2']])), axis=1)
# Group by study_pair and calculate average and best (minimum) distance
aggregated = df.groupby('study_pair')['Distance'].agg(['mean', 'min']).reset_index()
aggregated[['study1', 'study2']] = pd.DataFrame(aggregated['study_pair'].tolist(), index=aggregated.index)
aggregated.drop(columns='study_pair', inplace=True)
# Rename columns for clarity
aggregated.rename(columns={
    'mean': 'avg_distance',
    'min': 'best_distance'
}, inplace=True)
# Save to file
output_dir = os.path.join(distance_study_dir, "studies_only_similarity")
output_file_name = f"study_level_similarity_{os.path.basename(distance_study_file)}"
output_file = os.path.join(output_dir, output_file_name)
aggregated.to_csv(output_file, sep="\t", index=False)
print(f"Study-level similarity matrix saved to {output_file}")
"""


"""Step 2B: 
StudyDistance(studyA, studyB):
    i. average(min distances from each sample in A to any in B)
    ii. average(min distances from each sample in B to any in A) 
    iii. keep the min distance between them (== max similarity)
"""
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm

# Load distances with study info
distance_study_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/similarity_with_studies"
distance_study_file = "distances_with_studies_c_distances_batches_filtered_v4.1_SSU_ge_filtered.tsv"
df = pd.read_csv(os.path.join(distance_study_dir, distance_study_file), sep="\t")
# Drop rows with missing study IDs or same-study comparisons
df = df.dropna(subset=["study1", "study2"])
df = df[df["study1"] != df["study2"]].copy()
# Build fast lookup: (sample1, sample2) → distance
distance_lookup = defaultdict(dict)
for _, row in df.iterrows():
    s1, s2, d = row["Sample1"], row["Sample2"], row["Distance"]
    distance_lookup[s1][s2] = d
    distance_lookup[s2][s1] = d
# Build mapping: study → list of samples
study_to_samples = defaultdict(set)
for _, row in df.iterrows():
    study_to_samples[row["study1"]].add(row["Sample1"])
    study_to_samples[row["study2"]].add(row["Sample2"])
# Get all unique study pairs
study_pairs = set()
for s1, s2 in zip(df["study1"], df["study2"]):
    if s1 != s2:
        pair = tuple(sorted([s1, s2]))
        study_pairs.add(pair)
# Calculate distances for each study pair
results = []
for study1, study2 in tqdm(study_pairs, desc="Computing study-to-study distances"):
    samples1 = study_to_samples[study1]
    samples2 = study_to_samples[study2]
    # 1. min distances from each sample in study1 to study2
    min_dists_1_to_2 = []
    for s1 in samples1:
        dists = [distance_lookup[s1][s2] for s2 in samples2 if s2 in distance_lookup[s1]]
        if dists:
            min_dists_1_to_2.append(min(dists))
    # 2. min distances from each sample in study2 to study1
    min_dists_2_to_1 = []
    for s2 in samples2:
        dists = [distance_lookup[s2][s1] for s1 in samples1 if s1 in distance_lookup[s2]]
        if dists:
            min_dists_2_to_1.append(min(dists))
    # Only calculate if we have values on both sides
    if min_dists_1_to_2 and min_dists_2_to_1:
        avg1 = sum(min_dists_1_to_2) / len(min_dists_1_to_2)
        avg2 = sum(min_dists_2_to_1) / len(min_dists_2_to_1)
        fin_min_avg_distance = min(avg1, avg2)
        results.append({
            "study1": study1,
            "study2": study2,
            "avg1to2": avg1,
            "avg2to1": avg2,
            "final_min_avg_distance": fin_min_avg_distance
        })
# Save result
output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/similarity_with_studies/study_only_similarity/min_avg_distance"
output_file = os.path.join(output_dir, f"study_level_similarity_min_avg_{os.path.basename(distance_study_file)}")
pd.DataFrame(results).to_csv(output_file, sep="\t", index=False)
print(f"Study-level based on min average samples' distance saved to {output_file}")
