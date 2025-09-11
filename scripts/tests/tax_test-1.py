import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm

# Load distances with study info
distance_study_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances/similarity_with_studies"
distance_study_file = "distances_with_studies_c_distances_batches_normalized_aggregated_lf_v4.1_with_tfidf_tfidf_pruned.tsv"
df = pd.read_csv(os.path.join(distance_study_dir, distance_study_file), sep="\t")
# Drop rows with missing study IDs or same-study comparisons
df = df.dropna(subset=["study1", "study2"])
df = df[df["study1"] != df["study2"]].copy()
# Build fast lookup: (sample1, sample2) → distance
distance_lookup = defaultdict(dict)
for _, row in df.iterrows():
    s1, s2, d = row["Run1"], row["Run2"], row["Distance"]
    distance_lookup[s1][s2] = d
    distance_lookup[s2][s1] = d
# Build mapping: study → list of samples
study_to_samples = defaultdict(set)
for _, row in df.iterrows():
    study_to_samples[row["study1"]].add(row["Run1"])
    study_to_samples[row["study2"]].add(row["Run2"])
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
output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances/similarity_with_studies/studies_only_similarity/min_avg_distance"
output_file = os.path.join(output_dir, f"study_level_similarity_min_avg_{os.path.basename(distance_study_file)}")
pd.DataFrame(results).to_csv(output_file, sep="\t", index=False)
print(f"Study-level based on min average samples' distance saved to {output_file}")