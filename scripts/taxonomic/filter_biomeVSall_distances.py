#!/usr/bin/python3.5

# script name: filter_biomeVSall_distances.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve distances biome VS all for samples.
# framework: CCMRI
# last update: 13/06/2025


import random

# File paths
aquatic_ids_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/test_sample_matcher/aquatic/aquatic_ids_only.tsv"
distances_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances/c_distances_batches_normalized_aggregated_lf_v4.1_with_tfidf_tfidf_pruned.tsv"
metadata_file = "/ccmri/similarity_metrics/data/functional/raw_data/biome_info"
output_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/test_sample_matcher/aquatic/top_30_distances_aquaticVSall_random_5.tsv"

# Step 1: Load aquatic sample IDs
aquatic_samples = set()
with open(aquatic_ids_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            aquatic_samples.add(parts[1])  # assuming sample is in 2nd column

# Step 2: Load metadata (sample → study + biome)
sample_to_study = {}
with open(metadata_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            sample, study = parts[0], parts[1]
            sample_to_study[sample] = study

# Step 3: Randomly select 5 aquatic samples that have study info
valid_aquatic_samples = [s for s in aquatic_samples if s in sample_to_study]
random_samples = random.sample(valid_aquatic_samples, 5)
print(f"Selected samples: {random_samples}")

# Step 4: Gather distances for the selected samples
distances_by_sample = {s: [] for s in random_samples}

with open(distances_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        s1, s2, dist = parts
        try:
            dist = float(dist)
        except ValueError:
            continue

        # If s1 is in our selected samples, and s2 is from a different study
        if s1 in random_samples and s2 in sample_to_study:
            if sample_to_study[s1] != sample_to_study[s2]:
                distances_by_sample[s1].append((s1, s2, dist))

        # Also check reverse direction: s2 is selected, s1 is different
        elif s2 in random_samples and s1 in sample_to_study:
            if sample_to_study[s2] != sample_to_study[s1]:
                distances_by_sample[s2].append((s2, s1, dist))

# Step 5: Sort each sample's list by distance and keep top 30
top_30_per_sample = []
for sample, dists in distances_by_sample.items():
    top_30 = sorted(dists, key=lambda x: x[2])[:30]
    top_30_per_sample.extend(top_30)

# Step 6: Write to output
with open(output_file, 'w') as out:
    for s1, s2, dist in top_30_per_sample:
        out.write(f"{s1}\t{s2}\t{dist}\n")

print(f"Done! Top-30 distances for 5 random samples written to {output_file}")
