#!/usr/bin/python3.5

# script name: filter_biomeVSall_distances.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve distances biome VS all for samples.
# framework: CCMRI
# last update: 13/06/2025

"""
import random

# File paths
spec_biome_ids_file = "/ccmri/similarity_metrics/data/taxonomic/test_dataset/final_testing/03.07.2025_test/root:Engineered/engineered_ids.tsv"
distances_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v4.1_LSU_ge_filtered.tsv"
biome_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info/merged.tsv"
output_file = "/ccmri/similarity_metrics/data/taxonomic/test_dataset/final_testing/03.07.2025_test/root:Engineered/top_20_distances_engineeredVSall_random_5.tsv"

# Step 1: Load biome sample IDs
biome_samples = set()
with open(spec_biome_ids_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            biome_samples.add(parts[1]) 

# Step 2: Load biome file (study - sample - biome)
sample_to_study = {}
with open(biome_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            study, sample = parts[0], parts[1]
            sample_to_study[sample] = study

# Step 3: Randomly select 5 biome samples that have study info
valid_biome_samples = [s for s in biome_samples if s in sample_to_study]
random_samples = random.sample(valid_biome_samples, 5)
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
        elif s2 in random_samples and s1 in sample_to_study:
            if sample_to_study[s2] != sample_to_study[s1]:
                distances_by_sample[s2].append((s2, s1, dist))

for s, d in distances_by_sample.items():
    print(f"{s}: {len(d)} distances")

# Step 5: Sort each sample's list by distance and keep top 20
top_20_per_sample = []
for sample, dists in distances_by_sample.items():
    top_20 = sorted(dists, key=lambda x: x[2])[:20]
    top_20_per_sample.extend(top_20)

# Step 6: Write to output
with open(output_file, 'a') as out:
    for s1, s2, dist in top_20_per_sample:
        out.write(f"{s1}\t{s2}\t{dist}\n")

print(f"Done! Top-20 distances for 5 random samples written to {output_file}")
"""


#!/usr/bin/python3.5

# script name: filter_biomeVSall_distances_specific.py
# developed by: Nefeli Venetsianou
# description: Retrieve top 20 distances for specific sample(s) vs others from different studies

import sys

# ======== Input your sample IDs below ============
specific_samples = ["SRR7750095", "SRR7750096", "ERR5167522"]
# ================================================

# File paths
spec_biome_ids_file = "/ccmri/similarity_metrics/data/taxonomic/test_dataset/final_testing/03.07.2025_test/root:Environmental:Terrestrial/volcanic_ids.tsv"
distances_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v4.1_SSU_ge_filtered.tsv"
biome_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info/merged.tsv"
output_file = "/ccmri/similarity_metrics/data/taxonomic/test_dataset/final_testing/03.07.2025_test/root:Environmental:Terrestrial/top_20_distances_volcanicVSall_random_5.tsv"

# Step 1: Load sample -> study mapping
sample_to_study = {}
with open(biome_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            study, sample = parts[0], parts[1]
            sample_to_study[sample] = study

# Step 2: Validate sample IDs
valid_samples = [s for s in specific_samples if s in sample_to_study]
if not valid_samples:
    print("No valid sample IDs found with study info. Exiting.")
    sys.exit(1)

print(f"Processing sample(s): {valid_samples}")

# Step 3: Initialize distance storage
distances_by_sample = {s: [] for s in valid_samples}

# Step 4: Parse distances, filter by study
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

        for sample_id in valid_samples:
            if s1 == sample_id and s2 in sample_to_study:
                if sample_to_study[sample_id] != sample_to_study[s2] and dist < 0.8:
                    distances_by_sample[sample_id].append((sample_id, s2, dist))

            elif s2 == sample_id and s1 in sample_to_study:
                if sample_to_study[sample_id] != sample_to_study[s1] and dist < 0.8:
                    distances_by_sample[sample_id].append((sample_id, s1, dist))

# Step 5: Sort and keep top 20 distances
top_20_per_sample = []
for sample, dists in distances_by_sample.items():
    sorted_dists = sorted(dists, key=lambda x: x[2])[:20]
    top_20_per_sample.extend(sorted_dists)

# Step 6: Write output
with open(output_file, 'a') as out:
    for s1, s2, dist in top_20_per_sample:
        out.write(f"{s1}\t{s2}\t{dist}\n")

print(f"Done! Top-20 distances (excluding same-study) for {len(valid_samples)} sample(s) written to:\n{output_file}")
