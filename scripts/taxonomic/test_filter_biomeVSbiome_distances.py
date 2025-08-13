#!/usr/bin/python3.5

# script name: filter_biomeVSbiome_distances.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve distances biome VS biome for samples.
# framework: CCMRI
# last update: 13/06/2025


"""
# Taxonomic data #
#biome_ids_file = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/terrestrial_test/terrestrial_samples_ids.tsv"
biome_ids_file = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/aquatic_test/aquatic_sample_ids.tsv"
distances_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v4.1_LSU_ge_filtered.tsv"
#output_file = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/terrestrial_test/terrestrial_top30_distances.tsv"
output_file = "/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/aquatic_test/aquatic_top30_distances.tsv"
"""
# Functional data #
biome_ids_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/test_sample_matcher/aquatic/aquatic_ids_only.tsv"
distances_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/post_pruned_distances/c_distances_batches_normalized_aggregated_lf_v4.1_with_tfidf_tfidf_pruned.tsv"
output_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/test_sample_matcher/aquatic/top_30_distances_aquatic.tsv"

# Step 1: Map sample IDs to study IDs
sample_to_study = {}
with open(biome_ids_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            study, sample = parts[0], parts[1]
            sample_to_study[sample] = study

# Step 2: Read and filter distances
filtered_distances = []
with open(distances_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        sample1, sample2, distance = parts
        try:
            distance = float(distance)
        except ValueError:
            continue  # Skip header or malformed lines

        # Keep only if Sample1 is terrestrial and study1 != study2
        if sample1 in sample_to_study:
            study1 = sample_to_study[sample1]
            study2 = sample_to_study.get(sample2)  # might be None
            if study2 is None or study1 != study2:
                filtered_distances.append((sample1, sample2, distance))

# Step 3: Sort by distance and keep top 30
top_30 = sorted(filtered_distances, key=lambda x: x[2])[:30]

# Step 4: Write output
with open(output_file, 'a') as out:
    for sample1, sample2, distance in top_30:
        out.write(f"{sample1}\t{sample2}\t{distance}\n")

print(f"Done! Top 30 inter-study distances appended to {output_file}")


### PART B ###
# biome VS all distances #

