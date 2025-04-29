#!/bin/bash

# Inputs
BIOME_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/test_terrestrial_soil/soil_samples.tsv"
DIST_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v4.1_LSU_ge_filtered.tsv"
OUTPUT="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/test_terrestrial_soil/soil_samples_distances.tsv"

# Temp file to store 5 selected samples
TMP_SAMPLES="tmp_selected_samples.txt"

# Step 1: Extract 5 unique SampleIDs randomly from biome file (column 2)
awk 'NR > 1 {print $2}' "$BIOME_FILE" | sort -u | shuf -n 5 > "$TMP_SAMPLES"

# Step 2: Write header to output file
head -n 1 "$DIST_FILE" > "$OUTPUT"

# Step 3: For each selected sample, get top 10 closest distances
while read sample; do
    awk -v s="$sample" '$1 == s' "$DIST_FILE" | sort -k3,3g | head -n 10 >> "$OUTPUT"
done < "$TMP_SAMPLES"

# Optional: clean up
rm "$TMP_SAMPLES"
