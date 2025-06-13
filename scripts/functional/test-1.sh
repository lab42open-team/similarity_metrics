



### SHUFFLE PICK 5 SAMPLES ###
#!/bin/bash

# Inputs
SAMPLE_IDS_FILE="/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/aquatic_test/aquatic_sample_ids.tsv"
DIST_FILE="/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/c_distances_batches_normalized_f_aggregated_0.75_0.01_lf_v4.1_super_table.tsv"
OUTPUT="/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/aquatic_test/aquatic_test_distances.tsv"

# Temp file to store 5 selected samples
TMP_SAMPLES="tmp_selected_samples.txt"

# Step 1: Extract 5 unique SampleIDs randomly from biome file (column 2)
awk 'NR > 1 {print $2}' "$SAMPLE_IDS_FILE" | sort -u | shuf -n 5 > "$TMP_SAMPLES"

# Step 2: Write header to output file
head -n 1 "$DIST_FILE" > "$OUTPUT"

# Step 3: For each selected sample, get top 10 closest distances
while read sample; do
    awk -v s="$sample" '$1 == s' "$DIST_FILE" | sort -k3,3g | head -n 10 >> "$OUTPUT"
done < "$TMP_SAMPLES"

# Optional: clean up
rm "$TMP_SAMPLES"
