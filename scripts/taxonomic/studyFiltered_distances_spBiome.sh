#!/bin/bash

# Inputs
BIOME_FILE="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_samples_ids.tsv"
DIST_FILE="/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/c_distances_batches_normalized_f_aggregated_0.75_0.01_lf_v4.1_super_table.tsv"
OUTPUT="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_samples_distances_studyFiltered_top30_v4.1.tsv"

# Temp file to store 5 selected samples
TMP_SAMPLES="tmp_selected_samples.txt"

# Temp file to store study information
TMP_STUDY_INFO="tmp_study_info.txt"

# Step 1: Extract unique marine sample IDs randomly from DIST_FILE (assuming first column contains SampleID)
awk 'NR > 1 {print $1}' "$DIST_FILE" | sort -u | shuf -n 5 > "$TMP_SAMPLES"

# Step 2: Extract sample and study info from BIOME_FILE (SampleID and Study) and store in a temp file
awk 'NR > 1 {print $2, $3}' "$BIOME_FILE" > "$TMP_STUDY_INFO"

# Step 3: Write header to output file
head -n 1 "$DIST_FILE" > "$OUTPUT"

# Step 4: Compare Sample1 and Sample2 (to ensure they are from different studies)
while read sample; do
    # Extract the study for the current sample from the BIOME_FILE
    sample_study=$(grep -w "$sample" "$TMP_STUDY_INFO" | awk '{print $2}')

    # Now, pick the second sample from TMP_SAMPLES and check if they are from different studies
    while read sample2; do
        sample2_study=$(grep -w "$sample2" "$TMP_STUDY_INFO" | awk '{print $2}')

        # Check if the samples are from different studies
        if [[ "$sample_study" != "$sample2_study" ]]; then
            # Get the top 10 distances between sample and sample2 from DIST_FILE
            awk -v s1="$sample" -v s2="$sample2" '$1 == s1 && $2 == s2 || $1 == s2 && $2 == s1' "$DIST_FILE" | sort -k3,3g | head -n 30 >> "$OUTPUT"
        fi
    done < "$TMP_SAMPLES"
done < "$TMP_SAMPLES"

# Optional: clean up
rm "$TMP_SAMPLES"
rm "$TMP_STUDY_INFO"
