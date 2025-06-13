#!/bin/bash

# Inputs
DATA_FILE="/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/merged_output.tsv"
BASE_DIR="/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"
OUTPUT_FILE="/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/merged_sample_ids.tsv"

# Output header
echo -e "study_id\tSampleID" > "$OUTPUT_FILE"

# Extract unique MGYS study IDs from aquatic_only.tsv
cut -f3 "$DATA_FILE" | sort -u | while read study_id; do
    study_dir="$BASE_DIR/$study_id"

    # Skip if the directory doesn't exist
    [ -d "$study_dir" ] || continue

    # Loop through the taxonomy_abundances files, excluding phylum
    for file in "$study_dir"/*_taxonomy_abundances_*{LSU,SSU}*.tsv; do
        [ -e "$file" ] || continue

        [[ "$file" == *phylum* ]] && continue

        # Extract sample IDs from the header line
        header=$(head -n 1 "$file")
        sample_ids=$(echo "$header" | cut -f2-)

        # Print study ID and each sample ID
        for sample in $sample_ids; do
            echo -e "${study_id}\t${sample}" >> "$OUTPUT_FILE"
        done
    done
done
