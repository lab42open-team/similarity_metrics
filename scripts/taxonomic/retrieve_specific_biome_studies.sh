#!/bin/bash

# Define your inputs
BIOME_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/study_biome_info.tsv"
BASE_DIR="/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies"
OUTPUT_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/test_terrestrial_soil/soil_samples.tsv"

echo -e "study_id\tSampleID"

# Read biome file line-by-line
awk -F'\t' '$2 == "root:Environmental:Terrestrial:Soil" {print $1}' "$BIOME_FILE" | while read study_id; do
    study_dir="$BASE_DIR/$study_id"

    # Skip if study folder doesn't exist
    [ -d "$study_dir" ] || continue

    # Loop through matching files (taxonomy_abundances but NOT phylum)
    for file in "$study_dir"/*_taxonomy_abundances_*{LSU,SSU}*.tsv; do
        # Skip if file doesn't exist (avoid errors if glob fails)
        [ -e "$file" ] || continue

        # Skip phylum files
        if [[ "$file" == *phylum* ]]; then
            continue
        fi

        # Extract sample IDs from the header line (remove '#SampleID', print all remaining columns)
        header=$(head -n 1 "$file")
        sample_ids=$(echo "$header" | cut -f2-)

        # Print each sample ID with study_id
        for sample in $sample_ids; do
            echo -e "${study_id}\t${sample}" >> "$OUTPUT_FILE"
        done
    done
done
