#!/bin/bash

# Input file
INPUT_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/test_terrestrial_soil/soil_samples_distances_output.tsv"

# Step 1: Extract 5 random unique samples from column 1 of the input file
# Exclude the header, sort unique values, shuffle them, and select 5
selected_samples=$(awk 'NR > 1 {print $1}' "$INPUT_FILE" | sort -u | shuf -n 5)

# Step 2: For each selected sample, grep all references and save them to separate output files
for sample in $selected_samples; do
    # Step 3: Create output file with the sample name (appending distances of that sample)
    output_file="output_${sample}_distances.tsv"
    
    # Step 4: Get all lines where the sample is in the first column and append to the output file
    awk -v sample="$sample" '$1 == sample {print $0}' "$INPUT_FILE" > "$output_file"
    
    echo "Saved distances for $sample in $output_file"
done
