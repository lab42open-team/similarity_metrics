#!/bin/bash

# Input files
SAMPLES_FILE="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/merged_sample_ids.tsv"
METADATA_FILE="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_output.tsv"
OUTPUT_FILE="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_samples_ids.tsv"

# Step 1: Extract unique MGYS IDs from metadata file (column 3)
awk '{print $3}' "$METADATA_FILE" | sort -u > mgys_ids.txt

# Step 2: Match these MGYS IDs with sample file and extract matching SampleIDs
awk 'NR==FNR {mgys[$1]; next} 
     FNR==1 {next} 
     $1 in mgys {print $1 "\t" $2}' mgys_ids.txt "$SAMPLES_FILE" > "$OUTPUT_FILE"

# Cleanup
rm mgys_ids.txt

echo "Matched samples written to: $OUTPUT_FILE"