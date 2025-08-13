### RETRIEVE ALL DISTANCES FOR SPECIFIC SAMPLES ###
#!/bin/bash

# Inputs
#SAMPLE_IDS_FILE="/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/merged_sample_ids.tsv" 
SAMPLE_IDS_FILE="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_samples_ids.tsv" #terrestrial only
#DIST_FILE="/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/sim_metrics/c_distances_batches_normalized_f_aggregated_0.75_0.01_lf_v4.1_super_table.tsv" #functional
DIST_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v4.1_SSU_ge_filtered.tsv"
#OUTPUT="/ccmri/similarity_metrics/data/other/matched_mgnify_spire_data/merged_dataset_distances.tsv"
OUTPUT="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_taxonomic_distances.tsv" #terrestrial only

# Step 1: Extract unique SampleIDs (skip header)
awk 'NR > 1 {print $2}' "$SAMPLE_IDS_FILE" | sort -u > tmp_all_samples.txt

# Step 2: Write header to output
head -n 1 "$DIST_FILE" > "$OUTPUT"

# Step 3: Filter DIST_FILE to include only rows where column 1 is in the list
awk 'NR==FNR {samples[$1]; next} 
     FNR==1 || ($1 in samples)' tmp_all_samples.txt "$DIST_FILE" >> "$OUTPUT"

# Step 4: Cleanup
rm tmp_all_samples.txt

echo "Filtered distances written to: $OUTPUT"

