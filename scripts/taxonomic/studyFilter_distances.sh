#!/bin/bash

# Input files
SAMPLES_FILE="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_samples_ids.tsv"  # Format: MGYS_ID <tab> SampleID
DIST_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v4.1_SSU_ge_filtered.tsv"
OUTPUT_FILE="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/output_interstudy_terrestrial.tsv"

awk -v samples="$SAMPLES_FILE" -v output="$OUTPUT_FILE" '
    BEGIN {
        # Load SampleID -> StudyID
        while ((getline < samples) > 0) {
            if (NF >= 2) {
                sample_to_study[$2] = $1
                terrestrial[$2]      # mark sample as in terrestrial list
            }
        }
    }
    NR == 1 {
        print $0 > output  # write header
        next
    }
    {
        s1 = $1
        s2 = $2

        # Only keep rows where:
        # - Either s1 or s2 is from the terrestrial list
        # - Both s1 and s2 have known study IDs
        # - And their studies are different
        if ((s1 in terrestrial || s2 in terrestrial) &&
            (s1 in sample_to_study) && (s2 in sample_to_study) &&
            (sample_to_study[s1] != sample_to_study[s2])) {

            print $0 >> output
        }
    }
' "$DIST_FILE"

echo "✅ Output written to: $OUTPUT_FILE"
