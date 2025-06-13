#!/bin/bash

# Inputs
BIOME_FILE="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_samples_ids.tsv"
DIST_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v5.0_SSU_ge_filtered.tsv"
OUTPUT="/ccmri/similarity_metrics/data/other/mathed_mgnify_spire_functional_data/terrestrial_test/terrestrial_taxonomic_distances_studyFiltered_top30.tsv"

# Temp file to store study information: format -> SampleID StudyID
TMP_STUDY_INFO="tmp_study_info.txt"
awk 'NR > 1 {print $2, $3}' "$BIOME_FILE" > "$TMP_STUDY_INFO"

# Temp file to store all unique SampleIDs from DIST_FILE
TMP_SAMPLES="tmp_all_samples.txt"
awk 'NR > 1 {print $1; print $2}' "$DIST_FILE" | sort -u > "$TMP_SAMPLES"

# Create an associative array of SampleID -> StudyID in awk
awk -v dist_file="$DIST_FILE" -v output="$OUTPUT" '
    BEGIN {
        # Read study info into an associative array
        while ((getline < "'$TMP_STUDY_INFO'") > 0) {
            study[$1] = $2
        }

        # Open DIST_FILE and write the header
        getline header < dist_file
        print header > output

        # Process the rest of DIST_FILE
        while ((getline line < dist_file) > 0) {
            split(line, fields, "\t")
            s1 = fields[1]
            s2 = fields[2]
            dist = fields[3] + 0  # numeric distance

            # Only consider pairs from different studies
            if ((s1 in study) && (s2 in study) && (study[s1] != study[s2])) {
                key = s1 "_" s2
                lines[key] = line
                values[key] = dist
            }
        }

        # Sort keys by distance
        n = asorti(values, sorted_keys, "@val_num_asc")

        # Output top 30
        for (i = 1; i <= 30 && i <= n; i++) {
            print lines[sorted_keys[i]] >> output
        }
    }
'

# Cleanup
rm "$TMP_SAMPLES" "$TMP_STUDY_INFO"

echo "Filtered top 30 inter-study distances written to: $OUTPUT"
