#!/bin/bash

# Usage:
# ./append_distance.sh final_merged.tsv merged_similarity_with_relatedness.tsv output.tsv

merged_file="/ccmri/merged_total_studies_similarity_files/final_merged.tsv"
distance_file="/ccmri/merged_total_studies_similarity_files/input_files/merged_taxonomic_studies_similarity.tsv"
output_file="/ccmri/merged_total_studies_similarity_files/relatedness_with_distance_final.tsv"

if [[ ! -f "$merged_file" || ! -f "$distance_file" ]]; then
  echo "Error: One or both input files do not exist."
  exit 1
fi

# Detect if the distance_file path contains "taxonomic" or "functional" and assign label accordingly
if [[ "$distance_file" == *taxonomic* ]]; then
  label="Taxonomic"
elif [[ "$distance_file" == *functional* ]]; then
  label="Functional"
else
  label="Unknown"
fi

# Build the lookup table from the distance file
awk -F'\t' 'NR > 1 { dist[$1"\t"$2] = $3 } 
  END { for (k in dist) print k"\t"dist[k] }' "$distance_file" > /tmp/.dist_lookup.tsv

# Join with merged file and append distance and label columns (6th and 7th)
awk -F'\t' -v label="$label" '
  BEGIN { OFS="\t" }
  FNR==NR { d[$1"\t"$2]=$3; next }
  NR==1 { print $0, "distance", "label"; next }
  {
    key = $1"\t"$2;
    dist_val = (key in d ? d[key] : "NA")
    print $0, dist_val, (dist_val=="NA" ? "NA" : label)
  }
' /tmp/.dist_lookup.tsv "$merged_file" >> "$output_file"

rm /tmp/.dist_lookup.tsv

echo "Appended distance and label columns to: $output_file"
