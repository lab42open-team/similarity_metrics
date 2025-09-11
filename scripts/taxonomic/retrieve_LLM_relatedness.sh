#!/bin/bash

# Usage:
# ./append_distance.sh final_merged.tsv merged_similarity_with_relatedness.tsv output.tsv

merged_file="$1"     # e.g., final_merged.tsv
distance_file="$2"   # e.g., merged_similarity_with_relatedness.tsv
output_file="$3"     # e.g., final_with_distance.tsv

if [[ ! -f "$merged_file" || ! -f "$distance_file" ]]; then
  echo "❌ Error: One or both input files do not exist."
  exit 1
fi

# Build the lookup table from the distance file
awk -F'\t' 'NR > 1 { dist[$1"\t"$2] = $3 } 
  END { for (k in dist) print k"\t"dist[k] }' "$distance_file" > /tmp/.dist_lookup.tsv

# Join with merged file
awk -F'\t' '
  BEGIN { OFS="\t" }
  FNR==NR { d[$1"\t"$2]=$3; next }
  NR==1 { print $0, "distance"; next }
  {
    key = $1"\t"$2;
    print $0, (key in d ? d[key] : "NA")
  }
' /tmp/.dist_lookup.tsv "$merged_file" > "$output_file"

rm /tmp/.dist_lookup.tsv

echo "✅ Appended distance column to: $output_file"
