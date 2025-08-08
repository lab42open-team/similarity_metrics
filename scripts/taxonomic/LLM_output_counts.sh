#!/bin/bash

# Input files
FILES=(
  "/ccmri/merged_total_studies_similarity_files/LLM_pairwise_output_phi4_14b-q8_0_run1.json"
  "/ccmri/merged_total_studies_similarity_files/LLM_pairwise_output_phi4_14b-q8_0_run2.json"
  "/ccmri/merged_total_studies_similarity_files/LLM_pairwise_output_phi4_14b-q8_0_run3.json"
)

# Output file
OUTFILE="/ccmri/merged_total_studies_similarity_files/combined_counts.tsv"

# Categories of interest
CATEGORIES=("high" "medium" "low" "none")

# Declare associative array
declare -A COUNTS

# Extract and store counts
for i in "${!FILES[@]}"; do
    run=$((i+1))
    tmpfile="tmp_run_${run}.txt"

    grep '"answer"' "${FILES[$i]}" \
    | awk -F'"' '{gsub(/\*/, "", $4); gsub(/[<>]/, "", $4); print tolower($4)}' \
    | sed -E 's/^no$/none/' \
    | sort | uniq -c > "$tmpfile"

    while read -r count category; do
        COUNTS["$category,$run"]=$count
    done < "$tmpfile"

    rm "$tmpfile"
done

# Output combined results
{
    echo -e "category\trun1\trun2\trun3"
    for cat in "${CATEGORIES[@]}"; do
        r1=${COUNTS["$cat,1"]:-0}
        r2=${COUNTS["$cat,2"]:-0}
        r3=${COUNTS["$cat,3"]:-0}
        echo -e "$cat\t$r1\t$r2\t$r3"
    done
} > "$OUTFILE"

echo "✅ Done: Output saved to $OUTFILE"
