# script name: match_relatedness.py
# developed by: Nefeli Venetsianou
# description: 
    # Match manual check of study similarity with LLM assessed relatedness.
# framework: CCMRI
# last update: 23/07/2025


import pandas as pd
import os

# Input file paths
parent_dir = "/ccmri/merged_total_studies_similarity_files"
manual_file = os.path.join(parent_dir, "sample_to_study_manual_check.tsv")
relatedness_file = os.path.join(parent_dir, "relatedness_with_distance_final_dedup.tsv")
output_file = os.path.join(parent_dir, "matched_relatedness_output.tsv")

# Load the manual check file
manual_df = pd.read_csv(manual_file, sep='\t')

# Load the relatedness results
relatedness_df = pd.read_csv(relatedness_file, sep='\t')

# Normalize column names just in case
manual_df.columns = [col.strip() for col in manual_df.columns]
relatedness_df.columns = [col.strip() for col in relatedness_df.columns]

# Prepare the output rows
output_rows = []

# Loop over each pair in the manual file
for _, row in manual_df.iterrows():
    study1 = row['Study1']
    study2 = row['Study2']
    manual_answer = row['Manual Answer']

    # Find a match in either direction
    matched = relatedness_df[
        ((relatedness_df['Study_ID1'] == study1) & (relatedness_df['Study_ID2'] == study2)) |
        ((relatedness_df['Study_ID1'] == study2) & (relatedness_df['Study_ID2'] == study1))
    ]

    if not matched.empty:
        for _, match in matched.iterrows():
            output_rows.append([
                study1,
                study2,
                match['run1'],
                match['run2'],
                match['run3'],
                manual_answer,
                match['Distance'],
                match['SimCategory']
            ])

# Create a DataFrame from the matched results
output_df = pd.DataFrame(output_rows, columns=[
    'Study1', 'Study2', 'run1', 'run2', 'run3', 'Manual_Answer', 'Distance', 'SimCategory'
])

# Save to TSV
output_df.to_csv(output_file, sep='\t', index=False)

print(f"Done. Output written to: {output_file}")
