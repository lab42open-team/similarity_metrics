#!/usr/bin/python3.5


import json
import csv
from collections import defaultdict

# === File paths ===
json_file_path = "../results/phi4_14b-q8_0/LLM_pairwise_output_phi4_14b-q8_0_run1.json"       # Path to your JSON file
tsv_file_path = "../datasets/16.tsv"       # Path to your input TSV file
output_file_path = "../results/16_id_pair_counts.tsv"  # Path for the output TSV

# --- Load JSON with fix for '][' concatenation ---
with open(json_file_path) as f:
    raw = f.read()

# fix: replace ][ with , so arrays merge
raw = raw.replace("][", ",")

pairs = json.loads(raw)

# --- Load TSV IDs ---
tsv_ids = []
with open(tsv_file_path) as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        tsv_ids.append(row[0])

# --- Count occurrences ---
id_counts = defaultdict(int)
for pair in pairs:
    id1 = pair["Study_ID1"]
    id2 = pair["Study_ID2"]
    if id1 in tsv_ids:
        id_counts[id1] += 1
    if id2 in tsv_ids:
        id_counts[id2] += 1

# --- Write output ---
with open(output_file_path, "w") as out_file:
    writer = csv.writer(out_file, delimiter="\t")
    for study_id in tsv_ids:
        writer.writerow([study_id, id_counts.get(study_id, 0)])

print("✅ Parsed", len(pairs), "pairs")
print("Output written to", output_file_path)