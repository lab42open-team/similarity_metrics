# Script developed by: Nefeli Venetsianou
# Last update: 05.11.2025
# Aim:
#   Iterate through each JSON file in the parent directory
#   Retrieve Secondary accession number, experiment type, and study ID
#   Save summary as TSV

import json
import os
import pandas as pd

# Set directories
parent_dir = "/ccmri/similarity_metrics/data/experiment_type/runs_v1"
output_dir = "/ccmri/similarity_metrics/data/experiment_type"

results = []

for filename in os.listdir(parent_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(parent_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

            # Iterate through all data items in the JSON
            for item in data.get("data", []):
                attributes = item.get("attributes", {})
                relationships = item.get("relationships", {})

                # Extract fields safely
                secondary_accession = attributes.get("secondary-accession")
                experiment_type = attributes.get("experiment-type")

                study_info = relationships.get("study")
                if study_info and isinstance(study_info, dict):
                    study_data = study_info.get("data")
                    if study_data and isinstance(study_data, dict):
                        study_id = study_data.get("id")
                    else:
                        study_id = None
                else:
                    study_id = None

                results.append({
                    "filename": filename,
                    "study_id": study_id,
                    "secondary_accession": secondary_accession,
                    "experiment_type": experiment_type
                })

# Convert to DataFrame for easier handling and export
df = pd.DataFrame(results)

# Save to TSV
output_file = os.path.join(output_dir, "experiment_type_summary.tsv")
df.to_csv(output_file, sep="\t", index=False)

print(f"Process completed. Experiment type summary saved to: {output_file}")
