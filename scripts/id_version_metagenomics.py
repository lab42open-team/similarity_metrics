#!/usr/bin/python3.5

# script name: id_version_metagenomics.py
# developed by: Nefeli Venetsianou
# description: 
    # Isolate run ID and corresponding version pipeline from json files. 
# framework: CCMRI
# last update: 15/11/2024

import os 
import json
import csv

# Define folder path where JSON files are stored
folder_path = "/ccmri/similarity_metrics/data/experiment_type/runs_v1"

# Prepare list to store extracted data
extracted_data = []

# Loop through all the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith("json"):
        file_path = os.path.join(folder_path, filename)
        
        # Load json file
        with open(file_path, "r") as f:
            data = json.load(f)
            
            # Extract run ID and version pipeline ID
            for entry in data["data"]:
                run_id = entry["id"]
                # Check if the pipelines list exists and is not empty
                pipelines = entry['relationships'].get('pipelines', {}).get('data', [])
                if pipelines:
                    pipeline_id = pipelines[0]['id']  # Extract the pipeline version ID
                else:
                    pipeline_id = None  # If no pipeline is present, set as None
                
                # Append extracted information to the list 
                extracted_data.append({"run_id":run_id, "pipeline_id":pipeline_id})
                
# Define output TSV file 
output_file_path = "/ccmri/similarity_metrics/data/experiment_type/extracted_runs_data.tsv"

# Write extracted data to tsv file 
with open(output_file_path, "w") as tsvfile:
    # Create CSV writer with tab delimiter 
    writer = csv.DictWriter(tsvfile, fieldnames=["run_id", "pipeline_id"], delimiter="\t")
    # Write header
    writer.writeheader()
    # Write rows 
    writer.writerows(extracted_data)
    
print("Data has been written to: {}".format(output_file_path))

