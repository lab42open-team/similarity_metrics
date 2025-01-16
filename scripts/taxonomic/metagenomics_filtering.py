import os 
import csv

# Define files
extracted_runs_file = "/ccmri/similarity_metrics/data/experiment_type/extracted_runs_data.tsv"
distances_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_filtered_v5.0_LSU_ge_filtered.tsv"
output_path = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/metagenomics/"
output_filename = "mtgn_{}".format(os.path.basename(distances_file))
output_file = os.path.join(output_path, output_filename)

# Load runs IDs from the extracted_runs_data.tsv
extracted_runs = set()
with open(extracted_runs_file, "r") as file:
    reader = csv.DictReader(file, delimiter="\t")
    for row in reader:
        # add each run_id ro the set
        extracted_runs.add(row["run_id"])

# Read distances tsv file and filter rows 
filtered_rows = []
with open(distances_file, "r") as file:
    reader = csv.DictReader(file, delimiter="\t")
    for row in reader:
        # Check if both Sample 1 and Sample 2 are in the extracted_runs set
        if row["Sample1"] in extracted_runs and row["Sample2"] in extracted_runs:
            filtered_rows.append(row)

# Write filtered data into a new tsv file 
with open(output_file, "w", newline="") as file:
    fieldnames = ["Sample1", "Sample2", "Distance"]
    writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t") 
    writer.writeheader() # write header
    writer.writerows(filtered_rows) # write filtered rows

print("Filtered rows saved to: {}".format(output_file))