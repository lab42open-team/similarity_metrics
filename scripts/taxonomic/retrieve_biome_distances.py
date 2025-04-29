# File paths
BIOME_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/test_terrestrial_soil/soil_samples.tsv"
DIST_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/c_distances_batches_filtered_v5.0_SSU_ge_filtered.tsv"
OUTPUT_FILE="/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/test_terrestrial_soil/soil_samples_distances_output.tsv"

# Step 1: Load the marine sample IDs from the BIOME_FILE
marine_samples = set()
with open(BIOME_FILE, 'r') as biome_file:
    for line in biome_file:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            sample_id = parts[1]
            marine_samples.add(sample_id)

# Debugging: Print the first 5 marine samples to verify
print("First 5 marine samples (Sample IDs):")
for idx, sample in enumerate(marine_samples):
    if idx >= 5:
        break
    print(sample)

# Step 2: Open the DIST_FILE and process each line
with open(DIST_FILE, 'r') as dist_file:
    header = dist_file.readline().strip()  # Save header
    matches = []  # Store matches here
    
    for line in dist_file:
        parts = line.strip().split('\t')
        if len(parts) > 2:
            sample1, sample2, distance = parts[0], parts[1], parts[2]
            # Check if sample1 is in the marine_samples set
            if sample1 in marine_samples:
                matches.append(line.strip())  # Save the matching line

# Debugging: Print out how many matches we found
print(f"Number of matches found: {len(matches)}")

# Step 3: Append the matches to the output file
with open(OUTPUT_FILE, 'a') as output_file:
    # If the file is empty, you can choose to write the header again, otherwise, it will just append data.
    if output_file.tell() == 0:
        output_file.write(header + '\n')  # Write header if file is empty (first time appending)
    
    for match in matches:
        output_file.write(match + '\n')

print(f"All matches appended to {OUTPUT_FILE}")
