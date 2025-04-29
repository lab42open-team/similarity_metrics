import csv
import os

parent_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/initial_data/similarity_metrics/test_terrestrial_soil"

# Load marine samples mapping
marine_map = {}
with open(os.path.join(parent_dir, "soil_samples.tsv"), "r") as marine_file:
    reader = csv.reader(marine_file, delimiter="\t")
    for mgys, sample in reader:
        marine_map.setdefault(sample, set()).add(mgys)

# Process distances and compare 
with open(os.path.join(parent_dir, "output_SRR8488083_distances.tsv"), "r") as dist_file, \
     open(os.path.join(parent_dir, "filteredStudies_output_SRR8488083_distances.tsv"), "w", newline="") as out_file:

    reader = csv.reader(dist_file, delimiter="\t")
    writer = csv.writer(out_file, delimiter="\t")
    writer.writerow(["Sample1", "Sample2", "Distance"])  # Header

    for sample1, sample2, dist in reader:
        sample2_mgys = marine_map.get(sample2, None)

        # Case: sample2 is not in marine samples → save it
        if not sample2_mgys:
            writer.writerow([sample1, sample2, dist])
            continue

        # Case: sample2 exists in marine samples → check if MGYS matches sample1
        sample1_mgys = marine_map.get(sample1, set())  # if sample1 is in marine samples
        if not sample1_mgys or not sample1_mgys.intersection(sample2_mgys):
            writer.writerow([sample1, sample2, dist])
