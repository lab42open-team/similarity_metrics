#!/usr/bin/python3.5

"""Approach A: If you have samples, please use this approach. 

# script name: mined_info_retrieval.py
# developed by: Nefeli Venetsianou
# description: 
    # Retrieve all relative information for each Run1, Run2 automatically. 
    # Run1, Run2, Distance tsv file is created via "filter_biomeVSall_distances.py" script.
    # study_id, study_name, study_abstract, biome_info is retrieved from mined_info_MGYS*.txt files. 
    # The output is used for manual curation. 
# framework: CCMRI
# last update: 24/06/2025

import os
import glob
import csv
from collections import defaultdict


# === Define Global variables - file paths === # 
DISTANCES_FILE = "/ccmri/similarity_metrics/data/taxonomic/test_dataset/final_testing/03.07.2025_test/root:Host-associated:Plants/top_20_distances_plantsVSall_random_5.tsv"
distances_filename = os.path.basename(DISTANCES_FILE)
MAPPING_FILE = "/ccmri/similarity_metrics/data/taxonomic/raw_data/studies_samples/biome_info/merged.tsv"
MGYS_ROOT = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies/"
OUTPUT_FILE = os.path.join("/ccmri/similarity_metrics/data/taxonomic/test_dataset/final_testing/03.07.2025_test/root:Host-associated:Plants/", "enriched_{}".format(distances_filename))

# Function to load MGYS_id - sample_id from merged.tsv
def load_sample_to_mgys(path):
    mapping = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sample = row.get("run_id")
            mgys = row.get("study_id")
            if sample and mgys:
                mapping[sample.strip()] = mgys.strip()
    return mapping

# Define parameter to store metadata
metadata_cache = {}

# === Function to load metadata from mined info === # 
def get_mgys_metadata(mgys_id):
    if mgys_id in metadata_cache:
        return metadata_cache[mgys_id]
    meta = {
        "study_id": mgys_id,
        "study_name": "NA",
        "study_abstract": "NA",
        "biome_info": "NA"
    }
    mgys_path = os.path.join(MGYS_ROOT, mgys_id)
    if not os.path.isdir(mgys_path):
        return None
    txts = glob.glob(os.path.join(mgys_path, "mined_info_*.txt"))
    if not txts:
        return None
    with open(txts[0]) as f:
        for line in f:
            for key in meta:
                if line.lower().startswith(key + ""):
                    parts = line.strip().split("\t", 1)
                    if len(parts) > 1:
                        meta[key] = parts[1].strip()
    metadata_cache[mgys_id] = meta
    return meta

# === Function to add hyperlink in MGYS and run === #
def hyperlink_formula(mgys_id):
    url = f"https://www.ebi.ac.uk/metagenomics/studies/{mgys_id}"
    return f'=HYPERLINK("{url}"; "{mgys_id}")'
"""
def run_hyperlink(run_id):
    url = f"https://www.ebi.ac.uk/metagenomics/runs/{run_id}"
    return f'=HYPERLINK("{url}"; "{run_id}")'
"""

def main():
    sample_to_mgys = load_sample_to_mgys(MAPPING_FILE)
    blocks = defaultdict(list)
    with open(DISTANCES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split()
            if len(parts) != 3: continue
            run1, run2, distance = parts
            blocks[run1].append((run2, distance))
    with open(OUTPUT_FILE, "w", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writerow([
            "Run1", "Run2", "Distance",
            "Study_id", "Study_name", "Study_abstract", "Biome_info", "Comments"
        ])
        for run1, comparisons in blocks.items():
            # Summary row for Run1
            mgys1 = sample_to_mgys.get(run1)
            meta1 = get_mgys_metadata(mgys1) if mgys1 else None
            if meta1:
                writer.writerow([
                    run_hyperlink(run1), "-", "-",
                    hyperlink_formula(meta1['study_id']),
                    meta1['study_name'], meta1['study_abstract'], meta1['biome_info'], ""
                ])
            else:
                writer.writerow([run1, "-", "-", "NA", "NA", "NA", "NA", "NA"])

            # Data rows for each Run2
            for run2, distance in comparisons:
                mgys2 = sample_to_mgys.get(run2)
                meta2 = get_mgys_metadata(mgys2) if mgys2 else None
                if meta2:
                    writer.writerow([
                        run_hyperlink(run1), run_hyperlink(run2), distance.replace(".", ","),
                        hyperlink_formula(meta2['study_id']),
                        meta2['study_name'], meta2['study_abstract'], meta2['biome_info']
                    ])
                else:
                    writer.writerow([
                        run1, run2, distance,
                        "NA", "NA", "NA", "NA", "NA"
                    ])
    print(f"Mined ino retrieved. Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
"""


"""Approach B: If you have already moved to the study level similarity, then use this approach."""
# script name: study_info_retrieval.py
# developed by: Nefeli Venetsianou
# description:
#   Retrieve all relative information for each Study1, Study2 pair automatically.
#   Distance tsv file (Study1, Study2, Distance) is used as input.
#   Metadata (study_name, abstract, biome) is retrieved from mined_info_MGYS*.txt files.
#   Output is a TSV enriched for manual curation.
# framework: CCMRI
# last update: 10/07/2025

import os
import glob
import csv
from collections import defaultdict

# === Define Global variables - file paths === #
DISTANCES_FILE = "/ccmri/similarity_metrics/data/algorithm_testing/10.07.2025_test/environmental_terrestrial/mining_functional_distances.tsv"
distances_filename = os.path.basename(DISTANCES_FILE)
MGYS_ROOT = "/ccmri/data/mgnify/frozen_february_2024/2024_1_17/harvested_mgnify_studies/"
OUTPUT_FILE = os.path.join(os.path.dirname(DISTANCES_FILE), f"enriched_{distances_filename}")

# Metadata cache
metadata_cache = {}

# === Load metadata for a study === #
def get_study_metadata(mgys_id):
    if mgys_id in metadata_cache:
        return metadata_cache[mgys_id]
    meta = {
        "study_id": mgys_id,
        "study_name": "NA",
        "study_abstract": "NA",
        "biome_info": "NA"
    }
    mgys_path = os.path.join(MGYS_ROOT, mgys_id)
    if not os.path.isdir(mgys_path):
        return None
    txts = glob.glob(os.path.join(mgys_path, "mined_info_*.txt"))
    if not txts:
        return None
    with open(txts[0]) as f:
        for line in f:
            for key in meta:
                if line.lower().startswith(key + ""):
                    parts = line.strip().split("\t", 1)
                    if len(parts) > 1:
                        meta[key] = parts[1].strip()
    metadata_cache[mgys_id] = meta
    return meta

# === Add hyperlink to MGYS ID === #
def hyperlink_formula(mgys_id):
    url = f"https://www.ebi.ac.uk/metagenomics/studies/{mgys_id}"
    return f'=HYPERLINK("{url}"; "{mgys_id}")'

# === Main Function === #
def main():
    pairs = []
    with open(DISTANCES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Skip header if it looks like one
            if "study1" in line.lower() and "study2" in line.lower():
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            study1, study2, distance = parts
            pairs.append((study1.strip(), study2.strip(), distance.strip()))

    with open(OUTPUT_FILE, "w", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writerow([
            "Study1", "Study2", "Distance",
            "Study1_name", "Study1_abstract", "Study1_biome",
            "Study2_name", "Study2_abstract", "Study2_biome",
            "Comments"
        ])

        for study1, study2, distance in pairs:
            meta1 = get_study_metadata(study1)
            meta2 = get_study_metadata(study2)

            row = [
                hyperlink_formula(study1) if meta1 else study1,
                hyperlink_formula(study2) if meta2 else study2,
                distance.replace(".", ","),
                meta1['study_name'] if meta1 else "NA",
                meta1['study_abstract'] if meta1 else "NA",
                meta1['biome_info'] if meta1 else "NA",
                meta2['study_name'] if meta2 else "NA",
                meta2['study_abstract'] if meta2 else "NA",
                meta2['biome_info'] if meta2 else "NA",
                ""
            ]
            writer.writerow(row)

    print(f"Study metadata retrieved. Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
