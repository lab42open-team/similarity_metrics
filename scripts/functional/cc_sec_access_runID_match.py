#!/usr/bin/python3.5

import os

def extract_run_names(cc_tsv_dir, output_runs_file):
    """Iterate through all TSV files in cc_tsv_dir and extract unique Run Names (samples)."""
    
    run_names = set()  # Use a set to store unique Run Names

    # Iterate over all .tsv files in the directory
    for file_name in os.listdir(cc_tsv_dir):
        if file_name.endswith(".tsv"):  # Only process TSV files
            file_path = os.path.join(cc_tsv_dir, file_name)
            with open(file_path, "r") as f:
                header = f.readline().strip().split("\t")  # Read first line (header)

                # Exclude "GO", "description", and "category" columns (Runs start from the 4th column)
                run_names.update(header[3:])  

    # Save the unique Run Names to an output file
    with open(output_runs_file, "w") as f_out:
        for run in sorted(run_names):
            f_out.write(run + "\n")

    print(f"Extracted {len(run_names)} unique Run Names saved to {output_runs_file}")

def main():
    cc_tsv_dir = "/ccmri/similarity_metrics/data/cc_related_only/filtered_go_files/GO_abund/raw_data/data"
    output_runs_file = "/ccmri/similarity_metrics/data/cc_related_only/filtered_go_files/GO_abund/raw_data/cc_sec_access_runID_match.tsv"

    extract_run_names(cc_tsv_dir, output_runs_file)

if __name__ == "__main__":
    main()
