#!/usr/bin/python3.5

# script name: normalize_fun_data.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate relative abundances in raw data(super table).
    # RA Calculation can also be impelmented for downsampled data (data between 25-75% quartiles).
# framework: CCMRI
# last update: 22/01/2025

import os
import pandas as pd
import time
import logging
logging.basicConfig(level=logging.INFO)

def normalize_data(input_file):
    """Process input file and calculate relative abundances."""
    logging.info("Processing file: {}".format(input_file))
    # Read the input file into a DataFrame
    df = pd.read_csv(input_file, sep="\t")
    # Replace null values with 0
    df.fillna(0, inplace=True)
    # Calculate total counts per run
    total_run_counts = df.groupby("Run")["Count"].sum()
    # Prepare a list to store the relative abundance rows
    relative_abundance_rows = []
    # Iterate over rows in the DataFrame to calculate relative abundance
    for _, row in df.iterrows():
        go_term = row["GO"]
        run = row["Run"]
        count = row["Count"]
        # Get the total count for the run
        total_count_for_run = total_run_counts[run]
        # Calculate relative abundance
        relative_abundance = count / total_count_for_run if total_count_for_run != 0 else 0
        # Append the result as a dictionary
        relative_abundance_rows.append({
            "GO": go_term,
            "Run": run,
            "Relative_Abundance": relative_abundance
        })
     # Create a new DataFrame from the list of rows
    relative_abundance_df = pd.DataFrame(relative_abundance_rows)
    # Adjust relative abundances to ensure they sum to 1
    for run in relative_abundance_df["Run"].unique():
        run_mask = relative_abundance_df["Run"] == run
        run_total = relative_abundance_df.loc[run_mask, "Relative_Abundance"].sum()
        if run_total != 1.0:  # Adjust the last value to make the sum exactly 1
            diff = 1.0 - run_total
            last_index = relative_abundance_df[run_mask].index[-1]
            relative_abundance_df.at[last_index, "Relative_Abundance"] += diff
    # Sort the DataFrame by the "Run" column
    relative_abundance_df.sort_values(by="Run", inplace=True)
    # Reorder the columns
    relative_abundance_df = relative_abundance_df[["GO", "Run", "Relative_Abundance"]]
    return relative_abundance_df

def write_output(output_dir, input_file, dataframe):
    """Write the normalized data for a single file to the output directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Construct the output file name
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, "normalized_{}".format(base_name))
    # Save the DataFrame to a new .tsv file
    dataframe.to_csv(output_file, sep="\t", index=False, float_format="%.8f")
    logging.info("Output saved to: {}".format(output_file))

def process_folder(input_folder, output_dir):
    """Process all .tsv files in the input folder and save their normalized data."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Process each .tsv file in the folder
    for file in os.listdir(input_folder):
        if file.endswith(".tsv"):
            input_file = os.path.join(input_folder, file)
            file_start_time = time.time()
            # Normalize the data for the file
            normalized_df = normalize_data(input_file)
            # Write the normalized data to the output directory
            write_output(output_dir, input_file, normalized_df)
            file_end_time = time.time()
            logging.info("Execution time for {} is {} seconds.".format(file, file_end_time-file_start_time))

def main():
   # Define input folder and output directory from RAW data
    #input_folder = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/final_CCMRI_GO-slim"
    #output_dir = os.path.join(input_folder, "normalized_aggr_data")
    
    # Define input - output directories from DOWN-FILTERED data
    #input_folder = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/d_filtered"
    #output_dir = os.path.join(input_folder, "normalized_data")
    
    # Define input - output for 1000 Randomly selected Runs
    input_folder = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/r_1000"
    output_dir = os.path.join(input_folder, "normalized_data")
    
    # Record start time
    folder_start_time = time.time()
    # Process the input folder
    process_folder(input_folder, output_dir)
    # Record end time
    folder_end_time = time.time()
    logging.info("Total execution time: {:.2f} seconds.".format(folder_end_time - folder_start_time))

if __name__ == "__main__":
    main()