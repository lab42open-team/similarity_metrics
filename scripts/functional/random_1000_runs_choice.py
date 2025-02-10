import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def random_1000_runs(input_file, output_dir):
    df = pd.read_csv(input_file, sep="\t") 
    # Randomly pick 1000 Runs to proceed noise injection
    unique_runs = df["Run"].unique()
    if len(unique_runs) > 1000:
        selected_runs = np.random.choice(unique_runs, 1000, replace=False)
        logging.info("Randomly selected 1000 Runs.")
    else:
        logging.warning("File contains less than 1000 unique runs. No random choise is implemented.")
        selected_runs = unique_runs
     # Filter the DataFrame to include only the selected runs
    downsampled_df = df[df["Run"].isin(selected_runs)]
     # Save downsampled version
    file_name = os.path.basename(input_file)
    output_file = os.path.join(output_dir, "r_1000_{}".format(file_name))
    downsampled_df.to_csv(output_file, sep="\t", index=False)
    logging.info(f"Downsampled version saved to {output_file}")

if __name__ == "__main__":
    input_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/d_filtered/filtered_data_v4.1.tsv"
    output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/d_filtered/r_1000"
    random_1000_runs(input_file, output_dir)
    
    