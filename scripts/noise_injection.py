#!/usr/bin/python3.5

# script name: noise_injection.py
# developed by: Nefeli Venetsianou
# description: 
    # Noise injection to the downsampled data, in order to create noisy versions. 
    # Gaussian Noise Injection is selected, at different levels of noise, by adjusting standard deviation.  
# framework: CCMRI
# last update: 26/06/2024

import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def gaussian_noise_injection(input_file, output_dir):
    df = pd.read_csv(input_file, sep="\t")
    # Apply Gaussian noise injection
    # Mean is set to 0 in order to create noise centered around the original values. 
    mean_noise = 0
    # To change the intensity (level) of noise, adjust the standard deviation.
    # Higher std_dev_noise leads to noisier dataset, while lower std_dev_noise leads to a dataset closer to the original. 
    std_dev_noise = 5
    noise = np.random.normal(mean_noise, std_dev_noise, len(df["Count"]))
    # Create a copy of df
    noisy_counts_df = df.copy()
    # Add noise to the "Count" column 
    noisy_counts_df["Count"] = df["Count"] + noise
    # Ensure counts are non-negative
    noisy_counts_df["Count"] = noisy_counts_df["Count"].clip(lower=0) 
    # Normalize counts (sum to 1)
    # Initialize dictionary to store total count per sample
    sample_total_noisy = noisy_counts_df.groupby("Sample")["Count"].sum().to_dict()
    noisy_counts_df["Count"] = noisy_counts_df.apply(lambda row: row["Count"] / sample_total_noisy[row["Sample"]] if sample_total_noisy[row["Sample"]] !=0 else 0, axis=1)
    # Save noisy version
    # Extract filename from input file 
    file_name = os.path.basename(input_file)
    # Remove "ra_" from file_name
    filename = file_name[3:]
    # Construct output file_path 
    output_file = os.path.join(output_dir, "noisy_{}_stdDev_".format(std_dev_noise) + filename)
    # Save noisy version 
    noisy_counts_df.to_csv(output_file, sep="\t", index=False)    
    logging.info("Noisy version of {} level, saved to: {}".format(std_dev_noise, output_file))
    
    # Plotting Data
    plt.figure(figsize=(15,5))
    # Plot original data
    plt.subplot(1,3,1)
    plt.hist(df["Count"], bins=50, alpha=0.7, label="Original Count")
    plt.title("Original Data Plot")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    # Plot Gaussian noise
    plt.subplot(1,3,2)
    plt.hist(noise, bins=50, alpha=0.7, label="Gaussian noise")
    plt.title("Gaussian Noise Plot")
    plt.xlabel("Gaussian noise")
    plt.ylabel("Frequency")
    
    # Plot noisy data
    plt.subplot(1,3,3)
    plt.hist(noisy_counts_df["Count"], bins=50, alpha=0.7, label="Noisy count")
    plt.title("Noisy Data Plot")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    # Save plots
    original_plot_path = os.path.join(output_dir, "original_data_plot.png")
    gaussian_noise_plot_path = os.path.join(output_dir, "gaussian_noise_plot.png")
    noisy_plot_path = os.path.join(output_dir, "noisy_data_plot.png")
    # Save full figure
    # Remove .tsv suffix
    out_filename = filename[:-4]
    # Construct path 
    img_plot_path = os.path.join(output_dir, "histograms_n{}_stdDev_".format(std_dev_noise) + out_filename + ".png")
    plt.savefig(img_plot_path)
    logging.info("Plot saved to: {}".format(img_plot_path))
    
def impulse_noise_injection(input_file, output_dir):
    # Read input file
    df = pd.read_csv(input_file, sep="\t")
    # Create a copy of DataFrame
    noisy_counts_df = df.copy()
    # Probability of injecting impulse noise to counts - set to 1 to apply for all counts
    prob_impulse = 1
    # Lower bound value for impulse noise 
    low_val = 0
    # Higher bound value for impulse noise
    high_val = df["Count"].max()
    # Generate random impulse noise values # p represents the probability of choosing each low-high val
    random_values = np.random.choice([low_val, high_val], size=len(df["Count"]), p=[0.9,0.1])
    # Generate a mask fro where to apply the impulse noise
    mask = np.random.choice([True, False], size=len(df["Count"]), p=[1-prob_impulse, prob_impulse])
    # Apply the impulse noise to the Count column
    noisy_counts_df.loc[mask, "Count"] = random_values[mask]
    # Normalize counts (sum to 1)
    # Initialize dictionary to store total count per sample
    sample_total_noisy = noisy_counts_df.groupby("Sample")["Count"].sum().to_dict()
    noisy_counts_df["Count"] = noisy_counts_df.apply(lambda row: row["Count"] / sample_total_noisy[row["Sample"]] if sample_total_noisy[row["Sample"]] !=0 else 0, axis=1)
    # Save noisy version
    # Extract filename from input file 
    file_name = os.path.basename(input_file)
    # Remove "ra_" from file_name
    filename = file_name[3:]
    # Construct output file_path 
    output_file = os.path.join(output_dir, "noisy_impulse_" + filename)
    # Save noisy version 
    noisy_counts_df.to_csv(output_file, sep="\t", index=False)    
    logging.info("Impulse noisy version saved to: {}".format(output_file))
    
    # Plotting Data
    plt.figure(figsize=(15,5))
    # Plot original data
    plt.subplot(1,3,1)
    plt.hist(df["Count"], bins=50, alpha=0.7, label="Original Count")
    plt.title("Original Data Plot")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    # Plot Gaussian noise
    plt.subplot(1,3,2)
    plt.hist(random_values, bins=50, alpha=0.7, label="Impulse noise")
    plt.title("Gaussian Noise Plot")
    plt.xlabel("Gaussian noise")
    plt.ylabel("Frequency")
    
    # Plot noisy data
    plt.subplot(1,3,3)
    plt.hist(noisy_counts_df["Count"], bins=50, alpha=0.7, label="Noisy count")
    plt.title("Noisy Data Plot")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    # Save plots
    original_plot_path = os.path.join(output_dir, "original_data_plot.png")
    gaussian_noise_plot_path = os.path.join(output_dir, "impulse_noise_plot.png")
    noisy_plot_path = os.path.join(output_dir, "noisy_data_plot.png")
    # Save full figure
    # Remove .tsv suffix
    out_filename = filename[:-4]
    # Construct path 
    img_plot_path = os.path.join(output_dir, "histograms_impulse_" + out_filename + ".png")
    plt.savefig(img_plot_path)
    logging.info("Plot saved to: {}".format(img_plot_path))
    
def main():
    input_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/downsampled_data/1000_samples/normalized/ra_v4.1_LSU_ge_filtered.tsv"
    # output to be defined according to std_noise_level picked
    gaussian_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/gaussian_noisy_data/d_1000/std_noise_level_5"
    impulse_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noisy_versions/impulse_noisy_data"
    # Apply Gaussian injection noise
    #gaussian_noise_injection(input_file, gaussian_output_dir)    
    impulse_noise_injection(input_file, impulse_output_dir)
    
if __name__ == "__main__":
    main()
    