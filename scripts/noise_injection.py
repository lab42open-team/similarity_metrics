#!/usr/bin/python3.5

# script name: noise_injection.py
# developed by: Nefeli Venetsianou
# description: 
    # Downsampling noise injection via multinomial distribution. Noise level is adjusted by ratio (percentage) of counts that will be preserved as original. 
    ### TESTED ###
    # Noise injection to the downsampled data, in order to create noisy versions. 
    # Gaussian Noise Injection is selected, at different levels of noise, by adjusting standard deviation.  
    # Impulse Noise Injection increases/decreases counts per sample, in order to randomly add noise. Noise level van be adjusted by changing the probability of injecting noise.
# framework: CCMRI
# last update: 25/07/2024

import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

### DOWNSAMPLING COUNTS NOISE INJECTION ###
def downsample_counts_noise_injection(input_file, original_plot_file, downsample_ratios, base_output_dir):
    df = pd.read_csv(input_file, sep="\t")
    # Read original RA file - for plotting reasons only 
    original_plot_df = pd.read_csv(original_plot_file,sep="\t")
    # Downsample counts
    # Iterate over each unique sample in df
    for ratio in downsample_ratios:
        ratio_dir = os.path.join(base_output_dir, "{}_ratio".format(ratio))
        # Create a copy of DataFrame 
        downsampled_df = df.copy()
        # Find unique samples
        unique_samples = downsampled_df["Sample"].unique()
        # Determine ratio of samples to be affected - noise level
        num_samples_to_downsample = int(len(unique_samples) * ratio)
        samples_to_downsample = np.random.choice(unique_samples, num_samples_to_downsample, replace=False)
        # Iterate over samples to be downsampled
        for sample in samples_to_downsample:
            sample_mask = downsampled_df["Sample"] == sample
            sample_counts = downsampled_df[sample_mask]["Count"].values
            # Calculate total initial count
            total_count = sample_counts.sum()
            # Calculate downsampled count according to total initial count and ratio
            downsampled_count = total_count * ratio 
            # Apply multinomial downsampling to the selected sample based on probability distribution (pvalues)
            downsampled_sample_counts = np.random.multinomial(n=int(downsampled_count), pvals = sample_counts / sample_counts.sum())
            downsampled_df.loc[sample_mask, "Count"] = downsampled_sample_counts
        # Normalize counts to sum to 1 within each sample
        # Initialize dictionary to store total count per sample
        sample_total_noisy = downsampled_df.groupby("Sample")["Count"].sum().to_dict()
        downsampled_df["Count"] = downsampled_df.apply(lambda row: row["Count"] / sample_total_noisy[row["Sample"]] if sample_total_noisy[row["Sample"]] !=0 else 0, axis=1) 
        # Save downsampled version
        file_name = os.path.basename(input_file)
        output_file = os.path.join(ratio_dir, "downsampled_{}_ratio_{}".format(ratio, file_name))
        downsampled_df.to_csv(output_file, sep="\t", index=False)
        logging.info("Downsampled noisy version {} ratio, saved to {}.".format(ratio, output_file))
        output_plot_dir = os.path.join(ratio_dir, "plots")
        plot_results(original_plot_df, downsampled_df, ratio, input_file, output_plot_dir)
                
def plot_results(original_df, noisy_df, ratio, input_file, base_output_dir):
    plt.figure(figsize=(15,5))
    # Plot original data
    plt.subplot(1,3,1)
    plt.hist(original_df["Count"], bins=50, alpha=0.7, label="Original Count")
    plt.title("Original Data Plot")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    # Plot noisy data 
    plt.subplot(1,3,2)
    plt.hist(noisy_df["Count"], bins=50, alpha=0.7, label="Noisy Count")
    plt.title("Noisy Data Plot")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    # Plot difference in counts
    plt.subplot(1,3,3)
    difference = original_df["Count"] - noisy_df["Count"]
    plt.hist(difference, bins=50, alpha=0.7, label="Count Difference")
    plt.title("Difference Plot")
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.tight_layout()
    # Save plots 
    plot_file_name = "histograms_{}_ratio_{}.png".format(ratio, os.path.basename(input_file)[:-4])
    plot_path = os.path.join(base_output_dir, plot_file_name)
    plt.savefig(plot_path)
    logging.info("Plot saved to: {}".format(plot_path))
    plt.close()
     
def main():
    input_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/d_1000_data/d_v4.1_LSU_ge_filtered.tsv"
    original_plot_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/d_1000_data/normalized/ra_d_v4.1_LSU_ge_filtered.tsv"
    base_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/downsampled_noisy_version"
    # Define downsampling ratios - fraction of reducing total abundance - level of noise
    downsample_ratios = [0.1, 0.25, 0.5, 0.75]
    downsample_counts_noise_injection(input_file, original_plot_file, downsample_ratios, base_output_dir)

if __name__ == "__main__":
    main() 

"""
### GAUSSIAN & IMPULSE NOISE INJECTION ###
def gaussian_noise_injection(input_file, original_plot_file, output_dir, gaussian_std_dev_noise):
    df = pd.read_csv(input_file, sep="\t")
    # Read original RA file - for plotting reasons only 
    original_plot_df = pd.read_csv(original_plot_file,sep="\t")
    # Apply Gaussian noise injection
    # Mean is set to 0 in order to create noise centered around the original values. 
    mean_noise = 0
    noise = np.random.normal(mean_noise, gaussian_std_dev_noise, len(df["Count"]))
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
    # Construct output file_path 
    output_file = os.path.join(output_dir, "noisy_{}_stdDev_".format(gaussian_std_dev_noise) + file_name)
    # Save noisy version 
    noisy_counts_df.to_csv(output_file, sep="\t", index=False)    
    logging.info("Noisy version of {} level, saved to: {}".format(gaussian_std_dev_noise, output_file))
    
    # Plotting Data
    plt.figure(figsize=(15,5))
    # Plot original data
    plt.subplot(1,3,1)
    plt.hist(original_plot_df["Count"], bins=50, alpha=0.7, label="Original Count")
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
    out_filename = file_name[:-4]
    # Construct path 
    img_plot_path = os.path.join(output_dir, "histograms_n{}_stdDev_".format(gaussian_std_dev_noise) + out_filename + ".png")
    plt.savefig(img_plot_path)
    logging.info("Plot saved to: {}".format(img_plot_path))
    
def impulse_noise_injection(input_file, original_plot_file, output_dir, impulse_noise_level):
    # Read input file
    df = pd.read_csv(input_file, sep="\t")
    # Read original RA file - for plotting reasons only 
    original_plot_df = pd.read_csv(original_plot_file, sep="\t")
    # Create a copy of DataFrame
    noisy_counts_df = df.copy()
    # Lower bound value for impulse noise 
    low_val = 0
    # Higher bound value for impulse noise
    high_val = df["Count"].max()
    # Generate random impulse noise values # p represents the probability of choosing each low-high val
    random_values = np.random.choice([low_val, high_val], size=len(df["Count"]), p=[0.5,0.5])
    # Generate a mask fro where to apply the impulse noise
    mask = np.random.choice([True, False], size=len(df["Count"]), p=[impulse_noise_level, 1 - impulse_noise_level])
    # Apply the impulse noise to the Count column
    noisy_counts_df.loc[mask, "Count"] = random_values[mask]
    # Normalize counts (sum to 1)
    # Initialize dictionary to store total count per sample
    sample_total_noisy = noisy_counts_df.groupby("Sample")["Count"].sum().to_dict()
    noisy_counts_df["Count"] = noisy_counts_df.apply(lambda row: row["Count"] / sample_total_noisy[row["Sample"]] if sample_total_noisy[row["Sample"]] !=0 else 0, axis=1)
    # Save noisy version
    # Extract filename from input file 
    file_name = os.path.basename(input_file)
    # Construct output file_path 
    output_file = os.path.join(output_dir, "noisy_{}_impL_{}".format(impulse_noise_level, file_name))
    # Save noisy version 
    noisy_counts_df.to_csv(output_file, sep="\t", index=False)    
    logging.info("Impulse noisy version of level {} saved to: {}".format(impulse_noise_level, output_file))
    
    # Plotting Data
    plt.figure(figsize=(15,5))
    # Plot original data
    plt.subplot(1,3,1)
    plt.hist(original_plot_df["Count"], bins=50, alpha=0.7, label="Original Count")
    plt.title("Original Data Plot")
    plt.xlabel("Count")
    plt.ylabel("Frequency")
    # Plot Gaussian noise
    plt.subplot(1,3,2)
    plt.hist(random_values, bins=50, alpha=0.7, label="Impulse Noise")
    plt.title("Impulse Noise Plot")
    plt.xlabel("Impulse noise")
    plt.ylabel("Frequency")
    
    # Plot noisy data
    plt.subplot(1,3,3)
    plt.hist(noisy_counts_df["Count"], bins=50, alpha=0.7, label="Noisy Count")
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
    out_filename = file_name[:-4]
    # Construct path 
    img_plot_path = os.path.join(output_dir, "histograms_{}_impL_".format(impulse_noise_level) + out_filename + ".png")
    plt.savefig(img_plot_path)
    logging.info("Plot saved to: {}".format(img_plot_path))
    
def main():
    input_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/downsampled_data/1000_samples/d_v4.1_SSU_ge_filtered.tsv"
    original_plot_file = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/downsampled_data/1000_samples/normalized/ra_d_v4.1_SSU_ge_filtered.tsv"
    # output to be defined according to std_noise_level picked
    gaussian_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/gaussian_noisy_data/d_1000/0.2_stdDev"
    impulse_output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/impulse_noisy_data/d_1000/0.03_noiseLevel"
    # Define Noise Level per noise injection
    # To change the intensity of noise, adjust the standard deviation. - Noise Level
    # Higher std_dev_noise leads to noisier dataset, while lower std_dev_noise leads to a dataset closer to the original. 
    gaussian_std_dev_noise = 0.2
    # Probability of injecting impulse noise to counts - Noise Level
    impulse_noise_level = 0.03
    # Apply Gaussian injection noise
    gaussian_noise_injection(input_file, original_plot_file, gaussian_output_dir, gaussian_std_dev_noise)    
    impulse_noise_injection(input_file, original_plot_file, impulse_output_dir, impulse_noise_level)
    
if __name__ == "__main__":
    main()
"""