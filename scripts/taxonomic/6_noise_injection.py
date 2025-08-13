#!/usr/bin/python3.5

# script name: noise_injection.py
# developed by: Nefeli Venetsianou
# description: 
    # Downsampling noise injection via multinomial distribution. Noise level is adjusted by ratio (percentage) of counts that will be preserved as original. 
    # Noise injection to the downsampled data, in order to create noisy versions. 
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
        output_plot_dir = os.path.join(ratio_dir, "plots")
        # Create necessary directories if they don't exist
        os.makedirs(ratio_dir, exist_ok=True)
        os.makedirs(output_plot_dir, exist_ok=True)
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
    input_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/d_1000_data/d_v4.1_SSU_ge_filtered.tsv"
    original_plot_file = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/d_1000_data/normalized/ra_d_v4.1_SSU_ge_filtered.tsv"
    base_output_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/noisy_versions/downsampled_noisy_version/run3/"
    # Define downsampling ratios - fraction of reducing total abundance - level of noise
    downsample_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
    downsample_counts_noise_injection(input_file, original_plot_file, downsample_ratios, base_output_dir)

if __name__ == "__main__":
    main() 
