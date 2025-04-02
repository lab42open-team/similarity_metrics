#!/usr/bin/python3.5

# script name: noise_injection_fun.py
# developed by: Nefeli Venetsianou
# description: 
    # Downsampling noise injection via multinomial distribution. Noise level is adjusted by ratio (percentage) of Counts that will be preserved as original. 
# framework: CCMRI
# last update: 23/01/2025

import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

### DOWNSAMPLING CountS NOISE INJECTION ###
def downsample_Counts_noise_injection(input_file, original_plot_file, downsample_ratios, base_output_dir):
    df = pd.read_csv(input_file, sep="\t") # non-RA file
    # Read original RA file - for plotting reasons only 
    original_plot_df = pd.read_csv(original_plot_file,sep="\t")
    # Downsample Counts
    # Iterate over each unique sample in df
    for ratio in downsample_ratios:
        ratio_dir = os.path.join(base_output_dir, "{}_ratio".format(ratio))
        # Create direcotry if it doesn't exists 
        if not os.path.exists(ratio_dir):
            os.makedirs(ratio_dir)
            logging.info("Created directory: {}".format(ratio_dir))
        # Create subdirectories for plots 
        output_plot_dir = os.path.join(ratio_dir, "plots")
        if not os.path.exists(output_plot_dir):
            os.makedirs(output_plot_dir)
            logging.info("Created plots directory: {}".format(output_plot_dir))
        # Create a copy of DataFrame 
        downsampled_df = df.copy()
        # Find unique samples
        unique_samples = downsampled_df["Run"].unique()
        # Determine ratio of samples to be affected - noise level
        num_samples_to_downsample = int(len(unique_samples) * ratio)
        samples_to_downsample = np.random.choice(unique_samples, num_samples_to_downsample, replace=False)
        # Iterate over samples to be downsampled
        for sample in samples_to_downsample:
            sample_mask = downsampled_df["Run"] == sample
            sample_Counts = downsampled_df[sample_mask]["Count"].values
            # Calculate total initial Count
            total_Count = sample_Counts.sum()
            # Calculate downsampled Count according to total initial Count and ratio
            downsampled_Count = total_Count * ratio 
            """Noise Injection via Multinomial Distribution"""
            # Apply multinomial downsampling to the selected sample based on probability distribution (pvalues)
            downsampled_sample_Counts = np.random.multinomial(n=int(downsampled_Count), pvals = sample_Counts / sample_Counts.sum())
            downsampled_df.loc[sample_mask, "Count"] = downsampled_sample_Counts
        # Normalize Counts to sum to 1 within each sample
        # Initialize dictionary to store total Count per sample
        sample_total_noisy = downsampled_df.groupby("Run")["Count"].sum().to_dict()
        downsampled_df["Count"] = downsampled_df.apply(lambda row: row["Count"] / sample_total_noisy[row["Run"]] if sample_total_noisy[row["Run"]] !=0 else 0, axis=1) 
        # Rename "Count" to "Relative_Abundance" in the final DataFrame
        downsampled_df.rename(columns={"Count": "Relative_Abundance"}, inplace=True)
        # Save downsampled version
        file_name = os.path.basename(input_file)
        output_file = os.path.join(ratio_dir, "downsampled_{}_ratio_{}".format(ratio, file_name))
        downsampled_df.to_csv(output_file, sep="\t", index=False)
        logging.info("Downsampled noisy version {} ratio, saved to {}.".format(ratio, output_file))
        output_plot_dir = os.path.join(ratio_dir, "plots")
        plot_results(original_plot_df, downsampled_df, ratio, input_file, output_plot_dir)

def plot_results(original_df, noisy_df, ratio, input_file, base_output_dir):
    """Plotting"""
    plt.figure(figsize=(15,5))
    # Plot original data
    plt.subplot(1,3,1)
    plt.hist(original_df["Relative_Abundance"], bins=50, alpha=0.7, label="Original Count")
    plt.title("Original Data Plot")
    plt.xlabel("Relative_Abundance")
    plt.ylabel("Frequency")
    # Plot noisy data 
    plt.subplot(1,3,2)
    plt.hist(noisy_df["Relative_Abundance"], bins=50, alpha=0.7, label="Noisy Count")
    plt.title("Noisy Data Plot")
    plt.xlabel("Relative_Abundance")
    plt.ylabel("Frequency")
    # Plot difference in Counts
    plt.subplot(1,3,3)
    difference = original_df["Relative_Abundance"] - noisy_df["Relative_Abundance"]
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
    parent_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/r_1000"
    input_file = os.path.join(parent_dir, "r_1000_f_aggregated_0.75_0.01_lf_v5.0_super_table.tsv") # non-RA file -1000 Randomly selected Runs
    original_plot_filePath = os.path.join(parent_dir, "normalized_data")  
    original_plot_file = os.path.join(original_plot_filePath, "normalized_r_1000_f_aggregated_0.75_0.01_lf_v5.0_super_table.tsv") # RA file
    base_output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/downsampling_noise"
    # Define downsampling ratios - fraction of reducing total abundance - level of noise
    downsample_ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
    downsample_Counts_noise_injection(input_file, original_plot_file, downsample_ratios, base_output_dir)

if __name__ == "__main__":
    main()