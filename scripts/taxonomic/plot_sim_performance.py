#!/usr/bin/python3.5

# script name: plot_sim_performance.py
# developed by: Nefeli Venetsianou
# description: 
    # Plot similarity metrics performance.
    # Please check "### TO BE ADJUSTED ###" part for input file name changes. 
# framework: CCMRI
# last update: 26/07/2024

import os, re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)

def load_and_combine_data(euclidean_input_file, cosine_input_file, jsd_input_file):
    df_euclidean = pd.read_csv(euclidean_input_file, sep="\t")
    df_euclidean["Metric"] = "Euclidean"
    df_cosine = pd.read_csv(cosine_input_file, sep="\t")
    df_cosine["Metric"] = "Cosine"
    df_jensen_shannon = pd.read_csv(jsd_input_file, sep="\t")
    df_jensen_shannon["Metric"] = "Jensen-Shannon"
    logging.info("Data from e: {}, c: {}, and j: {} loaded correctly.".format(os.path.basename(euclidean_input_file), os.path.basename(cosine_input_file), 
                                                                              os.path.basename(jsd_input_file)))
    # Combine data into one DataFrame
    combined_df = pd.concat([df_euclidean, df_cosine, df_jensen_shannon])
    return combined_df

def extract_pattern(filename):
    pattern_match = re.search(r"_downsampled_(.*)\.tsv", filename)
    noise_level_match = re.search(r"_downsampled_(.*?)_d", filename)
    pattern = pattern_match.group(1) if pattern_match else ""
    noise_level = noise_level_match.group(1) if noise_level_match else ""
    return pattern, noise_level

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info("Created directory: {}".format(directory))
    else:
        logging.info("Directory already exists.")

### HISTOGRAMS ###
def hist_plot_sim_performance(combined_df, output_file, noise_level):
    plt.figure(figsize=(12,6))
    metrics = combined_df["Metric"].unique()
    # Calculate bins using data from both metrics 
    all_data = combined_df["Rank"]
    # Get bin edges using all data 
    bins = plt.hist(all_data, bins=50, alpha=0)[1]
    for metric in metrics:
        subset = combined_df[combined_df["Metric"] == metric]
        plt.hist(subset["Rank"], bins=bins, alpha=0.5, label=metric)
    plt.title("Histogram Similarity Metrics Performance (Noise Level: {})".format(noise_level))
    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.ylim((0,1000))
    plt.xticks(range(0, int(max(all_data)+1), 10))
    plt.xlim(0, 200)  # Set x-axis limits to start from 0
    plt.legend(title="Metric")
    plt.savefig(output_file)
    logging.info("Histogram saved successfully to: {}".format(output_file))    

### BOX PLOTS ###
def box_plot_sim_performance(combined_df, output_file, noise_level):
    plt.figure(figsize=(12,6))
    sns.boxplot(x="Metric", y="Rank", data=combined_df)
    plt.title("Box-Plot Similarity Metrics Performance (Noise Level: {})".format(noise_level))
    plt.xlabel("Metric")
    plt.ylabel("Rank")
    plt.savefig(output_file)
    logging.info("Box-plot saved successfully to: {}".format(output_file))

### TOTAL PERFORMANCE ###
def find_files(dir, noise_level):
    files = []
    logging.debug("searching in directory {}".format(dir))
    for root, dirs, filenames in os.walk(dir):
        euclidean_files = [os.path.join(root, f) for f in filenames if noise_level in f and "e_" in f]
        cosine_files = [os.path.join(root, f) for f in filenames if noise_level in f and "c_" in f]
        jsd_files = [os.path.join(root, f) for f in filenames if noise_level in f and "j_" in f]
        if euclidean_files and cosine_files and jsd_files:
            files.append((euclidean_files[0], cosine_files[0], jsd_files[0]))
    return files

def load_and_combine_multiple_data(files):
    euclidean_dfs = []
    cosine_dfs = []
    jsd_dfs = []
    for euclidean_file, cosine_file, jsd_file in files :
        df_euclidean = pd.read_csv(euclidean_file, sep="\t")
        df_euclidean["Metric"] = "Euclidean"
        euclidean_dfs.append(df_euclidean)
        df_cosine = pd.read_csv(cosine_file, sep="\t")
        df_cosine["Metric"] = "Cosine"
        cosine_dfs.append(df_cosine)
        df_jensen_shannon = pd.read_csv(jsd_file, sep="\t")
        df_jensen_shannon["Metric"] = "Jensen-Shannon"
        jsd_dfs.append(df_jensen_shannon)
    combined_df = pd.concat(euclidean_dfs + cosine_dfs + jsd_dfs, ignore_index=True)
    logging.debug("Data loaded and combined correctly. Shape: {}".format(combined_df.shape))
    return combined_df

def box_plot_sim_performance_overall(combined_df, output_file, noise_level):
    plt.figure(figsize=(12,6))
    sns.boxplot(x="Metric", y="Rank", data=combined_df)
    plt.title("Box-Plot Similarity Metrics Performance Noise Level: {}".format(noise_level))
    plt.xlabel("Metric")
    plt.ylabel("Rank")
    plt.ylim((0,4000))
    plt.savefig(output_file)
    logging.info("Overall box-plot saved successfully to: {}".format(output_file))

def hist_plot_sim_performance_overall(combined_df, output_file, noise_level):
    # Drop or handle NaN values in the 'Rank' column
    combined_df = combined_df.dropna(subset=["Rank"])
    plt.figure(figsize=(12,6))
    metrics = combined_df["Metric"].unique()
    # Calculate bins using data from both metrics 
    all_data = combined_df["Rank"]
    # Get bin edges using all data 
    bins = plt.hist(all_data, bins=50, alpha=0)[1]
    # Define color map 
    color_map = {
        "Cosine": "#d7191c",
        "Euclidean": "#2c7bb6", 
        "Jensen-Shannon": "#fdae61"       
    }
    for metric in ["Cosine", "Euclidean", "Jensen-Shannon"]:
        if metric in combined_df["Metric"].values:
            subset = combined_df[combined_df["Metric"] == metric]
            color = color_map.get(metric, "#7f7f7f") # set default to grey, if metric not in color_map
            plt.hist(subset["Rank"], bins=bins, alpha=0.8, label=metric, color=color)
    plt.title("Histogram Similarity Metrics Performance Noise Level: {}".format(noise_level))
    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.ylim((0,4000))
    plt.xticks(range(0, int(max(all_data)+1), 10))
    plt.xlim((0,200))
    plt.legend(title="Metric")
    plt.savefig(output_file)
    logging.info("Overall hist-plot saved successfully to: {}".format(output_file))

def main():
    ### PER VERSION AND NOISE LEVEL PERFORMANCE PLOTS ###
    # Define ranking parent directory of noisy version 
    downsampled_ranking_parent_dir = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSdownsampled_noisy/ranking_output/v5.0_LSU"
    
    ### TO BE ADJUSTED ###
    # Define file names to be compared  
    euclidean_downsampled_file = os.path.join(downsampled_ranking_parent_dir, "ranking_e_initial_VS_downsampled_0.9_ratio_d_v5.0_LSU_ge_filtered.tsv")
    cosine_downsampled_file = os.path.join(downsampled_ranking_parent_dir, "ranking_c_initial_VS_downsampled_0.9_ratio_d_v5.0_LSU_ge_filtered.tsv")
    jsd_downsampled_file = os.path.join(downsampled_ranking_parent_dir, "ranking_j_initial_VS_downsampled_0.9_ratio_d_v5.0_LSU_ge_filtered.tsv")
    ### --- ###
    
    # Extract pattern of interest and noise level from filename
    downsampled_pattern, downsampled_noise_level = extract_pattern(euclidean_downsampled_file)
    # Create subdirectories
    downsampled_plots_dir = os.path.join(downsampled_ranking_parent_dir, "plots")
    # Check if directory exists - if not, create dir 
    ensure_dir(downsampled_plots_dir)
    # Construct file name 
    downsampled_output_file = os.path.join(downsampled_plots_dir, downsampled_pattern) + ".png"
    # Load data
    #ranking_downsampled_noise = load_and_combine_data(euclidean_downsampled_file, cosine_downsampled_file, jsd_downsampled_file)
    # Plot per version & ratio - noise level  
    #hist_plot_sim_performance(ranking_downsampled_noise, downsampled_output_file, downsampled_noise_level)
    
    ### OVERALL PERFORMANCE ### 
    downsampled_directory = "/ccmri/similarity_metrics/data/taxonomic/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSdownsampled_noisy/ranking_output"
    # Define noise level
    downsampling_noise_level = "0.1_ratio"
    # find all relevant files 
    all_downsampled_files = find_files(downsampled_directory, downsampling_noise_level)
    logging.info("Total downsampled files found: {}".format(len(all_downsampled_files)))
    if not all_downsampled_files:
        logging.debug("No downsampled files found. Check patterns and directories.")
    # Load and combine data 
    downsampled_combined_df = load_and_combine_multiple_data(all_downsampled_files)
    # Construct output file path 
    downsampled_output_file = os.path.join(downsampled_directory, "{}_".format(downsampling_noise_level) + "overall_performance.png")
    # Plot overall performance
    hist_plot_sim_performance_overall(downsampled_combined_df, downsampled_output_file, downsampling_noise_level)
    logging.info("Downsampled Overall performance plots saved to: {}".format(downsampled_output_file))
    
if __name__ ==  "__main__":
    main()