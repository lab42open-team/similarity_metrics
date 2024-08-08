#!/usr/bin/python3.5

# script name: avg_ranking_sim_performance.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate top-X ranks for euclidean and cosine metrics. 
    # Calculate average ranking per noise level & type in already created ranking files (produced by: ranking_metrics_performance.py)
# framework: CCMRI
# last update: 19/07/2024

import logging.config
import os, re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.basicConfig(level=logging.INFO)

def find_files(dir, noise_level):
    files = []
    logging.debug("searching in directory {}".format(dir))
    for root, dirs, filenames in os.walk(dir):
        euclidean_files = [os.path.join(root, f) for f in filenames if noise_level in f and "e_" in f]
        cosine_files = [os.path.join(root, f) for f in filenames if noise_level in f and "c_" in f]
        if euclidean_files and cosine_files:
            files.append((euclidean_files[0], cosine_files[0]))
    return files

def load_ranking_data(file):
    df = pd.read_csv(file, sep="\t")
    logging.info("Data from {} loaded correctly.".format(format(os.path.basename(file))))
    return df["Rank"]

def calculate_summary_statistics(files):
    euclidean_ranks = []
    cosine_ranks = []
    for e_file, c_file in files:
        euclidean_ranks.extend(load_ranking_data(e_file))
        cosine_ranks.extend(load_ranking_data(c_file))
    def calculate_counts(ranks):
        counts = {
            "top_1": sum(1 for r in ranks if r <= 1), # 1 if expression satisfied, 0 if not
            "top_3": sum(1 for r in ranks if r <= 3),
            "top_5": sum(1 for r in ranks if r <= 5),
            "top_10": sum(1 for r in ranks if r <= 10), 
            "top_100": sum(1 for r in ranks if r <= 100)
        }
        return counts
    euclidean_counts = calculate_counts(euclidean_ranks)
    cosine_counts = calculate_counts(cosine_ranks)
    
    return euclidean_counts, cosine_counts

### Line Plot ###
def plot_results(data, output_dir):
    df = pd.DataFrame(data)
    df_melt = pd.melt(df, id_vars=["Noise_Ratio", "Metric"], var_name="Top-k", value_name="Performance")
    plt.figure(figsize=(14,8))
    sns.lineplot(data=df_melt, x="Top-k", y="Performance", hue="Noise_Ratio", style="Metric", markers=True)
    plt.title("Performance vs Top-k Level for Different Noise Ratios and Metrics")
    # Save plot
    output_file = os.path.join(output_dir, "performance_plot.png")
    plt.savefig(output_file)
    plt.close() 

### Bar Plot ###
def plot_bar_subplots(data, output_dir):
    df = pd.DataFrame(data)
    df_melt = pd.melt(df, id_vars=["Noise_Ratio", "Metric"], var_name="Top-k", value_name="Performance") 
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    top_k_levels = ["Top-1", "Top-3", "Top-5", "Top-10", "Top-100"]
    for i, top_k in enumerate(top_k_levels):
        ax = axes[i//3, i%3]  # Arrange plots in grid
        sns.barplot(data=df_melt[df_melt["Top-k"] == top_k], x="Noise_Ratio", y="Performance", hue="Metric", ax=ax)
        ax.set_title("Performance at {}".format(top_k))
        ax.set_xlabel('Noise Ratio')
        ax.set_ylabel('Count')
        if i % 3 != 0:
            ax.set_ylabel('')  # Remove y-labels for better layout
        if i < 3:
            ax.set_xlabel('')  # Remove x-labels for top plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_bar_plot.png"))
    plt.close()

### HeatMap ###
def plot_heatmap(data, output_dir):
    df = pd.DataFrame(data)
    df_melt = pd.melt(df, id_vars=["Noise_Ratio", "Metric"], var_name="Top-k", value_name="Performance")
    df_pivot = df_melt.pivot_table(index=["Noise_Ratio", "Metric"], columns="Top-k", values="Performance")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pivot, annot=True, fmt="d", cmap="Blues", linewidths=.5)
    plt.title("Performance Heatmap for Different Noise Ratios and Metrics")
    plt.xlabel("Top-k Levels")
    plt.ylabel("Noise Ratio & Metric")
    plt.savefig(os.path.join(output_dir, "performance_heatmap.png"))
    plt.close()

"""
def calculate_average_ranking(files):
    euclidean_ranks = []
    cosine_ranks = []
    for e_file, c_file in files:
        euclidean_ranks.extend(load_ranking_data(e_file))
        cosine_ranks.extend(load_ranking_data(c_file))
    avg_euclidean_rank = sum(euclidean_ranks) / len(euclidean_ranks) if euclidean_ranks else float("nan")
    avg_cosine_rank = sum(cosine_ranks) / len(cosine_ranks) if cosine_ranks else float("nan")
    return avg_euclidean_rank, avg_cosine_rank
"""
def save_results(output_file, noise_level, euclidean_counts, cosine_counts):
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("Noise\tMetric\tTop-1\tTop-3\tTop-5\tTop-10\tTop-100\n")
            #f.write("Noise\tEuclidean Rank\tCosine Rank\n")
    def write_counts(f, noise_level, metric, counts):
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(noise_level, metric, counts["top_1"], counts["top_3"], counts["top_5"],counts["top_10"], counts["top_100"]))
        
    with open(output_file, "a") as f:
        write_counts(f, noise_level, "Euclidean", euclidean_counts)
        write_counts(f, noise_level, "Cosine", cosine_counts)

def main():
    # Define output directories
    #gaussian_parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSgaussian_noisy/ranking_output"
    #impulse_parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSimpulse_noisy/ranking_output"
    downsampled_parent_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics/sim_initialVSdownsampled_noisy/ranking_output"
    output_dir = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/filtered_data/genus/noise_injection/similarity_metrics"
    output_file_name = "summary_downsampled_statistics.tsv"
    output_file = os.path.join(output_dir, output_file_name)
    # Define noise level
    #noise_levels = ["0.2_stdDev", "0.5_stdDev", "1_stdDev", "5_stdDev", "0.01_impL", "0.03_impL" ,"0.05_impL", "0.1_impL"]
    noise_levels = ["0.1_ratio", "0.25_ratio", "0.5_ratio", "0.75_ratio", "0.9_ratio"]
    summary_data = []
    # Iterate over each noise level
    for noise_level in noise_levels:
        """
        if "stdDev" in noise_level:
            parent_dir = gaussian_parent_dir
        else:
            parent_dir = impulse_parent_dir
        """
        parent_dir = downsampled_parent_dir
        # Find all relevant file
        all_files = find_files(parent_dir, noise_level)
        # Calculate summary statistics 
        euclidean_counts, cosine_counts = calculate_summary_statistics(all_files)
        # Save results 
        save_results(output_file, noise_level, euclidean_counts, cosine_counts)
        # Log results 
        #logging.info("Average euclidean rank for noise level {} : {}".format(noise_level, euclidean_counts))
        #logging.info("Average cosine rank for noise level {} : {}".format(noise_level, cosine_counts))
        logging.info("Summary for noise level {}: Euclidean = {}, Cosine = {}".format(noise_level, euclidean_counts, cosine_counts))

        # Prepare data for plotting
        summary_data.append({
            "Noise_Ratio": noise_level,
            "Metric": "Euclidean", 
            "Top-1": euclidean_counts["top_1"],
            "Top-3": euclidean_counts["top_3"],
            "Top-5": euclidean_counts["top_5"],
            "Top-10": euclidean_counts["top_10"],
            "Top-100": euclidean_counts["top_100"]
        })
        summary_data.append({
            "Noise_Ratio": noise_level,
            "Metric": "Cosine", 
            "Top-1": cosine_counts["top_1"],
            "Top-3": cosine_counts["top_3"],
            "Top-5": cosine_counts["top_5"],
            "Top-10": cosine_counts["top_10"],
            "Top-100": cosine_counts["top_100"]
        })

    # Plot results after processing all noise levels
    plot_results(summary_data, output_dir)
    plot_heatmap(summary_data, output_dir)
    plot_bar_subplots(summary_data, output_dir)

if __name__ == "__main__":
    main()