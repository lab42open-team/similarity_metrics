#!/usr/bin/python3.5

# script name: avg_plot_ranking_performance_top-K.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate top-X ranks for similarity metrics. 
    # Calculate average ranking per noise level & type in already created ranking files (produced by: ranking_metrics_performance.py)
# framework: CCMRI
# last update: 14/07/2025

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
        jsd_files = [os.path.join(root, f) for f in filenames if noise_level in f and "j_" in f]
        if euclidean_files and cosine_files and jsd_files:
            files.append((euclidean_files[0], cosine_files[0], jsd_files[0]))
    return files

def load_ranking_data(file):
    df = pd.read_csv(file, sep="\t")
    logging.info("Data from {} loaded correctly.".format(format(os.path.basename(file))))
    return df["Rank"]

def calculate_summary_statistics(files):
    euclidean_ranks = []
    cosine_ranks = []
    jsd_ranks = []
    for e_file, c_file, j_file in files:
        euclidean_ranks.extend(load_ranking_data(e_file))
        cosine_ranks.extend(load_ranking_data(c_file))
        jsd_ranks.extend(load_ranking_data(j_file))
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
    jd_counts = calculate_counts(jsd_ranks)
    
    return euclidean_counts, cosine_counts, jd_counts

### Scatter Plot ###
def plot_results(data, output_dir):
    df = pd.DataFrame(data)
    sorted_df = df[["Metric", "Noise_Ratio", "Top-1", "Top-3", "Top-5", "Top-10", "Top-100"]]
    
    # Melt for plotting
    df_melt = pd.melt(
        sorted_df,
        id_vars=["Noise_Ratio", "Metric"],
        var_name="Top-k",
        value_name="Performance"
    )
    # Ensure Noise_Ratio is ordered correctly for plotting
    noise_order = sorted(df["Noise_Ratio"].unique(), key=lambda x: float(re.findall(r"[\d.]+", x)[0]))
    df_melt["Noise_Ratio"] = pd.Categorical(df_melt["Noise_Ratio"], categories=noise_order, ordered=True)
    df_melt = df_melt[df_melt["Top-k"] == "Top-3"]
    # Plot
    plt.figure(figsize=(14, 8))
    # Lineplot: connect the dots per metric over increasing noise
    sns.lineplot(
        data=df_melt,
        x="Noise_Ratio",
        y="Performance",
        hue="Metric",
        #style="Top-k",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=8
    )
    plt.title("Performance vs Noise Level per Metric (Top-3 Scores)")
    plt.suptitle("Taxonomic")
    plt.xlabel("Noise Ratio")
    plt.ylabel("Number of Times Rank <= Top-3")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save plot
    output_file = os.path.join(output_dir, "performance_vs_noise_plot_top-3.png")
    plt.savefig(output_file)
    plt.close()

def save_results(output_file, noise_level, euclidean_counts, cosine_counts, jsd_counts):
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("Noise\tMetric\tTop-1\tTop-3\tTop-5\tTop-10\tTop-100\n")
            #f.write("Noise\tEuclidean Rank\tCosine Rank\n")
    def write_counts(f, noise_level, metric, counts):
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(noise_level, metric, counts["top_1"], counts["top_3"], counts["top_5"],counts["top_10"], counts["top_100"]))
        
    with open(output_file, "a") as f:
        write_counts(f, noise_level, "Euclidean", euclidean_counts)
        write_counts(f, noise_level, "Cosine", cosine_counts)
        write_counts(f, noise_level, "Jensen-Shannon", jsd_counts )

def main():
    # Define output directories
    downsampled_parent_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/ranking_output"
    output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/noise_injection/similarity_metrics/ranking_output/performance"
    output_file_name = "summary_downsampled_statistics.tsv"
    output_file = os.path.join(output_dir, output_file_name)
    # Define noise level
    noise_levels = ["0.1_ratio", "0.25_ratio", "0.75_ratio", "0.9_ratio"]
    summary_data = []
    # Iterate over each noise level
    for noise_level in noise_levels:
        parent_dir = downsampled_parent_dir
        # Find all relevant file
        all_files = find_files(parent_dir, noise_level)
        # Calculate summary statistics 
        euclidean_counts, cosine_counts, jsd_counts = calculate_summary_statistics(all_files)
        # Save results 
        save_results(output_file, noise_level, euclidean_counts, cosine_counts, jsd_counts)
        # Log results 
        logging.info("Average euclidean rank for noise level {} : {}".format(noise_level, euclidean_counts))
        logging.info("Average cosine rank for noise level {} : {}".format(noise_level, cosine_counts))
        logging.info("Summary for noise level {}: Euclidean = {}, Cosine = {}, Jensen-Shannon: {}".format(noise_level, euclidean_counts, cosine_counts, jsd_counts))

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
        }),
        summary_data.append({
            "Noise_Ratio": noise_level,
            "Metric": "Jensen-Shannon", 
            "Top-1": jsd_counts["top_1"],
            "Top-3": jsd_counts["top_3"],
            "Top-5": jsd_counts["top_5"],
            "Top-10": jsd_counts["top_10"],
            "Top-100": jsd_counts["top_100"]
        })

    # Plot results after processing all noise levels
    plot_results(summary_data, output_dir)
    logging.info("Plot saved successfully to: {}".format(output_dir))

if __name__ == "__main__":
    main()