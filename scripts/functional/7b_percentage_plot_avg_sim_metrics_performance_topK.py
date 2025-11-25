#!/usr/bin/python3.5

# script name: avg_plot_ranking_performance_top-K.py
# developed by: Nefeli Venetsianou
# description: 
#   Calculate top-X ranks for similarity metrics. 
#   Calculate average ranking per noise level & type in already created ranking files (produced by: ranking_metrics_performance.py)
# framework: CCMRI
# last update: 14/07/2025

import os, re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)

def find_files(dir, noise_level):
    files = []
    logging.debug(f"Searching in directory {dir}")
    for root, _, filenames in os.walk(dir):
        euclidean_files = [os.path.join(root, f) for f in filenames if noise_level in f and "euclidean_" in f]
        cosine_files = [os.path.join(root, f) for f in filenames if noise_level in f and "cosine_" in f]
        jsd_files = [os.path.join(root, f) for f in filenames if noise_level in f and "jsd_" in f]
        if euclidean_files and cosine_files and jsd_files:
            files.append((euclidean_files[0], cosine_files[0], jsd_files[0]))
    return files

def load_ranking_data(file):
    df = pd.read_csv(file, sep="\t")
    logging.info(f"Loaded {len(df)} rows from {os.path.basename(file)}")
    return df["Rank"]

def calculate_summary_statistics(files):
    euclidean_ranks = []
    cosine_ranks = []
    jsd_ranks = []

    for e_file, c_file, j_file in files:
        euclidean_ranks.extend(load_ranking_data(e_file))
        cosine_ranks.extend(load_ranking_data(c_file))
        jsd_ranks.extend(load_ranking_data(j_file))

    total_ranks_e = len(euclidean_ranks)
    total_ranks_c = len(cosine_ranks)
    total_ranks_j = len(jsd_ranks)

    def calculate_counts(ranks, total_ranks):
        return {
            "top_1": 100 * sum(1 for r in ranks if r <= 1) / total_ranks,
            "top_3": 100 * sum(1 for r in ranks if r <= 3) / total_ranks,
            "top_5": 100 * sum(1 for r in ranks if r <= 5) / total_ranks,
            "top_10": 100 * sum(1 for r in ranks if r <= 10) / total_ranks, 
            "top_100": 100 * sum(1 for r in ranks if r <= 100) / total_ranks
        }

    return (
        calculate_counts(euclidean_ranks, total_ranks_e),
        calculate_counts(cosine_ranks, total_ranks_c),
        calculate_counts(jsd_ranks, total_ranks_j)
    )

def save_results(output_file, noise_level, euclidean_counts, cosine_counts, jsd_counts):
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("Noise\tMetric\tTop-1\tTop-3\tTop-5\tTop-10\tTop-100\n")

    def write_counts(f, noise_level, metric, counts):
        f.write(f"{noise_level}\t{metric}\t{counts['top_1']:.2f}\t{counts['top_3']:.2f}\t{counts['top_5']:.2f}\t{counts['top_10']:.2f}\t{counts['top_100']:.2f}\n")

    with open(output_file, "a") as f:
        write_counts(f, noise_level, "Euclidean", euclidean_counts)
        write_counts(f, noise_level, "Cosine", cosine_counts)
        write_counts(f, noise_level, "Jensen-Shannon", jsd_counts)

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

    plt.figure(figsize=(14, 8))
    
    sns.barplot(
        data=df_melt,
        x="Noise_Ratio",
        y="Performance",
        hue="Metric",
        palette="Set2"
    )
    
    plt.title("Performance vs Noise Level per Metric (Top-3 Scores)", fontsize=18)
    plt.suptitle("Functional", fontsize=14)
    plt.xlabel("Noise Ratio", fontsize=14)
    plt.ylabel("Percentage of Times Rank ≤ Top-3 (%)", fontsize=14)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 100)  # percentage y-axis
    plt.legend(title="Metric", fontsize=12, title_fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_file = os.path.join(output_dir, "performance_vs_noise_plot_top-3_bar_percentage.png")
    plt.savefig(output_file)
    plt.close()


def main():
    downsampled_parent_dir = "/home1/nvenet/data/functional/GO_abundances/noise_injection/similarity_metrics/ranking_output"
    output_dir = "/home1/nvenet/data/functional/GO_abundances/noise_injection/similarity_metrics"
    output_file = os.path.join(output_dir, "summary_downsampled_statistics.tsv")

    noise_levels = ["0.1_ratio", "0.25_ratio", "0.75_ratio", "0.9_ratio"]
    summary_data = []

    for noise_level in noise_levels:
        all_files = find_files(downsampled_parent_dir, noise_level)
        if not all_files:
            logging.warning(f"No matching files found for noise level: {noise_level}")
            continue

        euclidean_counts, cosine_counts, jsd_counts = calculate_summary_statistics(all_files)

        save_results(output_file, noise_level, euclidean_counts, cosine_counts, jsd_counts)

        logging.info(f"Summary for {noise_level}:")
        logging.info(f"  Euclidean      : {euclidean_counts}")
        logging.info(f"  Cosine         : {cosine_counts}")
        logging.info(f"  Jensen-Shannon : {jsd_counts}")

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
        summary_data.append({
            "Noise_Ratio": noise_level,
            "Metric": "Jensen-Shannon", 
            "Top-1": jsd_counts["top_1"],
            "Top-3": jsd_counts["top_3"],
            "Top-5": jsd_counts["top_5"],
            "Top-10": jsd_counts["top_10"],
            "Top-100": jsd_counts["top_100"]
        })

    if summary_data:
        plot_results(summary_data, output_dir)
    else:
        logging.warning("No summary data to plot.")

if __name__ == "__main__":
    main()
