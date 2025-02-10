#!/usr/bin/python3.5

# script name: filter_fun_data.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate descriptive stats. 
    # Fitler down data according to stats.     
# framework: CCMRI
# last update: 23/01/2025

import pandas as pd
import matplotlib.pyplot as plt  # For plotting

def calculate_statistics(dataframe, output_txt_file):
    """Calculate descriptive statistics for counts and save results to a .txt file."""
    # Extract counts
    count = dataframe["Count"]

    # Descriptive statistics for counts
    stats = {
        "Min": count.min(),
        "Max": count.max(),
        "Mean": count.mean(),
        "Median": count.median(),
        "Standard Deviation": count.std(),
        "Quantiles": count.quantile([0.25, 0.5, 0.75]).to_dict(),
        "Most Common Value (Mode)": count.mode().values[0] if not count.mode().empty else None,
    }

    # Write statistics to a file
    with open(output_txt_file, "a") as f:
        f.write("\n### Version v5.0 ###\n")
        f.write("\nDescriptive Statistics for Counts:\n")
        for key, value in stats.items():
            if key == "Quantiles":
                f.write("{}:\n".format(key))
                for quantile, q_value in value.items():
                    f.write("  {}%: {}\n".format(quantile * 100, q_value))
            else:
                f.write("{}: {}\n".format(key, value))

    print("Statistics saved to {}".format(output_txt_file))

def calculate_global_frequency(dataframe, output_txt_file):
    """Calculate global frequency of GO terms and save descriptive stats to a .txt file."""
    # Calculate frequency of GO terms across all samples
    go_frequencies = dataframe.groupby("GO")["Run"].nunique()

    # Identify GO terms for specific statistics
    min_freq = go_frequencies.min()
    max_freq = go_frequencies.max()
    mode_freq = go_frequencies.mode().values if not go_frequencies.mode().empty else None

    stats = {
        "Min": {"value": min_freq, "GO_terms": go_frequencies[go_frequencies == min_freq].index.tolist()},
        "Max": {"value": max_freq, "GO_terms": go_frequencies[go_frequencies == max_freq].index.tolist()},
        "Mean": go_frequencies.mean(),
        "Median": go_frequencies.median(),
        "Standard Deviation": go_frequencies.std(),
        "Quantiles": go_frequencies.quantile([0.25, 0.5, 0.75]).to_dict(),
        "Most Common GO Terms (Mode)": {
            "value": mode_freq,
            "GO_terms": go_frequencies[go_frequencies.isin(mode_freq)].index.tolist() if mode_freq is not None else None,
        },
    }

    # Write statistics to the file
    with open(output_txt_file, "a") as f:
        f.write("\nDescriptive Statistics for Global GO Term Frequency:\n")
        for key, value in stats.items():
            if key == "Quantiles":
                f.write("{}:\n".format(key))
                for quantile, q_value in value.items():
                    f.write("  {}%: {}\n".format(quantile * 100, q_value))
            elif key in ["Min", "Max", "Most Common GO Terms (Mode)"]:
                f.write("{}: Value = {}, GO Terms = {}\n".format(
                    key, value["value"], value["GO_terms"]))
            else:
                f.write("{}: {}\n".format(key, value))

    print("Global frequency statistics saved to {}".format(output_txt_file))
    return go_frequencies


def create_histogram(go_frequencies, output_image_file):
    """Create a bar chart of GO terms vs their frequencies."""
    plt.figure(figsize=(12, 8))  # Increased size for better readability

    # Sort GO terms by frequency for a clean bar chart
    go_frequencies_sorted = go_frequencies.sort_values(ascending=False)

    # Create the bar chart
    go_frequencies_sorted.plot(kind="bar", color="skyblue", edgecolor="black")

    # Add labels and titles
    plt.title("GO Terms vs Frequencies", fontsize=16)
    plt.xlabel("GO Terms", fontsize=14)
    plt.ylabel("Frequency (across all runs)", fontsize=14)
    plt.xticks(fontsize=10, rotation=90)  # Rotate x-axis labels for readability
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_image_file)
    plt.close()
    print("Histogram (bar chart) saved to {}".format(output_image_file))


def save_lower_upper_data(raw_data_df, lower_output_file, upper_output_file):
    """
    Filter data into two groups:
    - Below 25% of counts (unchanged).
    - GO terms with a frequency > 75% across all samples.
    Save the results to separate files.
    """
    # Calculate the lower threshold for counts (25th percentile)
    lower_threshold = raw_data_df["Count"].quantile(0.25)
    
    # Calculate the frequency of each GO term across samples
    go_frequencies = raw_data_df.groupby("GO")["Run"].nunique() / raw_data_df["Run"].nunique()

    # Identify GO terms with a frequency > 75%
    high_frequency_terms = go_frequencies[go_frequencies > 0.75].index
    
    # Filter data for values below the 25th percentile of counts
    below_25 = raw_data_df[raw_data_df["Count"] < lower_threshold]
    
    # Filter data for GO terms with a frequency > 75%
    above_75 = raw_data_df[raw_data_df["GO"].isin(high_frequency_terms)]
    
    # Save the filtered data to separate files
    below_25.to_csv(lower_output_file, sep="\t", index=False)
    above_75.to_csv(upper_output_file, sep="\t", index=False)
    
    print("Runs below 25% saved to {}".format(lower_output_file))
    print("GO terms with frequency > 75% saved to {}".format(upper_output_file))


def filter_data_by_custom_criteria(raw_data_df, output_file, freq_threshold=0.75, count_threshold=0.5):
    """
    Filter out data where:
    - Frequency of GO terms > freq_threshold.
    - Count < count_threshold.
    Save the resulting data to a file.
    """
    # Calculate the frequency of each GO term across samples
    go_frequencies = raw_data_df.groupby("GO")["Run"].nunique() / raw_data_df["Run"].nunique()

    # Apply the filtering conditions
    filtered_data = raw_data_df[
        (raw_data_df["GO"].map(go_frequencies) <= freq_threshold) & 
        (raw_data_df["Count"] >= count_threshold)
    ]
    
    # Save the filtered data to the output file (without the frequency column)
    filtered_data.to_csv(output_file, sep="\t", index=False)
    print("Filtered data saved to {}".format(output_file))

def main():
    # Input file with raw data
    input_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/super_table/lf_v5.0_super_table.tsv"
    
    # Output files
    output_txt_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/super_table/stats/counts_statistics_output.txt"
    custom_filtered_data_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/d_filtered/filtered_data_v5.0.tsv"
    
    # Load the raw data
    raw_data_df = pd.read_csv(input_file, sep="\t")
    
    # Calculate statistics for counts and write to file
    calculate_statistics(raw_data_df, output_txt_file)
    
    # Calculate global frequency of GO terms and write to file
    go_frequencies = calculate_global_frequency(raw_data_df, output_txt_file)
    
    # Create and save histogram of GO frequencies
    histogram_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/super_table/stats/go_frequency_v5.0_histogram.png"
    create_histogram(go_frequencies, histogram_file)
    
    # Save data of lower counts and high-frequency GO terms
    below_25_out_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/super_table/stats/v5.0_below_25_Q.tsv"
    above_75_out_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO-slim_abundances/super_table/stats/v5.0_above_75_freq.tsv"
    save_lower_upper_data(raw_data_df, below_25_out_file, above_75_out_file)

    # Apply custom filtering and save the results
    filter_data_by_custom_criteria(raw_data_df, custom_filtered_data_file)

if __name__ == "__main__":
    main()