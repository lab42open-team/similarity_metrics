#!/usr/bin/python3.5

# script name: filter_fun_data.py
# developed by: Nefeli Venetsianou
# description: 
    # Filter data according to Local and Global Frequency.
        # Local Frequency is defined as reads for GO terms divided by the sum of reads per sample. (Within Sample)
        # Global Frequency is defined as sum of times a term appears accross all samples divided by the number of samples. (Within DataSet)
# input parameters from cmd:
    # upper threshold (e.g. 0.75): terms that are more frequent than 75% (based on Global FR) will be removed.
    # lower threshold (eg. 0.01): terms that are lower than 1% within a sample (based on Local FR) will be removed. 
# framework: CCMRI

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # For plotting

def calculate_statistics(dataframe, output_txt_file):
    """Calculate descriptive statistics for counts and save results to a .txt file."""
    count = dataframe["Count"]
    stats = {
        "Min": count.min(),
        "Max": count.max(),
        "Mean": count.mean(),
        "Median": count.median(),
        "Standard Deviation": count.std(),
        "Quantiles": count.quantile([0.25, 0.5, 0.75]).to_dict(),
        "Most Common Value (Mode)": count.mode().values[0] if not count.mode().empty else None,
    }
    with open(output_txt_file, "a") as f:
        f.write("\n### Descriptive Statistics for Counts ###\n")
        for key, value in stats.items():
            if key == "Quantiles":
                f.write("{}:\n".format(key))
                for quantile, q_value in value.items():
                    f.write("  {}%: {}\n".format(quantile * 100, q_value))
            else:
                f.write("{}: {}\n".format(key, value))
    print("Statistics saved to", output_txt_file)

def calculate_global_frequency(dataframe, output_txt_file):
    """Calculate global frequency of GO terms and save descriptive stats to a .txt file."""
    go_frequencies = dataframe.groupby("GO")["Run"].nunique()
    min_freq, max_freq = go_frequencies.min(), go_frequencies.max()
    mode_freq = go_frequencies.mode().values if not go_frequencies.mode().empty else None
    stats = {
        "Min": {"value": min_freq, "GO_terms": go_frequencies[go_frequencies == min_freq].index.tolist()},
        "Max": {"value": max_freq, "GO_terms": go_frequencies[go_frequencies == max_freq].index.tolist()},
        "Mean": go_frequencies.mean(),
        "Median": go_frequencies.median(),
        "Standard Deviation": go_frequencies.std(),
        "Quantiles": go_frequencies.quantile([0.25, 0.5, 0.75]).to_dict(),
        "Most Common GO Terms (Mode)": {"value": mode_freq, "GO_terms": go_frequencies[go_frequencies.isin(mode_freq)].index.tolist() if mode_freq is not None else None},
    }
    with open(output_txt_file, "a") as f:
        f.write("\n### Descriptive Statistics for Global GO Term Frequency ###\n")
        for key, value in stats.items():
            if key == "Quantiles":
                f.write("{}:\n".format(key))
                for quantile, q_value in value.items():
                    f.write("  {}%: {}\n".format(quantile * 100, q_value))
            elif key in ["Min", "Max", "Most Common GO Terms (Mode)"]:
                f.write("{}: Value = {}, GO Terms = {}\n".format(key, value["value"], value["GO_terms"]))
            else:
                f.write("{}: {}\n".format(key, value))
    print("Global frequency statistics saved to", output_txt_file)
    return go_frequencies

def save_filtered_data(raw_data_df, upper_threshold, lower_threshold, upper_log_file, lower_log_file, filtered_output_file):
    """
    Filter data dynamically based on user-defined thresholds and save the filtered data.
    """
    local_lower_threshold = raw_data_df.groupby("Run")["Count"].transform(lambda x: x.quantile(lower_threshold)) # filter data within a sample
    go_frequencies = raw_data_df.groupby("GO")["Run"].nunique() / raw_data_df["Run"].nunique()
    
    high_frequency_terms = go_frequencies[go_frequencies > upper_threshold].index.tolist() # filter overly ubiquitous GO terms
    below_lower = raw_data_df[raw_data_df["Count"] < local_lower_threshold]
    
    # Identify GO terms to remove
    removed_go_terms = set(high_frequency_terms).union(set(below_lower["GO"].unique()))
    
    # Filter the data by keeping only the GO terms that are not in removed_go_terms
    filtered_data = raw_data_df[~raw_data_df["GO"].isin(removed_go_terms)]
    
    # Count unique GO terms removed
    unique_removed_upper = len(set(high_frequency_terms))
    unique_removed_lower = below_lower["GO"].nunique()
    unique_remaining = filtered_data["GO"].nunique()
    
    # Save logs for removed GO terms
    with open(upper_log_file, "w") as log_file:
        log_file.write(f"### GO Terms Removed (Above {upper_threshold * 100}%) ###\n")
        for go_term in high_frequency_terms:
            log_file.write(go_term + "\n")
    
    with open(lower_log_file, "w") as lower_log:
        lower_log.write(f"### GO Terms Removed (Below {lower_threshold * 100}%) ###\n")
        for go_term in below_lower["GO"].unique().tolist():
            lower_log.write(go_term + "\n")
    
    # Save the filtered data to a TSV file
    filtered_data.to_csv(filtered_output_file, sep="\t", index=False)
    
    print(f"Removed GO terms (high frequency > {upper_threshold * 100}%) logged in {upper_log_file}")
    print(f"Total unique GO terms removed (high frequency): {unique_removed_upper}")
    print(f"Removed GO terms (below {lower_threshold * 100}% count) logged in {lower_log_file}")
    print(f"Total unique GO terms removed (low count): {unique_removed_lower}")
    print(f"Total unique GO terms remaining after filtering: {unique_remaining}")
    print(f"Filtered data saved to {filtered_output_file}")

def compute_tfidf_score(dataframe):
    """
    Compute TF-IDF-like ranking for GO terms across samples.
    Adds a new column 'Ranking (TF-IDF)' to the dataframe.
    """
    # Step 1: Compute Local Frequency (TF)
    dataframe["Local_FR"] = dataframe.groupby("Run")["Count"].transform(lambda x: x / x.sum())
    # Step 2: Compute Inverse Document Frequency (IDF)
    total_samples = dataframe["Run"].nunique()
    go_presence = dataframe.groupby("GO")["Run"].nunique()  # how many samples each GO term appears in
    idf = np.log(total_samples / go_presence)
    # Step 3: Merge IDF back into main dataframe
    dataframe = dataframe.merge(idf.rename("IDF"), on="GO")
    # Step 4: Compute TF-IDF
    dataframe["Ranking (TF-IDF)"] = dataframe["Local_FR"] * dataframe["IDF"]
    print("TF-IDF ranking added to dataframe.")
    return dataframe

def main():
    input_file = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/aggregated_lf_v5.0_super_table.tsv" 
    output_dir = "/ccmri/similarity_metrics/data/functional/raw_data/GO_abundances/aggregated_data/filtered_cutOff/0.75_GL_0.01_LO/v5.0"
    os.makedirs(output_dir, exist_ok=True)
    output_txt_file = os.path.join(output_dir, "stats.txt")
    upper_log_file = os.path.join(output_dir, "upper_log.txt")
    lower_log_file = os.path.join(output_dir, "lower_log.txt")
    filtered_output = os.path.join(output_dir, "filtered_aggregated_lf_v5.0_super_table.tsv")
    
    # User-defined thresholds
    upper_threshold = float(input("Enter upper threshold (e.g., 0.75): "))
    lower_threshold = float(input("Enter lower threshold (e.g., 0.25): "))
    
    raw_data_df = pd.read_csv(input_file, sep="\t")
    calculate_statistics(raw_data_df, output_txt_file)
    go_frequencies = calculate_global_frequency(raw_data_df, output_txt_file)
    save_filtered_data(raw_data_df, upper_threshold, lower_threshold, upper_log_file, lower_log_file, filtered_output)
    # Add TF-IDF style ranking
    raw_data_df = compute_tfidf_score(raw_data_df)
    # Save TF-IDF-enhanced data to separate output file
    tfidf_output_file = os.path.join(output_dir, "aggregated_lf_v5.0_with_tfidf.tsv")
    raw_data_df.to_csv(tfidf_output_file, sep="\t", index=False)
    print(f"TF-IDF ranked data saved to {tfidf_output_file}")
    
if __name__ == "__main__":
    main()