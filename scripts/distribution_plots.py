#!/usr/bin/python3.5

# script name: distribution_plot.py
# developed by: Nefeli Venetsianou
# description: 
    # Plot distribution in taxonomic depth accross all samples per version.
    # Applied in normalized & aggregated data in Super Table (produced by: normalization_counts_ra.py & ra_aggregation.py)
# no input parameters are allowed (for now)
# framework: CCMRI
# last update: 14/05/2024
import sys, os, re
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
# use non-interactive backend
matplotlib.use("Agg")

parent_directory = "/ccmri/similarity_metrics/data/super_table/long_format/"
output_directory = "/ccmri/similarity_metrics/data/super_table/long_format/plots/"

def read_file(filename):
    file_path = os.path.join(parent_directory, filename)
    # check is file exists and is a tsv file
    if not os.path.exists(file_path) or not file_path.endswith(".tsv"):
        print("File not found or not a .tsv file. Please provide a valid file. If you wish, check parent directory '/ccmri/similarity_metrics/data/super_table/long_format/' for available input files.")
        sys.exit(1)
    return pd.read_csv(file_path, sep="\t")
    
def bar_plot_taxa_frequency(df, filename):
    # Calculate frequency (number of occurrences) per unique taxon 
    # value_counts() function in pandas returns a Series containing counts of unique values in a DF column. 
    taxa_frequency = df["Taxa"].value_counts()
    # Get descriptive statistics on taxa frequency 
    taxa_description = taxa_frequency.describe()
    print(taxa_description)
    # Calculate lower quartile - if needed for filtered_taxa_frequency - uncomment below
    #lower_quartile = taxa_frequency.quantile(0.25)
    #med_quartile = taxa_frequency.quantile(0.50)
    #higher_quartile = taxa_frequency.quantile(0.75)
    # Filter out frequencies below 
    # If filtered_taxa_frequency used, please adjust code by replacing taxa_frewuence to filtered_taxa_frequency
    #filtered_taxa_frequency = taxa_frequency[taxa_frequency >= med_quartile] 
    #print("Number of filtered taxa: ", len(filtered_taxa_frequency))
    # Plot histogram 
    plt.figure(figsize=(20,15))
    taxa_frequency.plot(kind="bar")
    plt.xlabel("Taxa")
    plt.ylabel("Total Count")
    # Adjusting title name by filename provided
    plot_title = "Taxa Distribution Plot " + re.sub(r"^lf_", "", filename)[:-16] 
    plt.title(plot_title)
    plt.xticks(rotation=90)
    plt.yticks(range(0, max(taxa_frequency) + 1, 200))
    # Adjusting output filename according to input filename 
    output_filename = re.sub(r"^lf_", "", filename)[:-16] + "_taxonomy_distribution_plot.png"
    output_file_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_file_path)

def pie_plot_taxonomy_distribution(df, filename):
   # Extract taxa column
    taxa = df["Taxa"]
    # Initialize dictionary for each taxonomic level 
    counts = {"k__" : set(), "p__" : set(), "c__" : set(), "o__" : set(), "f__": set(), "g__" : set(), "s__" : set()}
    # Iterate over each taxon 
    for taxon in taxa: 
        # Split taxon by semicolon if necessary 
        sub_taxa = taxon.split(";")
        # Ensure each subtaxon is counted only once - remove duplicates
        unique_sub_taxa = set(sub_taxa)
        for sub in unique_sub_taxa:
            # Extract taxonomic level
            matched_taxa = re.match(r"([kpcofgs])__", sub) # regex used to match any taxonomic level of interest followed by underscores
            if matched_taxa:
                # Define level by grouping according to the single character matched within the regex
                level = matched_taxa.group(1)
                # Extract key to access counts dictionary and add subtaxon to the list
                counts[level + "__"].add(sub) 
    # Count unique taxa for each taxonomic level 
    unique_counts = {level:len(taxa) for level, taxa in counts.items()}
    # Calculate total taxa
    total_taxa_count = sum(unique_counts.values())
    # Calculate percentages per taxonomic group 
    percentages = {level: round((count / total_taxa_count) * 100, 2) for level, count in unique_counts.items()}    
    # Create pie plot
    fig = plt.figure(figsize=(10,7))
    # Filter out items with zero counts
    non_zero_counts = {level : count for level, count in unique_counts.items() if count != 0}
    # Combine taxonomic labels and counts for display
    labels_with_counts = ["{}:{} ({}%)".format(level, count, percentages[level]) for level, count in non_zero_counts.items()]
    plt.pie(non_zero_counts.values(), labels = [None]*len(non_zero_counts)) # Set labels = labels_with_counts if you prefer labels instead of legend
    # Add legend to avoid overlapping of labels - please comment below row if you prefer labels instead of legend
    plt.legend(labels_with_counts, loc = "center left", bbox_to_anchor=(1, 0.5), fontsize="small" )
    # Adjusting title name by filename provided
    plot_title = "Unique Taxa Counts " + re.sub(r"^lf_", "", filename)[:-16] 
    plt.title(plot_title)
    # Adjusting output filename according to input filename 
    output_filename = re.sub(r"^lf_", "", filename)[:-16] + "_taxonomy_counts.png"
    output_file_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_file_path)
    # Print counts of unique taxa found
    for level, count in unique_counts.items():
        print("{}:{}".format(level, count))
    print("Total Unique Taxa Count: {}".format(total_taxa_count))

def main():
    # Add filename from command-line arguments
    if len(sys.argv) < 2:
        print("Please provide the filename (eg. lf_v4.1_LSU_super_table.tsv)")
        sys.exit(1)   
    filename = sys.argv[1]
    df = read_file(filename)
    bar_plot_taxa_frequency(df, filename)
    pie_plot_taxonomy_distribution(df, filename)
    
if __name__ == "__main__":
    main()