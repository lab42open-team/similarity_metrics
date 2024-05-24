#!/usr/bin/python3.5

# script name: distribution_plots.py
# developed by: Nefeli Venetsianou
# description: 
    # bar_plot_taxa_frequency: creates taxa frequency accross all samples per version 
    # pie_plot_taxonomy_distribution: creates plot total percentage (count) of taxa per version 
    # total_unique_taxonomy_sample_counts: creates plot with unique taxa in sample counts
    # taxa_sample_ra_plot: created plot with total taxa found, the relative abundance and dots representing samples that contain specific taxa.  
    # Applied in normalized data in Super Table (produced by: normalization_counts_ra.py (oxygen) & long_format_super_table.py (zorba hpc))
# Input parameters: filename of super table to be analyzed (eg. lf_v1.0_super_table.tsv)
# framework: CCMRI
# last update: 24/05/2024

import sys, os, re
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
# use non-interactive backend
matplotlib.use("Agg")

#parent_directory = "/ccmri/similarity_metrics/data/lf_super_table/no_filtered/" # long format super table directory 
#output_directory = "/ccmri/similarity_metrics/data/lf_super_table/no_filtered/plots"
parent_directory = "/ccmri/similarity_metrics/data/lf_super_table/phylum_filtered/" # filtered data according to taxonomic depth of choice directory 
output_directory = "/ccmri/similarity_metrics/data/lf_super_table/phylum_filtered/plots/"

def read_file(filename):
    file_path = os.path.join(parent_directory, filename)
    # check is file exists and is a tsv file
    if not os.path.exists(file_path) or not file_path.endswith(".tsv"):
        print("File not found or not a .tsv file. Please provide a valid file. If you wish, check parent directory '{}' for available input files.".format(parent_directory))
        sys.exit(1)
    return pd.read_csv(file_path, sep="\t")
    
def bar_plot_taxa_frequency(df, filename):
    # Calculate frequency (number of occurrences) per unique taxon 
    # value_counts() function in pandas returns a Series containing counts of unique values in a DF column. 
    taxa_frequency = df["Taxa"].value_counts()
    # Get descriptive statistics on taxa frequency 
    taxa_description = taxa_frequency.describe()
    print("Taxa Description: ", taxa_description)
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
    # Adjust yticks step according to max frequency 
    max_frequency = taxa_frequency.max()
    if max_frequency > 10000000:
        yticks_step = 1000000
    elif 1000000 < max_frequency < 10000000:
        yticks_step = 50000
    elif 200000 < max_frequency < 1000000:
        yticks_step = 10000
    else:
        yticks_step = 500
    #yticks_step = 50000 if max_frequency > 1000000 else 500   
    plt.yticks(range(0, max(taxa_frequency) + 1, yticks_step)) 
    # Adjusting output filename according to input filename 
    output_filename = re.sub(r"^lf_", "", filename)[:-16] + "_bar_plot_taxa_frequency.png"
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
    output_filename = re.sub(r"^lf_", "", filename)[:-16] + "_pie_plot_taxa_distribution.png"
    output_file_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_file_path)
    # Print counts of unique taxa found
    for level, count in unique_counts.items():
        print("{}:{}".format(level, count))
    print("Total Unique Taxa Count: {}".format(total_taxa_count))

"""
def total_unique_taxonomy_sample_counts(df, filename):
    # Count total unique taxa occurrences
    unique_taxa_per_sample = df.drop_duplicates(subset=["Sample", "Taxa"])
    taxa_sample_counts = unique_taxa_per_sample.groupby("Sample")["Taxa"].nunique()
    # Plot histogram 
    plt.figure(figsize=(20,15))
    taxa_sample_counts.plot(kind="bar")
    plt.xlabel("Total Sample")
    plt.ylabel("Total Unique Taxa Count")
    # Adjusting title name by filename provided
    plot_title = "Unique Taxonomy Sample Count " + re.sub(r"^lf_", "", filename)[:-16] 
    plt.title(plot_title)
    plt.xticks(rotation=90,  fontsize=12)
    plt.yticks(range(0, max(taxa_sample_counts) + 1, 200))
    # Adjusting output filename according to input filename 
    output_filename = re.sub(r"^lf_", "", filename)[:-16] + "_unique_taxonomy_sample_plot.png"
    output_file_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_file_path)
    print( "Unique Taxonomy Sample Count Description: " , taxa_sample_counts.describe())
"""

def taxa_sample_ra_plot(df, filename):
    plt.figure(figsize=(20,15))
    sns.boxplot(x="Count", y="Taxa", data = df, palette="vlag")
    sns.stripplot(x="Count", y="Taxa", data = df, size = 4, color=".3", linewidth=0)
    plt.xlabel("Relative abundance")
    plt.ylabel("Taxa")
    plot_title = "Taxa Relative Abundance " + re.sub(r"^lf_", "", filename)[:-16] 
    plt.title(plot_title)
    plt.tight_layout()
    output_filename = re.sub(r"^lf_", "", filename)[:-16] + "_taxa_ra_sample_plot.png"
    output_file_path = os.path.join(output_directory, output_filename)
    plt.savefig(output_file_path)       

def main():
    # Add filename from command-line arguments
    if len(sys.argv) < 2:
        print("Please provide the filename (eg. lf_v4.1_LSU_super_table.tsv)")
        sys.exit(1)   
    filename = sys.argv[1]
    df = read_file(filename)
    bar_plot_taxa_frequency(df, filename)
    pie_plot_taxonomy_distribution(df, filename)
    #total_unique_taxonomy_sample_counts(df, filename)
    #taxa_sample_ra_plot(df, filename)
    
if __name__ == "__main__":
    main()