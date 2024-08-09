#!/usr/bin/python3.5

# script name: pieplot_taxonomy_distribution.py
# developed by: Nefeli Venetsianou
# description: 
    # Calculate counts per taxonomy (k > p > c > o > f > g > s) in each version Super Table and overall.
# framework: CCMRI
# last update: 01/08/2024

import pandas as pd 
import re
import sys, os
from matplotlib import pyplot as plt

def process_file(file_path):
    # Read file
    df = pd.read_csv(file_path, sep="\t")
    # Extract taxa column
    taxa = df["Taxa"]
    # Initialize directory to store taxonomic leve
    counts = {"k__" : set(), "p__" : set(), "c__" : set(), 
              "o__" : set(), "f__": set(), "g__" : set(), "s__" : set()}
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
    return unique_counts, counts

def create_pie_plot(unique_counts, plot_title, output_file_path):
    total_taxa_count = sum(unique_counts.values())
    percentages = {level: round((count / total_taxa_count) * 100, 2) for level, count in unique_counts.items()} 
    # Create pie plot
    fig = plt.figure(figsize=(10,7))
    # Filter out items with zero counts
    non_zero_counts = {level : count for level, count in unique_counts.items() if count != 0}
    # Combine taxonomic labels and counts for display
    labels_with_counts = ["{}:{} ({}%)".format(level, count, percentages[level]) for level, count in non_zero_counts.items()]
    # Custom color paletter to be colorblind safe
    custom_colors = ["#f0f9e8", "#ccebc5", "#a8ddb5", "#7bccc4", "#4eb3d3", "#2b8cbe", "#08589e"]
    # Ensure number of colors matches number of pie segments
    colors = custom_colors[:len(non_zero_counts)] 
    plt.pie(non_zero_counts.values(), colors=colors, labels = [None]*len(non_zero_counts)) # Set labels = labels_with_counts if you prefer labels instead of legend
    # Add legend to avoid overlapping of labels - please comment below row if you prefer labels instead of legend
    plt.legend(labels_with_counts, loc = "center left", bbox_to_anchor=(1, 0.5), fontsize="small" ) 
    plt.title(plot_title)
    plt.savefig(output_file_path)
    plt.close()

def main():
    parent_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/no_filtered"
    output_directory = "/ccmri/similarity_metrics/data/raw_data/lf_raw_super_table/no_filtered/plots"   
    # Initialize directory to store taxonomic level accross all files
    total_counts = {"k__" : set(), "p__" : set(), "c__" : set(), 
                    "o__" : set(), "f__": set(), "g__" : set(), "s__" : set()}
    # Iterate over each .tsv file in parent directory 
    for filename in os.listdir(parent_directory):
        if filename.endswith(".tsv"):
            file_path = os.path.join(parent_directory, filename)
            # Process each file individually 
            unique_counts, file_counts = process_file(file_path)
            # Create individual pie plot per version-pipeline 
            plot_title = "Unique Taxa Counts Distribution Pieplot " + re.sub(r"^lf_", "", filename)[:-16]
            output_filename = re.sub(r"^lf_", "", filename)[:-16] + "_taxonomy_level_pieplot.png"
            output_file_path = os.path.join(output_directory, output_filename)
            #create_pie_plot(unique_counts, plot_title, output_file_path)
            # Accumulate counts for the overall pie plot 
            for level in total_counts:
                total_counts[level].update(file_counts[level])
    # Create overall pie plot 
    overall_unique_counts = {level: len(taxa) for level, taxa in total_counts.items()}
    overall_plot_title = "Overall Unique Taxa Counts Distribution Pieplot"
    overall_output_filename = "overall_taxonomy_distribution_pieplot.png"
    overall_output_file_path = os.path.join(output_directory, overall_output_filename)
    create_pie_plot(overall_unique_counts, overall_plot_title, overall_output_file_path)
    # Print overall counts distribution of unique taxa found 
    for level, count in overall_unique_counts.items():
        print("{}: {}".format(level, count))
    print("Total Unique Taxa Counts: {}".format(sum(overall_unique_counts.values())))
    
if __name__ == "__main__":
    main()