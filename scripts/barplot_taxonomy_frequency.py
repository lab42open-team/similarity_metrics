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
# Add filename from command-line arguments
if len(sys.argv) < 2:
    print("Please provide the filename (eg. lf_v4.1_LSU_super_table.tsv)")
    sys.exit(1)   
filename = sys.argv[1]
file_path = os.path.join(parent_directory, filename)
# check is file exists and is a tsv file
if not os.path.exists(file_path) or not file_path.endswith(".tsv"):
    print("File not found or not a .tsv file. Please provide a valid file. If you wish, check parent directory '/ccmri/similarity_metrics/data/super_table/long_format/' for available input files.")
    sys.exit(1)

# Parse data into a DataFrame 
df = pd.read_csv(file_path, sep="\t")
#print(df)

# Calculate frequency (number of occurrences) per unique taxon 
# value_counts() function in pandas returns a Series containing counts of unique values in a DF column. 
taxa_frequency = df["Taxa"].value_counts()

# Get descriptive statistics on taxa frequency 
taxa_description = taxa_frequency.describe()
print(taxa_description)

# Calculate lower quartile
lower_quartile = taxa_frequency.quantile(0.25)
med_quartile = taxa_frequency.quantile(0.50)
higher_quartile = taxa_frequency.quantile(0.75)
# Filter out frequencies below 
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
#plt.tight_layout()
# Adjusting output filename according to input filename 
output_directory = "/ccmri/similarity_metrics/data/super_table/long_format/plots/"
output_filename = re.sub(r"^lf_", "", filename)[:-16] + "_taxonomy_distribution_plot.png"
output_file_path = os.path.join(output_directory, output_filename)
plt.savefig(output_file_path)


""" 
# Max - min taxa frequency
max_frequency = max(taxa_frequency)
min_frequency = min(taxa_frequency)
# Get corresponding taxa max - min frequency
taxa_max_frequency = taxa_frequency[taxa_frequency == max_frequency]
taxa_min_frequency = taxa_frequency[taxa_frequency == min_frequency]

# Print descriptive statistics
print("Max f:", max(taxa_frequency), "Taxon max f: ", taxa_max_frequency)
print("Min f:", min(taxa_frequency), "Taxon min f: ", taxa_min_frequency)
print("Mean f: ", mean(taxa_frequency))
print("Median f: ", median(taxa_frequency))
print("Variance f: ", variance(taxa_frequency))
print("Standard deviation f: ", stdev(taxa_frequency))

# Calculate mean, standard deviation
mean_freq = np.mean(taxa_frequency)
std_dev = np.std(taxa_frequency)
# Define threshold as percentage of mean (10% mean)
threshold = 0.05 * mean_freq
# Calculate z-score
z_scores = [(x - mean_freq)/std_dev for x in taxa_frequency]
# Find frequencies with z-scores extending threshold
outliers = [freq for freq, z_score in zip(taxa_frequency, z_scores) if z_score > threshold]
print(outliers, "Total outliers: ", len(outliers))  
"""
